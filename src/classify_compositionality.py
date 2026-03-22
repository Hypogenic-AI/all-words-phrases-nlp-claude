"""
Classify extracted vocabulary items by compositionality using:
1. LLM-based classification (GPT-4.1 via OpenAI API)
2. NER overlap analysis
3. Statistical analysis of score distributions

This script reads the extracted vocabulary from results/ and adds
compositionality labels and analysis.
"""

import os
import json
import time
import random
import numpy as np
import pandas as pd
from openai import OpenAI
from collections import Counter

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(WORKSPACE, 'results')

# ─── LLM Classification ─────────────────────────────────────────────────────

CLASSIFICATION_PROMPT = """You are a linguist classifying multi-token expressions by their compositionality.

For each expression, provide:
1. "category": One of: named_entity, idiom, technical_term, compound_word, fixed_expression, compositional_phrase, fragment
   - named_entity: proper nouns (people, places, organizations) e.g. "New York", "United Nations"
   - idiom: non-literal figurative expressions e.g. "red tape", "kick the bucket"
   - technical_term: domain-specific terminology e.g. "machine learning", "black hole"
   - compound_word: conventionalized multi-word items e.g. "ice cream", "hot dog"
   - fixed_expression: conventional but meaningful phrases e.g. "of course", "as well as"
   - compositional_phrase: meaning derives from parts e.g. "red car", "big house"
   - fragment: not a meaningful unit e.g. "the and", "ing of"
2. "compositionality": Float 0.0-1.0 where 0=fully non-compositional, 1=fully compositional
   - 0.0: meaning cannot be predicted from parts (e.g. "red herring")
   - 0.3: partly predictable but significant non-compositional meaning (e.g. "black market")
   - 0.5: metaphorical extension of literal meaning (e.g. "dark horse")
   - 0.7: mostly compositional with some conventionalization (e.g. "swimming pool")
   - 1.0: fully predictable from parts (e.g. "blue sky")
3. "confidence": Float 0.0-1.0 for your confidence in the classification

Respond ONLY with a JSON array. Each element should be:
{"idx": <index>, "category": "<category>", "compositionality": <float>, "confidence": <float>}

Expressions to classify:
"""


def classify_batch(client, expressions, batch_idx=0):
    """Classify a batch of expressions using GPT-4.1."""
    # Format expressions
    expr_list = "\n".join([f"{i}: \"{expr}\"" for i, expr in enumerate(expressions)])

    try:
        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {"role": "system", "content": "You are a precise linguist. Always respond with valid JSON only."},
                {"role": "user", "content": CLASSIFICATION_PROMPT + expr_list}
            ],
            temperature=0.1,
            max_tokens=4000,
        )
        content = response.choices[0].message.content.strip()
        # Parse JSON - handle potential markdown wrapping
        if content.startswith('```'):
            content = content.split('\n', 1)[1].rsplit('```', 1)[0]
        results = json.loads(content)
        return results
    except Exception as e:
        print(f"  Batch {batch_idx} error: {e}")
        return []


def classify_all(vocab_df, n_samples=500, batch_size=25):
    """Classify a stratified sample of vocabulary items."""
    client = OpenAI()

    # Stratified sample: take items across the score distribution
    vocab_sorted = vocab_df.sort_values('mean_combined_score', ascending=False).reset_index(drop=True)
    n_total = len(vocab_sorted)
    n_samples = min(n_samples, n_total)

    # Sample indices evenly across the distribution
    indices = np.linspace(0, n_total - 1, n_samples, dtype=int)
    sample = vocab_sorted.iloc[indices].copy()
    sample = sample.reset_index(drop=True)

    print(f"Classifying {len(sample)} expressions using GPT-4.1...")

    all_results = []
    expressions = sample['text'].tolist()

    for i in range(0, len(expressions), batch_size):
        batch = expressions[i:i + batch_size]
        results = classify_batch(client, batch, batch_idx=i // batch_size)

        # Map results back
        for r in results:
            idx = r.get('idx', 0)
            global_idx = i + idx
            if global_idx < len(expressions):
                r['global_idx'] = global_idx
                r['text'] = expressions[global_idx]
                all_results.append(r)

        if (i // batch_size + 1) % 5 == 0:
            print(f"  Classified {min(i + batch_size, len(expressions))}/{len(expressions)}")
        time.sleep(0.5)  # Rate limiting

    print(f"Successfully classified {len(all_results)} expressions")
    return sample, all_results


# ─── NER Overlap Analysis ───────────────────────────────────────────────────

def analyze_ner_overlap(vocab_df, ner_df):
    """Check which vocabulary items overlap with NER entities."""
    ner_texts = set(ner_df['text'].str.strip().str.lower()) if len(ner_df) > 0 else set()

    vocab_df = vocab_df.copy()
    vocab_df['is_ner'] = vocab_df['text'].str.strip().str.lower().apply(
        lambda x: any(x in ner or ner in x for ner in ner_texts)
    )
    return vocab_df


# ─── Random Baseline ────────────────────────────────────────────────────────

def generate_random_baseline(segments_df, n_random=500):
    """Generate random contiguous bigrams/trigrams as baseline."""
    # Get all documents and generate random spans
    random_spans = []
    docs = segments_df.groupby('doc_idx')

    for doc_idx, group in docs:
        n_tokens = group['n_tokens'].iloc[0] if 'n_tokens' in group.columns else 3
        for _ in range(5):  # 5 random spans per doc
            span_len = random.choice([2, 3])
            start = random.randint(0, max(0, 200 - span_len))
            random_spans.append({
                'doc_idx': doc_idx,
                'type': 'random',
                'n_tokens': span_len,
            })

    return random_spans[:n_random]


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    print("="*60)
    print("Compositionality Classification Pipeline")
    print("="*60)

    # Load extracted data
    vocab_df = pd.read_csv(os.path.join(RESULTS_DIR, 'implicit_vocabulary.csv'))
    segments_df = pd.read_csv(os.path.join(RESULTS_DIR, 'all_segments.csv'))
    ner_df = pd.read_csv(os.path.join(RESULTS_DIR, 'ner_entities.csv'))

    print(f"Loaded {len(vocab_df)} unique vocabulary items")
    print(f"Loaded {len(segments_df)} segment instances")
    print(f"Loaded {len(ner_df)} NER entities")

    # Filter out very short/noisy items
    vocab_filtered = vocab_df[
        (vocab_df['text'].str.len() > 3) &  # At least 4 chars
        (~vocab_df['text'].str.match(r'^[\s\W]+$'))  # Not just whitespace/punctuation
    ].copy()
    print(f"After filtering: {len(vocab_filtered)} items")

    # NER overlap
    vocab_filtered = analyze_ner_overlap(vocab_filtered, ner_df)
    n_ner = vocab_filtered['is_ner'].sum()
    print(f"Items overlapping with NER: {n_ner} ({100*n_ner/len(vocab_filtered):.1f}%)")

    # LLM Classification
    sample, classifications = classify_all(vocab_filtered, n_samples=500, batch_size=25)

    # Merge classifications back
    class_df = pd.DataFrame(classifications)
    if len(class_df) > 0:
        # Join on text
        sample_classified = sample.merge(
            class_df[['text', 'category', 'compositionality', 'confidence']],
            on='text', how='left'
        )
    else:
        sample_classified = sample.copy()
        sample_classified['category'] = 'unknown'
        sample_classified['compositionality'] = 0.5
        sample_classified['confidence'] = 0.0

    # Save classified results
    sample_classified.to_csv(os.path.join(RESULTS_DIR, 'classified_vocabulary.csv'), index=False)
    vocab_filtered.to_csv(os.path.join(RESULTS_DIR, 'vocabulary_with_ner.csv'), index=False)

    # Print summary
    print(f"\n{'='*60}")
    print("CLASSIFICATION SUMMARY")
    print(f"{'='*60}")

    if 'category' in sample_classified.columns:
        cat_counts = sample_classified['category'].value_counts()
        print("\nCategory distribution:")
        for cat, count in cat_counts.items():
            pct = 100 * count / len(sample_classified)
            print(f"  {cat:25s}: {count:4d} ({pct:.1f}%)")

        print(f"\nMean compositionality score: {sample_classified['compositionality'].mean():.3f}")
        print(f"Median compositionality score: {sample_classified['compositionality'].median():.3f}")

        # Score by category
        print("\nMean compositionality by category:")
        for cat in cat_counts.index:
            cat_data = sample_classified[sample_classified['category'] == cat]
            print(f"  {cat:25s}: {cat_data['compositionality'].mean():.3f} "
                  f"(erasure={cat_data['mean_combined_score'].mean():.3f})")

    # Also classify random baseline for comparison
    print("\nClassifying random bigrams/trigrams as baseline...")
    # For the random baseline, we'll sample from low-scoring spans
    all_segs = pd.read_csv(os.path.join(RESULTS_DIR, 'all_segments.csv'))
    low_score = all_segs.nsmallest(500, 'combined_score')
    low_texts = low_score['text'].dropna().unique()[:200].tolist()

    if len(low_texts) > 0:
        client = OpenAI()
        baseline_results = []
        for i in range(0, min(200, len(low_texts)), 25):
            batch = low_texts[i:i+25]
            results = classify_batch(client, batch, batch_idx=i//25)
            for r in results:
                idx = r.get('idx', 0)
                global_idx = i + idx
                if global_idx < len(low_texts):
                    r['text'] = low_texts[global_idx]
                    r['type'] = 'low_score_baseline'
                    baseline_results.append(r)
            time.sleep(0.5)

        baseline_df = pd.DataFrame(baseline_results)
        baseline_df.to_csv(os.path.join(RESULTS_DIR, 'baseline_classifications.csv'), index=False)

        if len(baseline_df) > 0 and 'compositionality' in baseline_df.columns:
            print(f"\nBaseline (low-score spans) compositionality: {baseline_df['compositionality'].mean():.3f}")
            print(f"High-score vocabulary compositionality: {sample_classified['compositionality'].mean():.3f}")

    print(f"\nResults saved to {RESULTS_DIR}/")


if __name__ == '__main__':
    main()
