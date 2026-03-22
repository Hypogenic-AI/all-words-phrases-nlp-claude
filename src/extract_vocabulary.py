"""
Extract and characterize implicit vocabulary items from an LLM.

Approach: Instead of using pre-trained probes (which require gated Llama models),
we use a probe-free method based on:
1. Layer-wise representation shift: How much does the hidden state at the last token
   position of a span change from early layers to late layers? Non-compositional
   phrases show distinctive patterns.
2. Compositional deviation: Compare the contextual embedding of a phrase to what
   would be predicted from composing its constituent embeddings.
3. Token co-occurrence statistics (PMI) as a frequency baseline.

Model: Mistral-7B-v0.1 (ungated, same hidden_size=4096 as Llama-2-7b)
"""

import os
import sys
import csv
import json
import time
import torch
import random
import numpy as np
import pandas as pd
import spacy
from collections import Counter, defaultdict
from transformers import AutoTokenizer, AutoModelForCausalLM
from datetime import datetime

# ─── Configuration ───────────────────────────────────────────────────────────

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(WORKSPACE, 'code', 'footprints', 'data')
RESULTS_DIR = os.path.join(WORKSPACE, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

MODEL_PATH = 'mistralai/Mistral-7B-v0.1'
WINDOW_SIZE = 256
EARLY_LAYER = 1
LATE_LAYER = 24  # Mistral has 32 layers; layer 24 is ~75%
MAX_SPAN_LEN = 6
N_DOCS = 150

# ─── Helper functions ────────────────────────────────────────────────────────

def cosine_sim(a, b):
    """Cosine similarity between two tensors."""
    return torch.nn.functional.cosine_similarity(a, b, dim=-1).item()


def load_model():
    """Load Mistral-7B with efficient settings."""
    print(f"Loading model: {MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        device_map='cuda:0',
        output_hidden_states=True,
    )
    model.eval()
    print(f"Model loaded. Vocab size: {tokenizer.vocab_size}")
    return model, tokenizer


def get_hidden_states(model, tokenizer, text, max_length=WINDOW_SIZE):
    """Get hidden states at early and late layers for all token positions."""
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=max_length)
    input_ids = inputs['input_ids'].to(model.device)

    with torch.no_grad():
        outputs = model(**inputs.to(model.device))

    # outputs.hidden_states: tuple of (n_layers+1) tensors, each (1, seq_len, hidden_size)
    early_states = outputs.hidden_states[EARLY_LAYER][0]  # (seq_len, hidden_size)
    late_states = outputs.hidden_states[LATE_LAYER][0]     # (seq_len, hidden_size)
    final_states = outputs.hidden_states[-1][0]            # (seq_len, hidden_size)

    token_ids = input_ids[0].cpu().tolist()
    return token_ids, early_states, late_states, final_states


# ─── Erasure-like score (probe-free) ────────────────────────────────────────

def compute_span_scores(token_ids, early_states, late_states, final_states, tokenizer):
    """
    Compute erasure-like scores for all spans up to MAX_SPAN_LEN.

    For a span [i, j], we compute:
    1. representation_shift: cosine distance between early and late layer states
       at the last token of the span, normalized by the mean shift for unigrams.
    2. compositional_deviation: 1 - cosine_sim(span_representation, mean_of_constituents)
       using final layer states.
    3. combined_score: weighted combination

    Intuition: If the model treats tokens [i..j] as a single unit, the representation
    at position j in late layers will diverge more from early layers (the model is
    "erasing" individual token info and replacing it with a merged representation).
    Additionally, the span representation will deviate from a simple average of
    its constituents.
    """
    n = len(token_ids)
    spans = []

    # Compute baseline: mean representation shift for unigrams
    unigram_shifts = []
    for i in range(n):
        shift = 1.0 - cosine_sim(early_states[i:i+1], late_states[i:i+1])
        unigram_shifts.append(shift)
    mean_unigram_shift = np.mean(unigram_shifts)
    std_unigram_shift = np.std(unigram_shifts) + 1e-8

    for span_len in range(2, MAX_SPAN_LEN + 1):
        for i in range(n - span_len + 1):
            j = i + span_len - 1  # last position

            # 1. Representation shift at last position (normalized)
            last_shift = 1.0 - cosine_sim(early_states[j:j+1], late_states[j:j+1])
            norm_shift = (last_shift - mean_unigram_shift) / std_unigram_shift

            # 2. Compositional deviation
            # Mean of constituent final-layer states
            constituent_mean = final_states[i:j+1].mean(dim=0, keepdim=True)
            # Representation at last position (final layer)
            span_repr = final_states[j:j+1]
            comp_dev = 1.0 - cosine_sim(span_repr, constituent_mean)

            # 3. Internal coherence: similarity between first and last token states
            # at late layers (non-compositional spans should show more coherence)
            if span_len > 1:
                internal_sim = cosine_sim(late_states[i:i+1], late_states[j:j+1])
            else:
                internal_sim = 1.0

            # Combined score
            combined = 0.4 * norm_shift + 0.4 * comp_dev + 0.2 * (1.0 - internal_sim)

            text = tokenizer.decode(token_ids[i:j+1])
            spans.append({
                'start': i,
                'end': j,
                'text': text.strip(),
                'token_ids': token_ids[i:j+1],
                'n_tokens': span_len,
                'repr_shift': last_shift,
                'norm_shift': norm_shift,
                'comp_deviation': comp_dev,
                'internal_sim': internal_sim,
                'combined_score': combined,
            })

    return spans


def greedy_segmentation(spans, n_tokens):
    """
    Greedy non-overlapping segmentation: pick highest-scoring spans first,
    then fill gaps with unigrams.
    """
    # Sort by combined_score descending
    sorted_spans = sorted(spans, key=lambda x: -x['combined_score'])

    covered = set()
    selected = []

    for span in sorted_spans:
        positions = set(range(span['start'], span['end'] + 1))
        if not positions & covered:
            selected.append(span)
            covered |= positions

    return selected


# ─── NER baseline ────────────────────────────────────────────────────────────

def get_ner_entities(text, nlp):
    """Extract named entities using spaCy."""
    doc = nlp(text)
    entities = []
    for ent in doc.ents:
        entities.append({
            'text': ent.text,
            'label': ent.label_,
            'start_char': ent.start_char,
            'end_char': ent.end_char,
        })
    return entities


# ─── Main Pipeline ───────────────────────────────────────────────────────────

def main():
    print(f"{'='*60}")
    print("Implicit Vocabulary Extraction Pipeline")
    print(f"{'='*60}")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Seed: {SEED}")
    print(f"Model: {MODEL_PATH}")
    print(f"Early layer: {EARLY_LAYER}, Late layer: {LATE_LAYER}")

    model, tokenizer = load_model()
    nlp = spacy.load('en_core_web_sm')

    # Load Wikipedia dataset
    wiki_path = os.path.join(DATA_DIR, 'wikipedia_test_500.csv')
    dataset = pd.read_csv(wiki_path)
    n_docs = min(N_DOCS, len(dataset))
    print(f"\nProcessing {n_docs} Wikipedia documents...")

    all_spans = []
    all_ner = []
    doc_stats = []

    tik = time.time()
    for doc_idx in range(n_docs):
        doc_text = str(dataset.iloc[doc_idx]['text'])

        # Get hidden states
        token_ids, early_states, late_states, final_states = get_hidden_states(
            model, tokenizer, doc_text
        )

        # Compute span scores
        spans = compute_span_scores(token_ids, early_states, late_states, final_states, tokenizer)

        # Greedy segmentation
        selected = greedy_segmentation(spans, len(token_ids))
        multi_token = [s for s in selected if s['n_tokens'] > 1]

        # NER baseline
        ner_ents = get_ner_entities(doc_text[:1000], nlp)  # limit text length for spaCy

        for s in multi_token:
            s['doc_idx'] = doc_idx
        all_spans.extend(multi_token)

        for e in ner_ents:
            e['doc_idx'] = doc_idx
        all_ner.extend(ner_ents)

        doc_stats.append({
            'doc_idx': doc_idx,
            'n_tokens': len(token_ids),
            'n_spans_computed': len(spans),
            'n_selected_multi': len(multi_token),
            'n_ner_entities': len(ner_ents),
        })

        # Free memory
        del early_states, late_states, final_states
        torch.cuda.empty_cache()

        if (doc_idx + 1) % 25 == 0:
            elapsed = time.time() - tik
            rate = (doc_idx + 1) / elapsed * 60
            print(f"  Doc {doc_idx + 1}/{n_docs} ({rate:.1f} docs/min, "
                  f"{len(all_spans)} spans so far)")

    elapsed = time.time() - tik
    print(f"\nProcessed {n_docs} docs in {elapsed/60:.1f} minutes")

    # Aggregate vocabulary
    vocab_counter = Counter()
    vocab_scores = defaultdict(list)
    for s in all_spans:
        text = s['text']
        vocab_counter[text] += 1
        vocab_scores[text].append(s['combined_score'])

    vocab_items = []
    for text, count in vocab_counter.items():
        scores = vocab_scores[text]
        vocab_items.append({
            'text': text,
            'count': count,
            'mean_combined_score': np.mean(scores),
            'std_combined_score': np.std(scores) if len(scores) > 1 else 0,
            'max_combined_score': np.max(scores),
            'mean_comp_deviation': np.mean([s['comp_deviation'] for s in all_spans if s['text'] == text]),
            'mean_norm_shift': np.mean([s['norm_shift'] for s in all_spans if s['text'] == text]),
            'n_tokens': len(text.split()),  # approximate
        })

    # Save results
    spans_df = pd.DataFrame(all_spans)
    # Remove token_ids list for CSV compatibility
    spans_df_save = spans_df.drop(columns=['token_ids'], errors='ignore')
    spans_df_save.to_csv(os.path.join(RESULTS_DIR, 'all_segments.csv'), index=False)

    vocab_df = pd.DataFrame(vocab_items).sort_values('mean_combined_score', ascending=False)
    vocab_df.to_csv(os.path.join(RESULTS_DIR, 'implicit_vocabulary.csv'), index=False)

    ner_df = pd.DataFrame(all_ner)
    ner_df.to_csv(os.path.join(RESULTS_DIR, 'ner_entities.csv'), index=False)

    doc_stats_df = pd.DataFrame(doc_stats)
    doc_stats_df.to_csv(os.path.join(RESULTS_DIR, 'doc_stats.csv'), index=False)

    # Save config
    config = {
        'seed': SEED,
        'model': MODEL_PATH,
        'n_docs': n_docs,
        'window_size': WINDOW_SIZE,
        'early_layer': EARLY_LAYER,
        'late_layer': LATE_LAYER,
        'max_span_len': MAX_SPAN_LEN,
        'timestamp': datetime.now().isoformat(),
        'elapsed_minutes': elapsed / 60,
        'total_unique_items': len(vocab_df),
        'total_segment_instances': len(spans_df),
        'total_ner_entities': len(ner_df),
    }
    with open(os.path.join(RESULTS_DIR, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    # Print summary
    print(f"\n{'='*60}")
    print("EXTRACTION SUMMARY")
    print(f"{'='*60}")
    print(f"Unique multi-token vocabulary items: {len(vocab_df)}")
    print(f"Total segment instances: {len(spans_df)}")
    print(f"NER entities found: {len(ner_df)}")
    if len(spans_df) > 0:
        print(f"Mean combined score: {spans_df['combined_score'].mean():.4f}")
        print(f"Median combined score: {spans_df['combined_score'].median():.4f}")
    print(f"\nTop 30 vocabulary items by mean combined score:")
    for _, row in vocab_df.head(30).iterrows():
        print(f"  {repr(row['text']):50s}  score={row['mean_combined_score']:.4f}  (n={row['count']})")

    print(f"\nResults saved to {RESULTS_DIR}/")


if __name__ == '__main__':
    main()
