"""
Comprehensive analysis and visualization of implicit vocabulary results.
Generates all figures and statistical tests for the research report.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from collections import Counter

WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(WORKSPACE, 'results')
PLOT_DIR = os.path.join(RESULTS_DIR, 'plots')
os.makedirs(PLOT_DIR, exist_ok=True)

plt.rcParams.update({
    'font.size': 11,
    'figure.figsize': (10, 6),
    'figure.dpi': 150,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})


def load_data():
    """Load all results."""
    vocab = pd.read_csv(os.path.join(RESULTS_DIR, 'implicit_vocabulary.csv'))
    segments = pd.read_csv(os.path.join(RESULTS_DIR, 'all_segments.csv'))
    classified = pd.read_csv(os.path.join(RESULTS_DIR, 'classified_vocabulary.csv'))
    ner = pd.read_csv(os.path.join(RESULTS_DIR, 'ner_entities.csv'))
    doc_stats = pd.read_csv(os.path.join(RESULTS_DIR, 'doc_stats.csv'))
    vocab_ner = pd.read_csv(os.path.join(RESULTS_DIR, 'vocabulary_with_ner.csv'))

    baseline_path = os.path.join(RESULTS_DIR, 'baseline_classifications.csv')
    baseline = pd.read_csv(baseline_path) if os.path.exists(baseline_path) else pd.DataFrame()

    config = json.load(open(os.path.join(RESULTS_DIR, 'config.json')))
    return vocab, segments, classified, ner, doc_stats, vocab_ner, baseline, config


def plot_score_distributions(segments, classified):
    """Plot distributions of scores."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Overall combined score distribution
    ax = axes[0, 0]
    ax.hist(segments['combined_score'], bins=50, color='steelblue', alpha=0.7, edgecolor='black', linewidth=0.3)
    ax.axvline(segments['combined_score'].median(), color='red', linestyle='--', label=f'Median={segments["combined_score"].median():.3f}')
    ax.set_xlabel('Combined Score')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Combined Scores (All Spans)')
    ax.legend()

    # 2. Score by span length
    ax = axes[0, 1]
    for n in sorted(segments['n_tokens'].unique()):
        if n <= 5:
            data = segments[segments['n_tokens'] == n]['combined_score']
            ax.hist(data, bins=30, alpha=0.5, label=f'{n}-gram (n={len(data)})')
    ax.set_xlabel('Combined Score')
    ax.set_ylabel('Count')
    ax.set_title('Score Distribution by Span Length')
    ax.legend()

    # 3. Compositional deviation vs representation shift
    ax = axes[1, 0]
    ax.scatter(segments['comp_deviation'], segments['norm_shift'],
               alpha=0.1, s=5, c='steelblue')
    ax.set_xlabel('Compositional Deviation')
    ax.set_ylabel('Normalized Representation Shift')
    ax.set_title('Compositional Deviation vs Representation Shift')

    # 4. Score components correlation
    ax = axes[1, 1]
    corr_data = segments[['norm_shift', 'comp_deviation', 'internal_sim', 'combined_score']].corr()
    sns.heatmap(corr_data, annot=True, fmt='.2f', cmap='RdBu_r', center=0, ax=ax,
                xticklabels=['Norm Shift', 'Comp Dev', 'Internal Sim', 'Combined'],
                yticklabels=['Norm Shift', 'Comp Dev', 'Internal Sim', 'Combined'])
    ax.set_title('Score Component Correlations')

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, 'score_distributions.png'))
    plt.close()
    print("  Saved score_distributions.png")


def plot_category_analysis(classified, baseline):
    """Plot compositionality analysis by category."""
    # Filter out items without classification
    clf = classified.dropna(subset=['category']).copy()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Category distribution (pie chart)
    ax = axes[0, 0]
    cat_counts = clf['category'].value_counts()
    colors = sns.color_palette('Set2', len(cat_counts))
    wedges, texts, autotexts = ax.pie(cat_counts.values, labels=cat_counts.index,
                                       autopct='%1.1f%%', colors=colors, pctdistance=0.85)
    for text in texts:
        text.set_fontsize(9)
    for autotext in autotexts:
        autotext.set_fontsize(8)
    ax.set_title('Category Distribution of Implicit Vocabulary')

    # 2. Compositionality by category (box plot)
    ax = axes[0, 1]
    order = clf.groupby('category')['compositionality'].mean().sort_values().index
    meaningful = clf[clf['category'] != 'fragment']
    if len(meaningful) > 0:
        sns.boxplot(data=meaningful, x='compositionality', y='category', order=[c for c in order if c != 'fragment'],
                    ax=ax, palette='Set2')
    ax.set_xlabel('Compositionality Score (0=non-compositional, 1=compositional)')
    ax.set_title('Compositionality by Category (excl. fragments)')

    # 3. Combined score by category
    ax = axes[1, 0]
    if len(meaningful) > 0:
        sns.boxplot(data=meaningful, x='mean_combined_score', y='category',
                    order=[c for c in order if c != 'fragment'],
                    ax=ax, palette='Set2')
    ax.set_xlabel('Mean Combined Score (Erasure-like)')
    ax.set_title('Erasure-like Score by Category (excl. fragments)')

    # 4. Compositionality spectrum
    ax = axes[1, 1]
    ax.hist(clf['compositionality'], bins=20, color='steelblue', alpha=0.7,
            edgecolor='black', linewidth=0.3, label='All items', density=True)
    if len(baseline) > 0 and 'compositionality' in baseline.columns:
        ax.hist(baseline['compositionality'].dropna(), bins=20, color='coral', alpha=0.5,
                edgecolor='black', linewidth=0.3, label='Low-score baseline', density=True)
    ax.set_xlabel('Compositionality Score')
    ax.set_ylabel('Density')
    ax.set_title('Compositionality Spectrum: Vocabulary vs Baseline')
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, 'category_analysis.png'))
    plt.close()
    print("  Saved category_analysis.png")


def plot_ner_analysis(vocab_ner, segments):
    """Analyze overlap between implicit vocabulary and NER."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 1. NER overlap bar chart
    ax = axes[0]
    ner_counts = vocab_ner['is_ner'].value_counts()
    labels = ['Not NER', 'NER Match']
    values = [ner_counts.get(False, 0), ner_counts.get(True, 0)]
    bars = ax.bar(labels, values, color=['steelblue', 'coral'])
    ax.set_ylabel('Count')
    ax.set_title('Implicit Vocabulary: NER Overlap')
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 50,
                f'{val}', ha='center', va='bottom', fontsize=10)

    # 2. Score distribution: NER vs non-NER
    ax = axes[1]
    ner_items = vocab_ner[vocab_ner['is_ner'] == True]['mean_combined_score']
    non_ner = vocab_ner[vocab_ner['is_ner'] == False]['mean_combined_score']
    ax.hist(ner_items, bins=30, alpha=0.5, label=f'NER match (n={len(ner_items)})',
            color='coral', density=True)
    ax.hist(non_ner, bins=30, alpha=0.5, label=f'Not NER (n={len(non_ner)})',
            color='steelblue', density=True)
    ax.set_xlabel('Mean Combined Score')
    ax.set_ylabel('Density')
    ax.set_title('Score Distribution: NER vs Non-NER Items')
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, 'ner_analysis.png'))
    plt.close()
    print("  Saved ner_analysis.png")


def plot_vocabulary_size(vocab, segments):
    """Plot vocabulary size at different score thresholds."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 1. Vocabulary size vs threshold
    ax = axes[0]
    thresholds = np.linspace(0, vocab['mean_combined_score'].quantile(0.95), 50)
    sizes = [len(vocab[vocab['mean_combined_score'] >= t]) for t in thresholds]
    ax.plot(thresholds, sizes, color='steelblue', linewidth=2)
    ax.set_xlabel('Minimum Combined Score Threshold')
    ax.set_ylabel('Vocabulary Size (unique items)')
    ax.set_title('Implicit Vocabulary Size vs Score Threshold')
    ax.grid(True, alpha=0.3)

    # 2. Span length distribution
    ax = axes[1]
    len_counts = segments['n_tokens'].value_counts().sort_index()
    ax.bar(len_counts.index, len_counts.values, color='steelblue', edgecolor='black', linewidth=0.3)
    ax.set_xlabel('Span Length (tokens)')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Span Lengths')

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, 'vocabulary_size.png'))
    plt.close()
    print("  Saved vocabulary_size.png")


def plot_compositionality_vs_score(classified):
    """Plot relationship between erasure-like score and compositionality."""
    clf = classified.dropna(subset=['category', 'compositionality']).copy()
    meaningful = clf[clf['category'] != 'fragment']

    fig, ax = plt.subplots(figsize=(10, 6))

    if len(meaningful) > 0:
        # Color by category
        categories = meaningful['category'].unique()
        colors = dict(zip(categories, sns.color_palette('Set2', len(categories))))

        for cat in categories:
            cat_data = meaningful[meaningful['category'] == cat]
            ax.scatter(cat_data['mean_combined_score'], cat_data['compositionality'],
                      alpha=0.6, s=30, c=[colors[cat]], label=f'{cat} (n={len(cat_data)})')

        ax.set_xlabel('Mean Combined Score (Erasure-like)')
        ax.set_ylabel('Compositionality (0=non-compositional, 1=compositional)')
        ax.set_title('Erasure-like Score vs Compositionality')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)

        # Add correlation
        r, p = stats.spearmanr(meaningful['mean_combined_score'], meaningful['compositionality'])
        ax.text(0.02, 0.98, f'Spearman ρ={r:.3f}, p={p:.3e}',
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, 'compositionality_vs_score.png'))
    plt.close()
    print("  Saved compositionality_vs_score.png")


def statistical_tests(classified, baseline):
    """Run statistical tests and return results."""
    results = {}

    clf = classified.dropna(subset=['category', 'compositionality']).copy()
    meaningful = clf[clf['category'] != 'fragment']

    # Test 1: Are high-score items less compositional than low-score items?
    if len(meaningful) > 0:
        median_score = meaningful['mean_combined_score'].median()
        high = meaningful[meaningful['mean_combined_score'] >= median_score]['compositionality']
        low = meaningful[meaningful['mean_combined_score'] < median_score]['compositionality']

        if len(high) > 5 and len(low) > 5:
            u_stat, u_p = stats.mannwhitneyu(high, low, alternative='two-sided')
            d = (high.mean() - low.mean()) / np.sqrt((high.std()**2 + low.std()**2) / 2)
            results['high_vs_low_score'] = {
                'test': 'Mann-Whitney U',
                'high_mean': float(high.mean()),
                'low_mean': float(low.mean()),
                'U_statistic': float(u_stat),
                'p_value': float(u_p),
                'cohens_d': float(d),
                'n_high': int(len(high)),
                'n_low': int(len(low)),
            }

    # Test 2: Correlation between erasure score and compositionality
    if len(meaningful) > 5:
        r, p = stats.spearmanr(meaningful['mean_combined_score'], meaningful['compositionality'])
        results['score_compositionality_correlation'] = {
            'test': 'Spearman rank correlation',
            'rho': float(r),
            'p_value': float(p),
            'n': int(len(meaningful)),
        }

    # Test 3: Category differences in compositionality
    if len(meaningful) > 0:
        non_comp_cats = ['named_entity', 'idiom', 'technical_term', 'compound_word']
        comp_cats = ['compositional_phrase']

        non_comp = meaningful[meaningful['category'].isin(non_comp_cats)]['compositionality']
        comp = meaningful[meaningful['category'].isin(comp_cats)]['compositionality']

        if len(non_comp) > 5 and len(comp) > 5:
            u_stat, u_p = stats.mannwhitneyu(non_comp, comp, alternative='less')
            d = (non_comp.mean() - comp.mean()) / np.sqrt((non_comp.std()**2 + comp.std()**2) / 2)
            results['noncomp_vs_comp_categories'] = {
                'test': 'Mann-Whitney U (one-sided)',
                'non_compositional_mean': float(non_comp.mean()),
                'compositional_mean': float(comp.mean()),
                'U_statistic': float(u_stat),
                'p_value': float(u_p),
                'cohens_d': float(d),
                'n_non_compositional': int(len(non_comp)),
                'n_compositional': int(len(comp)),
            }

    # Test 4: NER items vs non-NER compositionality
    ner_comp = meaningful[meaningful.get('is_ner', False) == True]['compositionality'] if 'is_ner' in meaningful.columns else pd.Series()
    non_ner_comp = meaningful[meaningful.get('is_ner', False) == False]['compositionality'] if 'is_ner' in meaningful.columns else pd.Series()

    # Test 5: Vocabulary vs baseline compositionality
    if len(baseline) > 0 and 'compositionality' in baseline.columns:
        vocab_comp = clf['compositionality'].dropna()
        base_comp = baseline['compositionality'].dropna()
        if len(vocab_comp) > 5 and len(base_comp) > 5:
            u_stat, u_p = stats.mannwhitneyu(vocab_comp, base_comp, alternative='two-sided')
            results['vocab_vs_baseline'] = {
                'test': 'Mann-Whitney U',
                'vocab_mean': float(vocab_comp.mean()),
                'baseline_mean': float(base_comp.mean()),
                'U_statistic': float(u_stat),
                'p_value': float(u_p),
                'n_vocab': int(len(vocab_comp)),
                'n_baseline': int(len(base_comp)),
            }

    # Test 6: Category distribution (chi-squared)
    if len(meaningful) > 0:
        cat_counts = meaningful['category'].value_counts()
        # Test against uniform distribution
        expected = np.ones(len(cat_counts)) * len(meaningful) / len(cat_counts)
        chi2, chi_p = stats.chisquare(cat_counts.values, f_exp=expected)
        results['category_distribution'] = {
            'test': 'Chi-squared goodness of fit (vs uniform)',
            'chi2': float(chi2),
            'p_value': float(chi_p),
            'df': int(len(cat_counts) - 1),
            'categories': dict(cat_counts),
        }

    return results


def generate_example_table(classified):
    """Generate a table of representative examples for each category."""
    clf = classified.dropna(subset=['category']).copy()

    examples = []
    for cat in clf['category'].unique():
        cat_data = clf[clf['category'] == cat].sort_values('mean_combined_score', ascending=False)
        for _, row in cat_data.head(5).iterrows():
            examples.append({
                'category': cat,
                'text': row['text'],
                'erasure_score': round(row['mean_combined_score'], 3),
                'compositionality': round(row.get('compositionality', 0.5), 2),
            })

    return pd.DataFrame(examples)


def main():
    print("="*60)
    print("Analysis and Visualization Pipeline")
    print("="*60)

    vocab, segments, classified, ner, doc_stats, vocab_ner, baseline, config = load_data()

    print(f"\nData summary:")
    print(f"  Unique vocabulary items: {len(vocab)}")
    print(f"  Segment instances: {len(segments)}")
    print(f"  Classified items: {len(classified)}")
    print(f"  NER entities: {len(ner)}")
    print(f"  Documents processed: {len(doc_stats)}")

    # Generate all plots
    print("\nGenerating visualizations...")
    plot_score_distributions(segments, classified)
    plot_category_analysis(classified, baseline)
    plot_ner_analysis(vocab_ner, segments)
    plot_vocabulary_size(vocab, segments)
    plot_compositionality_vs_score(classified)

    # Run statistical tests
    print("\nRunning statistical tests...")
    test_results = statistical_tests(classified, baseline)

    for name, result in test_results.items():
        print(f"\n  {name}:")
        for k, v in result.items():
            if isinstance(v, float):
                print(f"    {k}: {v:.4f}")
            else:
                print(f"    {k}: {v}")

    # Save statistical results
    # Convert numpy types for JSON
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    def deep_convert(d):
        if isinstance(d, dict):
            return {k: deep_convert(v) for k, v in d.items()}
        if isinstance(d, list):
            return [deep_convert(v) for v in d]
        return convert(d)

    with open(os.path.join(RESULTS_DIR, 'statistical_tests.json'), 'w') as f:
        json.dump(deep_convert(test_results), f, indent=2)

    # Generate example table
    examples = generate_example_table(classified)
    examples.to_csv(os.path.join(RESULTS_DIR, 'example_items.csv'), index=False)
    print(f"\nSaved {len(examples)} example items")

    # Summary statistics
    clf = classified.dropna(subset=['category']).copy()
    meaningful = clf[clf['category'] != 'fragment']

    summary = {
        'total_unique_items': len(vocab),
        'total_instances': len(segments),
        'meaningful_classified': len(meaningful),
        'fragment_rate': float(len(clf[clf['category'] == 'fragment']) / len(clf)) if len(clf) > 0 else 0,
        'non_compositional_rate': float(
            len(meaningful[meaningful['compositionality'] < 0.5]) / len(meaningful)
        ) if len(meaningful) > 0 else 0,
        'mean_compositionality_meaningful': float(meaningful['compositionality'].mean()) if len(meaningful) > 0 else 0,
        'ner_overlap_rate': float(vocab_ner['is_ner'].mean()) if 'is_ner' in vocab_ner.columns else 0,
        'category_counts': dict(clf['category'].value_counts()) if len(clf) > 0 else {},
    }

    with open(os.path.join(RESULTS_DIR, 'summary_statistics.json'), 'w') as f:
        json.dump(deep_convert(summary), f, indent=2)

    print(f"\n{'='*60}")
    print("KEY FINDINGS")
    print(f"{'='*60}")
    print(f"1. Total implicit vocabulary size: {len(vocab)} unique multi-token items")
    print(f"2. Fragment rate: {summary['fragment_rate']*100:.1f}% (noise in segmentation)")
    print(f"3. Among meaningful items:")
    print(f"   - Non-compositional (score < 0.5): {summary['non_compositional_rate']*100:.1f}%")
    print(f"   - Mean compositionality: {summary['mean_compositionality_meaningful']:.3f}")
    print(f"4. NER overlap: {summary['ner_overlap_rate']*100:.1f}% of items match NER entities")
    if 'noncomp_vs_comp_categories' in test_results:
        t = test_results['noncomp_vs_comp_categories']
        print(f"5. Non-compositional vs compositional categories: p={t['p_value']:.3e}, d={t['cohens_d']:.2f}")

    print(f"\nAll results saved to {RESULTS_DIR}/")
    print(f"Plots saved to {PLOT_DIR}/")


if __name__ == '__main__':
    main()
