"""
BERT Classification Analysis
Analyzes BERT classification results from bert_test_predictions.xlsx

Input:
  <PROJECT_ROOT>/classifyOutcome/BERT_output/BERT/bert_test_predictions.xlsx

Output:
  <PROJECT_ROOT>/analyzeOutcome/BERT/
    - BERT_Analysis_Results.xlsx
    - BERT_Confusion_Matrix.png
    - BERT_Performance_Summary.png
    - BERT_Calibration_Curve.png
    - BERT_Confidence_Distribution.png
    - BERT_Class_Metrics_Heatmap.png
    - BERT_Report.txt
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (confusion_matrix, accuracy_score,
                             precision_score, recall_score, f1_score)
from scipy.stats import pearsonr
import warnings
import os
from datetime import datetime

warnings.filterwarnings('ignore')

# ============================================================================
# 全局样式
# ============================================================================
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 600
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['legend.fontsize'] = 10

VIRIDIS_PALETTE = [
    '#440154', '#482878', '#3e4989', '#31688e',
    '#26828e', '#1f9e89', '#35b779', '#6ece58',
    '#b5de2b', '#fde725'
]

# ============================================================================
# 常量
# ============================================================================
CATEGORY_MAP = {
    'composite': 'Composite',
    'conditional': 'Conditional',
    'interactive': 'Interactive',
    'sequential': 'Sequential',
    'structural': 'Structural'
}
CATEGORY_ORDER = ['composite', 'conditional', 'interactive', 'sequential', 'structural']

PROB_COLUMNS = [
    'prob_composite', 'prob_conditional', 'prob_interactive',
    'prob_sequential', 'prob_structural'
]


# ============================================================================
# 数据加载
# ============================================================================
def load_bert_data(file_path):
    """Load BERT predictions from XLSX."""
    print(f"\n[LOAD] {file_path}")
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return None

    df = pd.read_excel(file_path)
    df.columns = [str(c).strip().replace('\ufeff', '') for c in df.columns]
    print(f"   shape={df.shape}")
    print(f"   columns={list(df.columns)}")
    return df


def adapt_columns(df):
    """列名映射到脚本内部标准列名。"""
    df = df.copy()
    rename_map = {
        'true_label':      'label',
        'bert_pred':       'predicted_label',
        'bert_confidence': 'prediction_confidence',
        'bert_correct':    'is_correct',
    }
    for old, new in rename_map.items():
        if old in df.columns and new not in df.columns:
            df[new] = df[old]

    if 'label' in df.columns:
        df['label'] = df['label'].astype(str).str.strip().str.lower()
    if 'predicted_label' in df.columns:
        df['predicted_label'] = df['predicted_label'].astype(str).str.strip().str.lower()

    if 'prediction_confidence' in df.columns:
        df['prediction_confidence'] = pd.to_numeric(
            df['prediction_confidence'], errors='coerce'
        ).fillna(0.0)
        print(f"   Confidence range: [{df['prediction_confidence'].min():.4f}, "
              f"{df['prediction_confidence'].max():.4f}]")

    if 'is_correct' in df.columns:
        def _to_int(x):
            if pd.isna(x): return 0
            if isinstance(x, (bool, np.bool_)): return int(x)
            if isinstance(x, (int, np.integer, float, np.floating)): return int(x != 0)
            s = str(x).strip().lower()
            if s in ('1', 'true', 'yes', 'y', 't', 'correct'): return 1
            if s in ('0', 'false', 'no', 'n', 'f', 'wrong', 'incorrect', ''): return 0
            try: return int(float(s) != 0)
            except Exception: return 0
        df['is_correct'] = df['is_correct'].apply(_to_int)
        print(f"   is_correct distribution: {df['is_correct'].value_counts().to_dict()}")

    if 'is_correct' in df.columns and 'Is_Correct' not in df.columns:
        df['Is_Correct'] = df['is_correct']

    if 'is_high_confidence' not in df.columns:
        df['is_high_confidence'] = (df['prediction_confidence'] >= 0.5).astype(int)
        print(f"   is_high_confidence distribution: "
              f"{df['is_high_confidence'].value_counts().to_dict()}")

    return df


def validate_data(df):
    required = ['label', 'predicted_label', 'prediction_confidence',
                'is_high_confidence', 'is_correct']
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"⚠️ Missing columns: {missing}")
        return False

    print(f"\n📊 Data Quality Check:")
    print(f"   Total samples: {len(df)}")
    print(f"   Unique true labels: {sorted(df['label'].unique())}")
    print(f"   Unique predicted labels: {sorted(df['predicted_label'].unique())}")

    n_correct = int(df['is_correct'].sum())
    n_total = len(df)
    print(f"   Correct predictions: {n_correct} ({n_correct/n_total*100:.2f}%)")
    print(f"   Incorrect predictions: {n_total-n_correct} ({(n_total-n_correct)/n_total*100:.2f}%)")

    mismatch = 0
    for _, row in df.iterrows():
        if int(row['label'] == row['predicted_label']) != int(row['is_correct']):
            mismatch += 1
    if mismatch == 0:
        print(f"   ✅ is_correct column is consistent with labels")
    else:
        print(f"   ⚠️ {mismatch} inconsistencies in is_correct")

    print(f"\n   True Label Distribution:")
    for label, count in df['label'].value_counts().items():
        disp = CATEGORY_MAP.get(label, label)
        print(f"      {disp} ({label}): {count} ({count/n_total*100:.1f}%)")
    return True


# ============================================================================
# 指标计算
# ============================================================================
def calculate_ece(confidences, correctness, n_bins=10):
    if len(confidences) == 0:
        return 0, []
    conf = np.array(confidences)
    corr = np.array(correctness)
    bins = np.linspace(0, 1, n_bins + 1)
    bin_idx = np.digitize(conf, bins, right=False)
    stats = []
    total = len(conf)
    for i in range(1, n_bins + 1):
        mask = bin_idx == i
        cnt = int(np.sum(mask))
        if cnt == 0:
            continue
        stats.append({
            'bin_center': (bins[i-1] + bins[i]) / 2,
            'accuracy':   float(np.mean(corr[mask])),
            'confidence': float(np.mean(conf[mask])),
            'count':      cnt,
            'weight':     cnt / total,
        })
    ece = sum(s['weight'] * abs(s['accuracy'] - s['confidence']) for s in stats)
    return ece, stats


def compute_metrics(df):
    y_true = df['label'].tolist()
    y_pred = df['predicted_label'].tolist()
    y_conf = df['prediction_confidence'].tolist()
    y_correct = df['is_correct'].tolist()
    y_high = df['is_high_confidence'].tolist()

    n_total = len(df)
    n_correct = int(sum(y_correct))
    n_incorrect = n_total - n_correct
    n_high = int(sum(y_high))
    n_low = n_total - n_high

    acc = (n_correct / n_total) * 100
    acc_computed = accuracy_score(y_true, y_pred) * 100
    print(f"\n   📊 Accuracy:")
    print(f"   from is_correct: {acc:.2f}%")
    print(f"   computed:        {acc_computed:.2f}%")
    if abs(acc - acc_computed) > 0.01:
        print(f"   ⚠️ Mismatch, using computed.")
        acc = acc_computed
        n_correct = int(round(acc / 100 * n_total))
        n_incorrect = n_total - n_correct

    precision = precision_score(y_true, y_pred, average='macro', zero_division=0) * 100
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0) * 100
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0) * 100

    per_class = {}
    for label in CATEGORY_ORDER:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)
        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0
        per_class[label] = {
            'display_name': CATEGORY_MAP.get(label, label),
            'precision': p * 100, 'recall': r * 100, 'f1': f * 100,
            'support': sum(1 for t in y_true if t == label),
            'correct': tp, 'tp': tp, 'fp': fp, 'fn': fn,
        }

    avg_conf = float(np.mean(y_conf)) * 100 if y_conf else 0
    std_conf = float(np.std(y_conf)) * 100 if y_conf else 0
    min_conf = float(np.min(y_conf)) * 100 if y_conf else 0
    max_conf = float(np.max(y_conf)) * 100 if y_conf else 0

    correct_conf = [c for c, ok in zip(y_conf, y_correct) if ok]
    incorrect_conf = [c for c, ok in zip(y_conf, y_correct) if not ok]
    avg_correct_conf = float(np.mean(correct_conf)) * 100 if correct_conf else 0
    avg_incorrect_conf = float(np.mean(incorrect_conf)) * 100 if incorrect_conf else 0

    correctness = [1 if c else 0 for c in y_correct]
    ece, bin_stats = calculate_ece(y_conf, correctness)
    ece_percent = ece * 100

    if len(y_conf) > 1 and len(set(y_conf)) > 1:
        pearson_corr, _ = pearsonr(y_conf, correctness)
        if np.isnan(pearson_corr):
            pearson_corr = 0
    else:
        pearson_corr = 0

    all_labels = sorted(set(y_true + y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=all_labels)

    high_correct = sum(1 for c, h in zip(y_correct, y_high) if c and h)
    high_acc = (high_correct / n_high) * 100 if n_high > 0 else 0
    low_correct = sum(1 for c, h in zip(y_correct, y_high) if c and not h)
    low_acc = (low_correct / n_low) * 100 if n_low > 0 else 0

    return {
        'n_total': n_total, 'n_correct': n_correct, 'n_incorrect': n_incorrect,
        'n_high_conf': n_high, 'n_low_conf': n_low,
        'accuracy': acc, 'accuracy_computed': acc_computed,
        'precision': precision, 'recall': recall, 'f1': f1,
        'avg_confidence': avg_conf, 'std_confidence': std_conf,
        'min_confidence': min_conf, 'max_confidence': max_conf,
        'avg_correct_conf': avg_correct_conf, 'avg_incorrect_conf': avg_incorrect_conf,
        'ece': ece_percent, 'pearson_corr': pearson_corr * 100,
        'high_conf_accuracy': high_acc, 'low_conf_accuracy': low_acc,
        'confusion_matrix': cm, 'confusion_labels': all_labels,
        'true_labels': y_true, 'predictions': y_pred,
        'confidences': y_conf, 'is_correct': y_correct,
        'is_high_conf': y_high, 'bin_stats': bin_stats,
        'per_class_metrics': per_class,
        'df': df,
    }


# ============================================================================
# Excel 导出
# ============================================================================
def export_excel(metrics, output_dir):
    output_file = os.path.join(output_dir, 'BERT_Analysis_Results.xlsx')
    print(f"\n📊 Exporting results to {output_file}...")

    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        # Sheet 1: Performance Summary
        summary_data = [
            {'Metric': 'Total Samples', 'Value': metrics['n_total']},
            {'Metric': 'Correct Predictions', 'Value': metrics['n_correct']},
            {'Metric': 'Incorrect Predictions', 'Value': metrics['n_incorrect']},
            {'Metric': 'High Confidence Samples', 'Value': metrics['n_high_conf']},
            {'Metric': 'Low Confidence Samples', 'Value': metrics['n_low_conf']},
            {'Metric': '', 'Value': ''},
            {'Metric': 'Accuracy (%)', 'Value': round(metrics['accuracy'], 2)},
            {'Metric': 'Precision (macro) (%)', 'Value': round(metrics['precision'], 2)},
            {'Metric': 'Recall (macro) (%)', 'Value': round(metrics['recall'], 2)},
            {'Metric': 'F1 Score (macro) (%)', 'Value': round(metrics['f1'], 2)},
            {'Metric': '', 'Value': ''},
            {'Metric': 'High Confidence Accuracy (%)', 'Value': round(metrics['high_conf_accuracy'], 2)},
            {'Metric': 'Low Confidence Accuracy (%)', 'Value': round(metrics['low_conf_accuracy'], 2)},
            {'Metric': '', 'Value': ''},
            {'Metric': 'Average Confidence (%)', 'Value': round(metrics['avg_confidence'], 2)},
            {'Metric': 'Std Confidence (%)', 'Value': round(metrics['std_confidence'], 2)},
            {'Metric': 'Min Confidence (%)', 'Value': round(metrics['min_confidence'], 2)},
            {'Metric': 'Max Confidence (%)', 'Value': round(metrics['max_confidence'], 2)},
            {'Metric': '', 'Value': ''},
            {'Metric': 'Avg Confidence (Correct) (%)', 'Value': round(metrics['avg_correct_conf'], 2)},
            {'Metric': 'Avg Confidence (Incorrect) (%)', 'Value': round(metrics['avg_incorrect_conf'], 2)},
            {'Metric': '', 'Value': ''},
            {'Metric': 'ECE (%)', 'Value': round(metrics['ece'], 2)},
            {'Metric': 'Pearson Correlation (%)', 'Value': round(metrics['pearson_corr'], 2)},
        ]
        pd.DataFrame(summary_data).to_excel(writer, sheet_name='Performance Summary', index=False)

        # Sheet 2: Per-Class Metrics
        class_data = []
        for label, m in metrics['per_class_metrics'].items():
            class_data.append({
                'Class': m['display_name'], 'Class Code': label,
                'Precision (%)': round(m['precision'], 2),
                'Recall (%)':    round(m['recall'], 2),
                'F1 Score (%)':  round(m['f1'], 2),
                'Support': m['support'], 'Correct': m['correct'],
                'TP': m['tp'], 'FP': m['fp'], 'FN': m['fn'],
            })
        pd.DataFrame(class_data).to_excel(writer, sheet_name='Per-Class Metrics', index=False)

        # Sheet 3: Confusion Matrix
        cm = metrics['confusion_matrix']
        labels = metrics['confusion_labels']
        disp = [CATEGORY_MAP.get(l, l) for l in labels]
        pd.DataFrame(cm, index=disp, columns=disp).to_excel(
            writer, sheet_name='Confusion Matrix', index=True)

        # Sheet 4: Calibration Statistics
        calib = [{
            'Bin Center':     round(s['bin_center'], 3),
            'Accuracy (%)':   round(s['accuracy'] * 100, 2),
            'Confidence (%)': round(s['confidence'] * 100, 2),
            'Count':          s['count'],
            'Weight':         round(s['weight'], 3),
        } for s in metrics['bin_stats']]
        if calib:
            pd.DataFrame(calib).to_excel(writer, sheet_name='Calibration Statistics', index=False)

        # Sheet 5: Raw Data
        metrics['df'].to_excel(writer, sheet_name='Raw Data', index=False)

    print(f"✅ Exported to {output_file}")
    return output_file


# ============================================================================
# 图表
# ============================================================================
def plot_confusion_matrix(metrics, output_dir):
    output_file = os.path.join(output_dir, 'BERT_Confusion_Matrix.png')
    print(f"\n📊 Generating confusion matrix...")
    cm = metrics['confusion_matrix']
    labels = metrics['confusion_labels']
    disp = [CATEGORY_MAP.get(l, l) for l in labels]

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='viridis',
                xticklabels=disp, yticklabels=disp,
                ax=ax, cbar=True,
                cbar_kws={'label': f'Count (N={metrics["n_total"]:,})'})
    ax.set_xlabel('Predicted Label', fontsize=13)
    ax.set_ylabel('True Label', fontsize=13)
    ax.set_title(f'BERT - Confusion Matrix\n'
                 f'Accuracy: {metrics["accuracy"]:.1f}%, N={metrics["n_total"]:,}',
                 fontsize=14, fontweight='bold')
    ax.set_xticklabels(disp, rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✅ Saved: {output_file}")
    return output_file


def plot_performance_summary(metrics, output_dir):
    output_file = os.path.join(output_dir, 'BERT_Performance_Summary.png')
    print(f"\n📊 Generating performance summary...")
    names = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'Avg Confidence']
    values = [metrics['accuracy'], metrics['precision'], metrics['recall'],
              metrics['f1'], metrics['avg_confidence']]
    colors = [VIRIDIS_PALETTE[i] for i in [3, 4, 5, 6, 8]]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(names))
    bars = ax.bar(x, values, color=colors, edgecolor='black', linewidth=0.8)
    ax.set_xlabel('Metric', fontsize=13)
    ax.set_ylabel('Percentage (%)', fontsize=13)
    ax.set_title('BERT - Performance Metrics', fontsize=15, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=11)
    ax.set_ylim(0, 105)
    ax.grid(axis='y', alpha=0.3)
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 1,
                f'{h:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_file, dpi=600, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✅ Saved: {output_file}")
    return output_file


def plot_calibration_curve(metrics, output_dir):
    output_file = os.path.join(output_dir, 'BERT_Calibration_Curve.png')
    print(f"\n📊 Generating calibration curve...")
    bins = metrics['bin_stats']
    if not bins:
        print("   ⚠️ No calibration data")
        return None

    fig, ax = plt.subplots(figsize=(8, 8))
    confs = [s['confidence'] for s in bins]
    accs = [s['accuracy'] for s in bins]
    cnts = [s['count'] for s in bins]
    total = sum(cnts)
    weights = [c / total for c in cnts]

    ax.scatter(confs, accs, s=[w * 2000 + 50 for w in weights],
               c=VIRIDIS_PALETTE[5], alpha=0.6, edgecolors='black', linewidth=0.5)
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration', linewidth=2)
    for c, a, n in zip(confs, accs, cnts):
        ax.annotate(f'n={n}', (c, a), textcoords="offset points",
                    xytext=(0, 10), ha='center', fontsize=8)
    ax.set_xlabel('Confidence', fontsize=13)
    ax.set_ylabel('Accuracy', fontsize=13)
    ax.set_title(f'BERT - Calibration Curve\n'
                 f'ECE: {metrics["ece"]:.2f}%, N={metrics["n_total"]:,}',
                 fontsize=14, fontweight='bold')
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✅ Saved: {output_file}")
    return output_file


def plot_confidence_distribution(metrics, output_dir):
    output_file = os.path.join(output_dir, 'BERT_Confidence_Distribution.png')
    print(f"\n📊 Generating confidence distribution...")
    confs = metrics['confidences']
    corr = metrics['is_correct']
    correct_conf = [c for c, x in zip(confs, corr) if x]
    incorrect_conf = [c for c, x in zip(confs, corr) if not x]

    fig, ax = plt.subplots(figsize=(10, 6))
    bins = np.linspace(0, 1, 21)
    ax.hist(correct_conf, bins=bins, alpha=0.7,
            label=f'Correct (n={len(correct_conf)})',
            color=VIRIDIS_PALETTE[5], edgecolor='black', linewidth=0.5)
    ax.hist(incorrect_conf, bins=bins, alpha=0.7,
            label=f'Incorrect (n={len(incorrect_conf)})',
            color=VIRIDIS_PALETTE[1], edgecolor='black', linewidth=0.5)
    ax.set_xlabel('Confidence', fontsize=13)
    ax.set_ylabel('Count', fontsize=13)
    ax.set_title('BERT - Confidence Distribution by Correctness',
                 fontsize=14, fontweight='bold')
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✅ Saved: {output_file}")
    return output_file


def plot_class_metrics_heatmap(metrics, output_dir):
    output_file = os.path.join(output_dir, 'BERT_Class_Metrics_Heatmap.png')
    print(f"\n📊 Generating class metrics heatmap...")
    data = []
    for label, m in metrics['per_class_metrics'].items():
        data.append({'Class': m['display_name'],
                     'Precision': m['precision'],
                     'Recall': m['recall'],
                     'F1 Score': m['f1']})
    df_class = pd.DataFrame(data).set_index('Class')

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df_class, annot=True, fmt='.1f', cmap='viridis',
                ax=ax, cbar=True, cbar_kws={'label': 'Percentage (%)'})
    ax.set_title('BERT - Per-Class Performance Metrics',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✅ Saved: {output_file}")
    return output_file


# ============================================================================
# 文本报告
# ============================================================================
def write_report(metrics, output_dir):
    output_file = os.path.join(output_dir, 'BERT_Report.txt')
    print(f"\n📄 Generating text report...")

    lines = []
    lines.append("=" * 70)
    lines.append("BERT - CLASSIFICATION PERFORMANCE REPORT")
    lines.append("=" * 70)
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Total Samples: {metrics['n_total']:,}")
    lines.append(f"Correct: {metrics['n_correct']:,} "
                 f"({metrics['n_correct']/metrics['n_total']*100:.1f}%)")
    lines.append(f"Incorrect: {metrics['n_incorrect']:,} "
                 f"({metrics['n_incorrect']/metrics['n_total']*100:.1f}%)")
    lines.append(f"High Confidence: {metrics['n_high_conf']:,} "
                 f"({metrics['n_high_conf']/metrics['n_total']*100:.1f}%)")
    lines.append(f"Low Confidence: {metrics['n_low_conf']:,} "
                 f"({metrics['n_low_conf']/metrics['n_total']*100:.1f}%)")
    lines.append("")
    lines.append("-" * 50)
    lines.append("PERFORMANCE METRICS")
    lines.append("-" * 50)
    lines.append(f"Accuracy:              {metrics['accuracy']:.2f}%")
    lines.append(f"Precision (macro):     {metrics['precision']:.2f}%")
    lines.append(f"Recall (macro):        {metrics['recall']:.2f}%")
    lines.append(f"F1 Score (macro):      {metrics['f1']:.2f}%")
    lines.append("")
    lines.append(f"High Confidence Accuracy: {metrics['high_conf_accuracy']:.2f}%")
    lines.append(f"Low Confidence Accuracy:  {metrics['low_conf_accuracy']:.2f}%")
    lines.append("")
    lines.append("-" * 50)
    lines.append("CONFIDENCE STATISTICS")
    lines.append("-" * 50)
    lines.append(f"Average Confidence:    {metrics['avg_confidence']:.2f}%")
    lines.append(f"Std Confidence:        {metrics['std_confidence']:.2f}%")
    lines.append(f"Min Confidence:        {metrics['min_confidence']:.2f}%")
    lines.append(f"Max Confidence:        {metrics['max_confidence']:.2f}%")
    lines.append("")
    lines.append(f"Avg Confidence (Correct):   {metrics['avg_correct_conf']:.2f}%")
    lines.append(f"Avg Confidence (Incorrect): {metrics['avg_incorrect_conf']:.2f}%")
    lines.append("")
    lines.append("-" * 50)
    lines.append("CALIBRATION METRICS")
    lines.append("-" * 50)
    lines.append(f"ECE:                   {metrics['ece']:.2f}%")
    lines.append(f"Pearson Correlation:   {metrics['pearson_corr']:.2f}%")
    lines.append("")
    lines.append("-" * 50)
    lines.append("PER-CLASS PERFORMANCE")
    lines.append("-" * 50)
    lines.append(f"{'Class':<20} {'Precision':>12} {'Recall':>12} {'F1 Score':>12} {'Support':>10}")
    lines.append("-" * 69)
    for label, m in metrics['per_class_metrics'].items():
        lines.append(f"{m['display_name']:<20} {m['precision']:>11.2f}% "
                     f"{m['recall']:>11.2f}% {m['f1']:>11.2f}% {m['support']:>10}")
    lines.append("")
    lines.append("-" * 50)
    lines.append("INTERPRETATION")
    lines.append("-" * 50)
    if metrics['accuracy'] >= 90:
        lines.append("• Accuracy: Excellent")
    elif metrics['accuracy'] >= 80:
        lines.append("• Accuracy: Good")
    elif metrics['accuracy'] >= 70:
        lines.append("• Accuracy: Moderate")
    else:
        lines.append("• Accuracy: Needs improvement")

    if metrics['ece'] < 5:
        lines.append("• Calibration (ECE): Excellent")
    elif metrics['ece'] < 10:
        lines.append("• Calibration (ECE): Good")
    elif metrics['ece'] < 20:
        lines.append("• Calibration (ECE): Moderate")
    else:
        lines.append("• Calibration (ECE): Poor (overconfident)")

    if metrics['pearson_corr'] > 30:
        lines.append("• Confidence-Accuracy Correlation: Strong positive")
    elif metrics['pearson_corr'] > 10:
        lines.append("• Confidence-Accuracy Correlation: Moderate positive")
    elif metrics['pearson_corr'] > 0:
        lines.append("• Confidence-Accuracy Correlation: Weak positive")
    else:
        lines.append("• Confidence-Accuracy Correlation: Weak / negative")
    lines.append("")
    lines.append("=" * 70)

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"✅ Saved: {output_file}")
    return output_file


# ============================================================================
# 主函数
# ============================================================================
def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)

    input_file = os.path.join(
        project_dir, 'classifyOutcome', 'BERT_output', 'BERT',
        'bert_test_predictions.xlsx'
    )
    output_dir = os.path.join(project_dir, 'analyzeOutcome', 'BERT')
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("BERT - PERFORMANCE ANALYSIS")
    print("=" * 70)
    print(f"Script dir: {script_dir}")
    print(f"Project dir: {project_dir}")
    print(f"Input file:  {input_file}")
    print(f"Output dir:  {output_dir}")

    if not os.path.exists(input_file):
        print(f"❌ Input file not found: {input_file}")
        return

    df = load_bert_data(input_file)
    if df is None:
        return
    df = adapt_columns(df)

    if not validate_data(df):
        print("❌ Data validation failed.")
        return

    metrics = compute_metrics(df)

    print(f"\n   ✅ Summary:")
    print(f"   N:               {metrics['n_total']:,}")
    print(f"   Accuracy:        {metrics['accuracy']:.2f}%")
    print(f"   ECE:             {metrics['ece']:.2f}%")
    print(f"   Pearson:         {metrics['pearson_corr']:.2f}%")
    print(f"   High-Conf Acc:   {metrics['high_conf_accuracy']:.2f}%")
    print(f"   Low-Conf Acc:    {metrics['low_conf_accuracy']:.2f}%")

    excel_file = export_excel(metrics, output_dir)
    cm_file = plot_confusion_matrix(metrics, output_dir)
    perf_file = plot_performance_summary(metrics, output_dir)
    calib_file = plot_calibration_curve(metrics, output_dir)
    conf_file = plot_confidence_distribution(metrics, output_dir)
    class_file = plot_class_metrics_heatmap(metrics, output_dir)
    report_file = write_report(metrics, output_dir)

    print("\n" + "=" * 70)
    print("✅ ANALYSIS COMPLETE!")
    print("=" * 70)
    print(f"\n📂 Output directory: {output_dir}")
    for f in [excel_file, cm_file, perf_file, calib_file,
              conf_file, class_file, report_file]:
        if f:
            print(f"   {os.path.basename(f)}")
    print("=" * 70)


if __name__ == "__main__":
    main()