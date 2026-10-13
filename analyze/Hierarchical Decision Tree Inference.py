"""
Hierarchical Binary Decision Tree Analysis - Round-by-Round Binary Classification Performance
Analyzes binary classification performance for each decision round in the hierarchical process.
Each round is evaluated as a binary classifier (Pass/Reject) based on the decision tree logic.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
from scipy.stats import pearsonr
import warnings
import os
from pathlib import Path

warnings.filterwarnings('ignore')

# Set font and style for publication
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 600
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['legend.fontsize'] = 10

# Viridis color palette
VIRIDIS_PALETTE = [
    '#440154', '#482878', '#3e4989', '#31688e',
    '#26828e', '#1f9e89', '#35b779', '#6ece58',
    '#b5de2b', '#fde725'
]

# Step names mapping
STEP_NAMES = {
    1: 'Atomicity Check',
    2: 'User Action Check',
    3: 'System Condition Check',
    4: 'Conditional Check'
}

# Decision tree rules: for each round, which final labels should pass this check
DECISION_TREE_RULES = {
    1: {  # Atomicity Check
        'composite': False,  # Composite should REJECT (not atomic)
        'interactive': True,
        'sequential': True,
        'conditional': True,
        'structural': True
    },
    2: {  # User Action Check
        'interactive': True,  # Interactive should PASS
        'sequential': False,  # Others should REJECT
        'conditional': False,
        'structural': False
    },
    3: {  # System Condition Check
        'sequential': True,  # Sequential should PASS
        'conditional': False,  # Others should REJECT
        'structural': False
    },
    4: {  # Conditional Check
        'conditional': True,  # Conditional should PASS
        'structural': False  # Structural should REJECT
    }
}

# NOTE: TARGET_SAMPLES is no longer hardcoded.
# The analysis now uses the actual dataset size (len(df)) without any resampling.

# ============================================================================
# Path Configuration (Relative Paths)
# ============================================================================

# Get current script directory
try:
    SCRIPT_DIR = Path(__file__).parent.absolute()
except NameError:
    SCRIPT_DIR = Path(os.getcwd()).absolute()

# Base directory (parent of analyze)
BASE_DIR = SCRIPT_DIR.parent

# Input file path (relative to base directory)
INPUT_FILE = BASE_DIR / "classifyOutcome" / "Hierarchical Binary Decision Routing.xlsx"

# Output directory (relative to base directory)
OUTPUT_DIR = BASE_DIR / "analyzeOutcome" / "Hierarchical Decision Tree Inference"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Output file paths
OUTPUT_EXCEL = OUTPUT_DIR / "Hierarchical_Binary_Decision_Tree_Analysis.xlsx"
OUTPUT_CHART = OUTPUT_DIR / "Hierarchical_Binary_Decision_Tree_Performance_Bar_Charts.png"

# ============================================================================
# Column Name Mapping
# ============================================================================

DECISION_COL_TEMPLATE = "step{}_decision"
RESULT_COL_TEMPLATE = "step{}_result"
CONFIDENCE_COL_TEMPLATE = "step{}_confidence"
EVIDENCE_COL_TEMPLATE = "step{}_evidence"

# True label column
TRUE_LABEL_COL = "true_label"


# ============================================================================
# Data Loading Functions
# ============================================================================

def load_data(file_path):
    """Load the sequential classification results"""
    if not file_path.exists():
        print(f"❌ File not found: {file_path}")
        return None

    df = pd.read_excel(file_path)
    print(f"✅ Loaded {len(df)} samples from {file_path}")
    print(f"   Columns: {df.columns.tolist()}")
    return df


# ============================================================================
# Ground Truth & Binary Prediction Functions
# ============================================================================

def get_ground_truth_binary_label(final_label, round_num):
    """
    Determine the ground truth binary label for a sample at a given round
    Returns: True (should pass), False (should reject), or None (not applicable)
    """
    if round_num not in DECISION_TREE_RULES:
        return None

    if final_label not in DECISION_TREE_RULES[round_num]:
        return None

    return DECISION_TREE_RULES[round_num][final_label]


def get_round_binary_predictions(df, round_num):
    """
    Get the binary predictions (Pass/Reject) for each sample at a given round
    Only includes samples that have data in the decision column
    """
    decision_col = DECISION_COL_TEMPLATE.format(round_num)
    result_col = RESULT_COL_TEMPLATE.format(round_num)
    conf_col = CONFIDENCE_COL_TEMPLATE.format(round_num)

    # Get rows that have data for this round (non-null in decision column)
    mask = df[decision_col].notna()

    predictions = []  # True = Pass, False = Reject
    confidences = []
    sample_indices = []

    for idx in df[mask].index:
        result = df.loc[idx, result_col] if result_col in df.columns else None
        decision = df.loc[idx, decision_col]

        # Determine binary prediction (Pass = True, Reject = False)
        if pd.notna(result):
            result_str = str(result).strip().lower()
            if result_str in ['通过', 'pass', 'true', 'yes', '1', 'pass']:
                pred = True
            elif result_str in ['不通过', 'reject', 'false', 'no', '0', 'rejection']:
                pred = False
            else:
                # Fallback to decision column
                pred = str(decision).strip().lower() in ['通过', 'pass', 'true', 'yes', '1', 'pass']
        else:
            pred = str(decision).strip().lower() in ['通过', 'pass', 'true', 'yes', '1', 'pass']

        predictions.append(pred)
        sample_indices.append(idx)

        # Get confidence
        if pd.notna(df.loc[idx, conf_col]):
            try:
                confidences.append(float(df.loc[idx, conf_col]))
            except:
                confidences.append(0.5)
        else:
            confidences.append(0.5)

    return sample_indices, predictions, confidences


# ============================================================================
# Metric Calculation Functions
# ============================================================================

def calculate_ece(confidences, correctness, n_bins=10):
    """
    Calculate Expected Calibration Error (ECE)
    """
    if len(confidences) == 0:
        return 0, []

    confidences = np.array(confidences)
    correctness = np.array(correctness)

    bins = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(confidences, bins, right=False)

    bin_stats = []
    total_samples = len(confidences)

    for i in range(1, n_bins + 1):
        mask = (bin_indices == i)
        bin_count = np.sum(mask)

        if bin_count == 0:
            continue

        bin_conf = np.mean(confidences[mask])
        bin_acc = np.mean(correctness[mask])
        bin_center = (bins[i - 1] + bins[i]) / 2

        bin_stats.append({
            'bin_center': bin_center,
            'accuracy': bin_acc,
            'confidence': bin_conf,
            'count': bin_count,
            'weight': bin_count / total_samples
        })

    ece = 0
    for stat in bin_stats:
        ece += stat['weight'] * abs(stat['accuracy'] - stat['confidence'])

    return ece, bin_stats


def calculate_round_binary_metrics(df, round_num, true_label_col=TRUE_LABEL_COL):
    """
    Calculate binary classification metrics for a specific round
    Each round is evaluated as a binary classifier (Pass vs Reject)
    Only includes samples that should reach this round based on decision tree
    """
    # Get samples with data for this round
    sample_indices, predictions, confidences = get_round_binary_predictions(df, round_num)

    if len(predictions) == 0:
        return None

    # Get ground truth binary labels based on final classification
    true_binary_labels = []
    valid_predictions = []
    valid_confidences = []
    valid_sample_indices = []

    for i, idx in enumerate(sample_indices):
        final_label = df.loc[idx, true_label_col]
        if pd.isna(final_label) or final_label == '':
            continue

        final_label_str = str(final_label).strip().lower()
        ground_truth = get_ground_truth_binary_label(final_label_str, round_num)

        if ground_truth is not None:  # Sample should reach this round
            true_binary_labels.append(ground_truth)
            valid_predictions.append(predictions[i])
            valid_confidences.append(confidences[i])
            valid_sample_indices.append(idx)

    if len(true_binary_labels) == 0:
        return None

    # Convert to numpy arrays
    true_binary_labels = np.array(true_binary_labels)
    valid_predictions = np.array(valid_predictions)
    valid_confidences = np.array(valid_confidences)

    # Calculate accuracy
    accuracy = accuracy_score(true_binary_labels, valid_predictions) * 100

    # Average confidence
    avg_confidence = np.mean(valid_confidences) * 100 if len(valid_confidences) > 0 else 0

    # Correctness for calibration
    correctness = [1 if p == t else 0 for p, t in zip(valid_predictions, true_binary_labels)]
    ece, bin_stats = calculate_ece(valid_confidences, correctness)
    ece_percent = ece * 100

    # Pearson correlation between confidence and correctness
    if len(valid_confidences) > 1 and len(correctness) > 1:
        try:
            pearson_corr, p_value = pearsonr(valid_confidences, correctness)
            pearson_corr = pearson_corr * 100
        except:
            pearson_corr = 0
    else:
        pearson_corr = 0

    # Counts
    n_true_pass = np.sum(true_binary_labels)
    n_true_reject = len(true_binary_labels) - n_true_pass
    n_pred_pass = np.sum(valid_predictions)
    n_pred_reject = len(valid_predictions) - n_pred_pass

    return {
        'round': round_num,
        'step_name': STEP_NAMES.get(round_num, f'Step_{round_num}'),
        'n_samples': len(true_binary_labels),
        'n_true_pass': n_true_pass,
        'n_true_reject': n_true_reject,
        'n_pred_pass': n_pred_pass,
        'n_pred_reject': n_pred_reject,
        'accuracy': accuracy,
        'avg_confidence': avg_confidence,
        'ece': ece_percent,
        'pearson_corr': pearson_corr,
        'true_labels': true_binary_labels,
        'predictions': valid_predictions,
        'confidences': valid_confidences,
        'sample_indices': valid_sample_indices,
        'bin_stats': bin_stats
    }


# ============================================================================
# Excel Export Function
# ============================================================================

def export_round_results_to_excel(round_metrics, output_file):
    """
    Export all round results to a single Excel file with multiple sheets
    """
    print(f"\n📊 Exporting results to {output_file}...")

    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:

        # Sheet 1: Round Performance Summary
        print("  Writing Sheet 1: Round Performance Summary...")
        summary_data = []
        for rm in round_metrics:
            summary_data.append({
                'Step': rm['step_name'],
                'Round': rm['round'],
                'Samples': rm['n_samples'],
                'True Pass': rm['n_true_pass'],
                'True Reject': rm['n_true_reject'],
                'Pred Pass': rm['n_pred_pass'],
                'Pred Reject': rm['n_pred_reject'],
                'Accuracy (%)': round(rm['accuracy'], 2),
                'Avg Confidence (%)': round(rm['avg_confidence'], 2),
                'ECE (%)': round(rm['ece'], 2),
                'Pearson Correlation (%)': round(rm['pearson_corr'], 2)
            })
        df_summary = pd.DataFrame(summary_data)
        df_summary.to_excel(writer, sheet_name='Round Performance Summary', index=False)

        # Sheet 2: Calibration Statistics
        print("  Writing Sheet 2: Calibration Statistics...")
        calib_data = []
        for rm in round_metrics:
            for stat in rm['bin_stats']:
                calib_data.append({
                    'Step': rm['step_name'],
                    'Bin Center': round(stat['bin_center'], 3),
                    'Accuracy (%)': round(stat['accuracy'] * 100, 2),
                    'Confidence (%)': round(stat['confidence'] * 100, 2),
                    'Count': stat['count'],
                    'Weight': round(stat['weight'], 3)
                })
        if calib_data:
            df_calib = pd.DataFrame(calib_data)
            df_calib.to_excel(writer, sheet_name='Calibration Statistics', index=False)

        # Sheet 3: Detailed Binary Predictions
        print("  Writing Sheet 3: Detailed Binary Predictions...")
        for rm in round_metrics:
            if len(rm['true_labels']) > 0:
                detail_data = []
                for i in range(len(rm['true_labels'])):
                    detail_data.append({
                        'Sample Index': rm['sample_indices'][i],
                        'True Label': 'Pass' if rm['true_labels'][i] else 'Reject',
                        'Predicted': 'Pass' if rm['predictions'][i] else 'Reject',
                        'Confidence': round(rm['confidences'][i], 3),
                        'Correct': 'Yes' if rm['true_labels'][i] == rm['predictions'][i] else 'No'
                    })
                df_detail = pd.DataFrame(detail_data)
                sheet_name = f'{rm["step_name"].replace(" ", "_")[:25]}_binary_details'
                df_detail.to_excel(writer, sheet_name=sheet_name)

    print(f"✅ Exported to {output_file}")


# ============================================================================
# Chart Generation Functions
# ============================================================================

def create_performance_bar_charts(round_metrics, output_file):
    """
    Create a single combined figure with bar charts for accuracy, confidence, ECE, and Pearson
    """

    print(f"\n📊 Generating combined performance bar charts...")

    step_names = [rm['step_name'] for rm in round_metrics]
    accuracy = [rm['accuracy'] for rm in round_metrics]
    avg_conf = [rm['avg_confidence'] for rm in round_metrics]
    ece = [rm['ece'] for rm in round_metrics]
    pearson = [rm['pearson_corr'] for rm in round_metrics]
    n_samples = [rm['n_samples'] for rm in round_metrics]

    # Calculate overall metrics (average across rounds, weighted by samples)
    total_samples = sum(n_samples)
    overall_accuracy = np.average(accuracy, weights=n_samples) if total_samples > 0 else 0
    overall_avg_conf = np.average(avg_conf, weights=n_samples) if total_samples > 0 else 0
    overall_ece = np.average(ece, weights=n_samples) if total_samples > 0 else 0
    overall_pearson = np.average(pearson, weights=n_samples) if total_samples > 0 else 0

    # Create a single figure with 4 subplots in a 2x2 grid
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    x = np.arange(len(step_names))

    # ---- Subplot 1: Accuracy vs Confidence ----
    ax1 = axes[0, 0]
    width = 0.35

    bars1 = ax1.bar(x - width / 2, accuracy, width, label='Accuracy (%)',
                    color=VIRIDIS_PALETTE[4], edgecolor='black', linewidth=0.8)
    bars2 = ax1.bar(x + width / 2, avg_conf, width, label='Avg Confidence (%)',
                    color=VIRIDIS_PALETTE[6], edgecolor='black', linewidth=0.8)

    ax1.set_xlabel('Decision Step', fontsize=12)
    ax1.set_ylabel('Percentage (%)', fontsize=12)
    ax1.set_title('(a) Accuracy vs. Average Confidence by Step', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(step_names, rotation=15, ha='right')
    ax1.legend(loc='upper right', fontsize=10)
    ax1.set_ylim(0, 105)
    ax1.grid(axis='y', alpha=0.3)

    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        if height > 0:
            ax1.text(bar.get_x() + bar.get_width() / 2, height + 1,
                     f'{height:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    for bar in bars2:
        height = bar.get_height()
        if height > 0:
            ax1.text(bar.get_x() + bar.get_width() / 2, height + 1,
                     f'{height:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

    # ---- Subplot 2: ECE ----
    ax2 = axes[0, 1]
    bars3 = ax2.bar(x, ece, color=VIRIDIS_PALETTE[2], edgecolor='black', linewidth=0.8)
    ax2.set_xlabel('Decision Step', fontsize=12)
    ax2.set_ylabel('ECE (%)', fontsize=12)
    ax2.set_title('(b) Expected Calibration Error (ECE) by Step', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(step_names, rotation=15, ha='right')
    ax2.grid(axis='y', alpha=0.3)

    for bar in bars3:
        height = bar.get_height()
        if height > 0:
            ax2.text(bar.get_x() + bar.get_width() / 2, height + 0.5,
                     f'{height:.2f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

    # ---- Subplot 3: Pearson Correlation ----
    ax3 = axes[1, 0]
    colors_pearson = [VIRIDIS_PALETTE[6] if p >= 0 else VIRIDIS_PALETTE[2] for p in pearson]
    bars4 = ax3.bar(x, pearson, color=colors_pearson, edgecolor='black', linewidth=0.8)
    ax3.set_xlabel('Decision Step', fontsize=12)
    ax3.set_ylabel('Pearson Correlation (%)', fontsize=12)
    ax3.set_title('(c) Pearson Correlation (Confidence vs. Accuracy) by Step', fontsize=14, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(step_names, rotation=15, ha='right')
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax3.grid(axis='y', alpha=0.3)

    for bar in bars4:
        height = bar.get_height()
        if height != 0:
            va = 'bottom' if height >= 0 else 'top'
            y_offset = 1 if height >= 0 else -1
            ax3.text(bar.get_x() + bar.get_width() / 2, height + y_offset,
                     f'{height:+.1f}%', ha='center', va=va, fontsize=9, fontweight='bold')

    # ---- Subplot 4: Sample Count with Overall Metrics Box ----
    ax4 = axes[1, 1]
    bars5 = ax4.bar(x, n_samples, color=VIRIDIS_PALETTE[0], edgecolor='black', linewidth=0.8)
    ax4.set_xlabel('Decision Step', fontsize=12)
    ax4.set_ylabel('Number of Samples', fontsize=12)
    ax4.set_title('(d) Sample Count by Step', fontsize=14, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(step_names, rotation=15, ha='right')
    ax4.grid(axis='y', alpha=0.3)

    for bar in bars5:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width() / 2, height + 5,
                 f'{int(height)}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Add overall metrics box in the upper right corner of subplot d
    info_text = (
        f"Overall Performance Summary\n"
        f"{'─' * 30}\n"
        f"Overall Accuracy:  {overall_accuracy:.2f}%\n"
        f"Avg Confidence:    {overall_avg_conf:.2f}%\n"
        f"ECE:               {overall_ece:.2f}%\n"
        f"Pearson Corr:      {overall_pearson:.2f}%"
    )

    # Add a bounding box with the text
    ax4.text(0.98, 0.98, info_text,
             transform=ax4.transAxes,
             fontsize=10,
             verticalalignment='top',
             horizontalalignment='right',
             bbox=dict(boxstyle='round,pad=0.5',
                       facecolor='white',
                       edgecolor='black',
                       alpha=0.9,
                       linewidth=1.5))

    plt.tight_layout()
    plt.savefig(output_file, dpi=600, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✅ Saved: {output_file}")


# ============================================================================
# Print Summary Function
# ============================================================================

def print_summary(round_metrics):
    """Print a summary of the analysis results"""

    print("\n" + "=" * 80)
    print("BINARY DECISION TREE ANALYSIS - ROUND-BY-ROUND PERFORMANCE")
    print("=" * 80)

    print("\nRound Performance Summary:")
    print("-" * 80)
    print(f"{'Step':<20} {'Samples':<10} {'Accuracy':<12} {'Avg Conf':<12} {'ECE':<12} {'Pearson':<12}")
    print("-" * 80)

    for rm in round_metrics:
        print(f"{rm['step_name']:<20} {rm['n_samples']:<10} {rm['accuracy']:>6.1f}%     "
              f"{rm['avg_confidence']:>6.1f}%     {rm['ece']:>6.2f}%     {rm['pearson_corr']:>6.1f}%")

    print("-" * 80)

    # Calculate overall metrics
    total_samples = sum([rm['n_samples'] for rm in round_metrics])
    if total_samples > 0:
        overall_accuracy = np.average([rm['accuracy'] for rm in round_metrics],
                                      weights=[rm['n_samples'] for rm in round_metrics])
        overall_avg_conf = np.average([rm['avg_confidence'] for rm in round_metrics],
                                      weights=[rm['n_samples'] for rm in round_metrics])
        overall_ece = np.average([rm['ece'] for rm in round_metrics],
                                 weights=[rm['n_samples'] for rm in round_metrics])
        overall_pearson = np.average([rm['pearson_corr'] for rm in round_metrics],
                                     weights=[rm['n_samples'] for rm in round_metrics])

        print(f"\nOverall Performance (Weighted Average):")
        print(f"  Overall Accuracy:  {overall_accuracy:.2f}%")
        print(f"  Overall Avg Conf:  {overall_avg_conf:.2f}%")
        print(f"  Overall ECE:       {overall_ece:.2f}%")
        print(f"  Overall Pearson:   {overall_pearson:.2f}%")

    # Best performance
    best_acc = max(round_metrics, key=lambda x: x['accuracy'])
    best_ece = min(round_metrics, key=lambda x: x['ece'])
    best_pearson = max(round_metrics, key=lambda x: x['pearson_corr'])

    print("\nBest Performance:")
    print(f"  Best Accuracy: {best_acc['step_name']} ({best_acc['accuracy']:.1f}%)")
    print(f"  Best Calibration (lowest ECE): {best_ece['step_name']} ({best_ece['ece']:.2f}%)")
    print(f"  Best Confidence-Accuracy Correlation: {best_pearson['step_name']} ({best_pearson['pearson_corr']:.1f}%)")


# ============================================================================
# Main Function
# ============================================================================

def main():
    print("=" * 80)
    print("HIERARCHICAL BINARY DECISION TREE ANALYSIS")
    print("Binary Classification Performance for Each Decision Round")
    print("=" * 80)

    print(f"\n📁 Script directory: {SCRIPT_DIR}")
    print(f"📁 Base directory: {BASE_DIR}")
    print(f"📁 Input file: {INPUT_FILE}")
    print(f"📁 Output directory: {OUTPUT_DIR}")

    # Check if input file exists
    if not INPUT_FILE.exists():
        print(f"\n❌ Input file not found: {INPUT_FILE}")
        print("\n   Please ensure the file exists at:")
        print(f"   {INPUT_FILE}")
        return

    # Load data
    print("\n📁 Loading data...")
    df = load_data(INPUT_FILE)

    if df is None:
        return

    # ---- 直接用原始数据，不做任何重采样 ----
    print(f"\n📊 Using full dataset as-is: {len(df)} samples (no resampling)")

    # Detect maximum rounds from columns
    max_round = 0
    for col in df.columns:
        if col.startswith('step') and '_decision' in col:
            try:
                round_num = int(col.replace('step', '').replace('_decision', ''))
                if round_num > max_round:
                    max_round = round_num
            except:
                pass

    print(f"   Detected {max_round} rounds")

    # Calculate metrics for each round
    print("\n📊 Calculating binary classification metrics...")
    round_metrics = []

    for round_num in range(1, max_round + 1):
        rm = calculate_round_binary_metrics(df, round_num)
        if rm is not None and rm['n_samples'] > 0:
            round_metrics.append(rm)
            print(f"  {rm['step_name']}: {rm['n_samples']} samples, "
                  f"Accuracy = {rm['accuracy']:.2f}%, "
                  f"Avg Conf = {rm['avg_confidence']:.2f}%, "
                  f"ECE = {rm['ece']:.2f}%, "
                  f"Pearson = {rm['pearson_corr']:.2f}%")
        else:
            step_name = STEP_NAMES.get(round_num, f'Step_{round_num}')
            print(f"  {step_name}: No valid data")

    if not round_metrics:
        print("❌ No valid round data found!")
        return

    # Print summary
    print_summary(round_metrics)

    # Export results to Excel
    print("\n📁 Exporting results to Excel...")
    export_round_results_to_excel(round_metrics, OUTPUT_EXCEL)

    # Generate bar charts (single combined figure)
    print("\n📊 Generating bar charts...")
    create_performance_bar_charts(round_metrics, OUTPUT_CHART)

    print("\n" + "=" * 80)
    print("✅ ANALYSIS COMPLETE!")
    print("=" * 80)
    print(f"\n📁 Generated files in: {OUTPUT_DIR}")
    print(f"  1. {OUTPUT_EXCEL.name}")
    print(f"  2. {OUTPUT_CHART.name}")
    print("=" * 80)


if __name__ == "__main__":
    main()