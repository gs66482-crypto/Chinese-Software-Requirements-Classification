"""
Annotation-Enhanced LLM Classification Analysis
Round-by-Round Performance Analysis for Doubao with Smart Retry Mechanism
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from scipy.stats import pearsonr
import warnings
import os
import re
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

# ============================================================================
# Constants
# ============================================================================

# Get current script directory
try:
    SCRIPT_DIR = Path(__file__).parent.absolute()
except NameError:
    SCRIPT_DIR = Path(os.getcwd()).absolute()

# Project root directory (parent of analyze)
PROJECT_DIR = SCRIPT_DIR.parent

# Input file path
INPUT_FILE = PROJECT_DIR / "classifyOutcome" / "Annotation-Enhanced LLM Classification.xlsx"

# Output directory
OUTPUT_DIR = PROJECT_DIR / "analyzeOutcome" / "Annotation-guided LLM Classification"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Output file names
OUTPUT_EXCEL = OUTPUT_DIR / "Annotation_Enhanced_Round_Analysis_Results.xlsx"
OUTPUT_CHART = OUTPUT_DIR / "Annotation_Enhanced_Performance_Combined.png"
OUTPUT_REPORT = OUTPUT_DIR / "Annotation_Enhanced_Report.txt"

# NOTE: TARGET_SAMPLES is no longer hardcoded.
# It will be determined dynamically from the actual dataset size.


# ============================================================================
# Data Loading Functions
# ============================================================================

def load_annotation_data(file_path):
    """Load Annotation-Enhanced LLM Classification data from Excel file"""
    try:
        df = pd.read_excel(file_path)
        print(f"✅ Loaded {len(df)} samples from {file_path}")
        print(f"   Columns: {list(df.columns)}")
        return df
    except Exception as e:
        print(f"❌ Failed to load data: {e}")
        return None


def detect_available_rounds(df):
    """
    Detect which rounds have data available in the dataframe.
    Returns list of available round numbers.
    """
    available_rounds = []
    round_num = 1
    while True:
        pred_col = f'round{round_num}_pred'
        if pred_col in df.columns:
            available_rounds.append(round_num)
            round_num += 1
        else:
            break
    return available_rounds


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


def calculate_round_metrics(df, round_num):
    """
    Calculate all metrics for a specific round.

    Round 1: All samples
    Round 2: Only samples where Round 1 was incorrect (cascade flow)
    Round 3: Only samples where Round 2 was incorrect (cascade flow)
    """
    if round_num == 1:
        # Round 1: All samples with predictions
        pred_col = 'round1_pred'
        conf_col = 'round1_conf'
        valid_mask = df[pred_col].notna() & (df[pred_col] != 'N/A')
    else:
        # Round 2 and beyond: Only samples that reached this round
        # i.e., samples where the previous round was incorrect
        prev_pred_col = f'round{round_num - 1}_pred'
        prev_correct_col = f'round{round_num - 1}_correct'

        # Must have previous round prediction and it must be incorrect
        prev_valid_mask = df[prev_pred_col].notna() & (df[prev_pred_col] != 'N/A')
        prev_incorrect_mask = df[prev_correct_col] == 0

        # Current round must have prediction
        curr_pred_col = f'round{round_num}_pred'
        curr_valid_mask = df[curr_pred_col].notna() & (df[curr_pred_col] != 'N/A')

        # Combine: samples that reached this round (prev incorrect + curr has prediction)
        valid_mask = prev_valid_mask & prev_incorrect_mask & curr_valid_mask

    if valid_mask.sum() == 0:
        return None

    df_round = df[valid_mask].copy()

    # Get true labels
    true_labels = df_round['Human_Label'].astype(str).str.strip().str.lower()

    # Get predictions for this round
    if round_num == 1:
        pred_col = 'round1_pred'
        conf_col = 'round1_conf'
    else:
        pred_col = f'round{round_num}_pred'
        conf_col = f'round{round_num}_conf'

    predictions = df_round[pred_col].astype(str).str.strip().str.lower()

    # Get confidences
    if conf_col in df_round.columns:
        confidences = pd.to_numeric(df_round[conf_col], errors='coerce').fillna(0.5).values
    else:
        confidences = np.ones(len(df_round)) * 0.5

    # Calculate accuracy
    accuracy = accuracy_score(true_labels, predictions) * 100

    # Precision, Recall, F1 (macro average)
    try:
        precision = precision_score(true_labels, predictions, average='macro', zero_division=0) * 100
        recall = recall_score(true_labels, predictions, average='macro', zero_division=0) * 100
        f1 = f1_score(true_labels, predictions, average='macro', zero_division=0) * 100
    except:
        precision = recall = f1 = 0

    # Confidence statistics
    avg_confidence = np.mean(confidences) * 100 if len(confidences) > 0 else 0
    std_confidence = np.std(confidences) * 100 if len(confidences) > 0 else 0
    min_confidence = np.min(confidences) * 100 if len(confidences) > 0 else 0
    max_confidence = np.max(confidences) * 100 if len(confidences) > 0 else 0

    # Correctness for ECE and Pearson
    correctness = (true_labels == predictions).astype(int).values

    # ECE
    ece, bin_stats = calculate_ece(confidences, correctness)
    ece_percent = ece * 100

    # Pearson correlation
    if len(confidences) > 1 and len(correctness) > 1:
        try:
            if len(set(confidences)) == 1:
                pearson_corr = 0
            else:
                pearson_corr, p_value = pearsonr(confidences, correctness)
                if np.isnan(pearson_corr):
                    pearson_corr = 0
        except Exception as e:
            pearson_corr = 0
    else:
        pearson_corr = 0

    # Confusion matrix
    all_labels = sorted(set(true_labels) | set(predictions))
    cm = confusion_matrix(true_labels, predictions, labels=all_labels)

    # Sample count
    n_samples = len(df_round)

    return {
        'round': round_num,
        'n_samples': n_samples,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'avg_confidence': avg_confidence,
        'std_confidence': std_confidence,
        'min_confidence': min_confidence,
        'max_confidence': max_confidence,
        'ece': ece_percent,
        'pearson_corr': pearson_corr * 100,
        'confusion_matrix': cm,
        'confusion_labels': all_labels,
        'true_labels': true_labels.tolist(),
        'predictions': predictions.tolist(),
        'confidences': confidences.tolist(),
        'bin_stats': bin_stats
    }


# ============================================================================
# Excel Export Function
# ============================================================================

def export_results_to_excel(round_metrics, output_file):
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
                'Round': rm['round'],
                'Samples': rm['n_samples'],
                'Accuracy (%)': round(rm['accuracy'], 2),
                'Precision (%)': round(rm['precision'], 2),
                'Recall (%)': round(rm['recall'], 2),
                'F1 Score (%)': round(rm['f1'], 2),
                'Avg Confidence (%)': round(rm['avg_confidence'], 2),
                'Std Confidence (%)': round(rm['std_confidence'], 2),
                'ECE (%)': round(rm['ece'], 2),
                'Pearson Correlation (%)': round(rm['pearson_corr'], 2)
            })
        df_summary = pd.DataFrame(summary_data)
        df_summary.to_excel(writer, sheet_name='Round Performance Summary', index=False)

        # Sheet 2: Confusion Matrices
        print("  Writing Sheet 2: Confusion Matrices...")
        for rm in round_metrics:
            cm = rm['confusion_matrix']
            labels = rm['confusion_labels']
            df_cm = pd.DataFrame(cm, index=labels, columns=labels)
            sheet_name = f'Round_{rm["round"]}_CM'[:31]
            df_cm.to_excel(writer, sheet_name=sheet_name)

        # Sheet 3: Calibration Statistics
        print("  Writing Sheet 3: Calibration Statistics...")
        calib_data = []
        for rm in round_metrics:
            for stat in rm['bin_stats']:
                calib_data.append({
                    'Round': rm['round'],
                    'Bin Center': round(stat['bin_center'], 3),
                    'Accuracy (%)': round(stat['accuracy'] * 100, 2),
                    'Confidence (%)': round(stat['confidence'] * 100, 2),
                    'Count': stat['count'],
                    'Weight': round(stat['weight'], 3)
                })
        if calib_data:
            df_calib = pd.DataFrame(calib_data)
            df_calib.to_excel(writer, sheet_name='Calibration Statistics', index=False)

        # Sheet 4: Raw Data
        print("  Writing Sheet 4: Raw Data...")
        df_raw = pd.read_excel(INPUT_FILE)
        df_raw.to_excel(writer, sheet_name='Raw Data', index=False)

    print(f"✅ Exported to {output_file}")


# ============================================================================
# Combined Chart Generation Function
# ============================================================================

def create_combined_chart(round_metrics, output_file):
    """
    Create a combined figure with two subplots:
    - Top: Performance bar chart (sample count + metrics)
    - Bottom: Confusion matrices for each round
    """

    print(f"\n📊 Generating combined chart...")

    # Filter out rounds with no data
    valid_rounds = [rm for rm in round_metrics if rm['n_samples'] > 0]

    if len(valid_rounds) == 0:
        print("   No valid data to display")
        return

    # Create figure with 2 rows
    fig = plt.figure(figsize=(16, 14))

    # Define grid: 2 rows, 1 column
    gs = fig.add_gridspec(2, 1, height_ratios=[1.2, 1], hspace=0.3)

    # ============================================================
    # (a) Performance Bar Chart - Top subplot
    # ============================================================
    ax1 = fig.add_subplot(gs[0])

    rounds = [rm['round'] for rm in valid_rounds]
    n_samples = [rm['n_samples'] for rm in valid_rounds]
    accuracy = [rm['accuracy'] for rm in valid_rounds]
    avg_conf = [rm['avg_confidence'] for rm in valid_rounds]
    ece = [rm['ece'] for rm in valid_rounds]
    pearson = [rm['pearson_corr'] for rm in valid_rounds]

    x = np.arange(len(rounds))

    # Number of bar groups
    n_bars = 5
    total_width = 0.8
    width = total_width / n_bars
    start_pos = -total_width / 2
    positions = [start_pos + i * width for i in range(n_bars)]

    # Primary Y-axis (Left): Sample Count
    bars1 = ax1.bar(x + positions[0], n_samples, width, label='Sample Count',
                    color=VIRIDIS_PALETTE[0], edgecolor='black', linewidth=0.8, zorder=2)
    ax1.set_xlabel('Interaction Round', fontsize=13)
    ax1.set_ylabel('Number of Samples', fontsize=13, color='black')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'Round {r}' for r in rounds], fontsize=11)
    ax1.set_ylim(-max(n_samples) * 0.05, max(n_samples) * 1.15)
    ax1.grid(axis='y', alpha=0.3, zorder=0)

    # Add value labels for sample count bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, height + max(n_samples) * 0.02,
                 f'{int(height):,}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Secondary Y-axis (Right): Performance Metrics
    ax2 = ax1.twinx()

    bars2 = ax2.bar(x + positions[1], accuracy, width, label='Accuracy',
                    color=VIRIDIS_PALETTE[3], edgecolor='black', linewidth=0.8, zorder=2)
    bars3 = ax2.bar(x + positions[2], avg_conf, width, label='Avg Confidence',
                    color=VIRIDIS_PALETTE[5], edgecolor='black', linewidth=0.8, zorder=2)
    bars4 = ax2.bar(x + positions[3], ece, width, label='ECE',
                    color=VIRIDIS_PALETTE[7], edgecolor='black', linewidth=0.8, zorder=2)
    bars5 = ax2.bar(x + positions[4], pearson, width, label='Pearson Correlation',
                    color=VIRIDIS_PALETTE[9], edgecolor='black', linewidth=0.8, zorder=2)

    ax2.set_ylabel('Percentage (%)', fontsize=13, color='black')
    max_val = max(max(accuracy), max(avg_conf), max(ece), max(pearson))
    ax2.set_ylim(-max_val * 0.05, max_val * 1.15)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5, zorder=1)

    # Add value labels for performance metrics bars
    for bars in [bars2, bars3, bars4, bars5]:
        for bar in bars:
            height = bar.get_height()
            if not np.isnan(height):
                va = 'bottom' if height >= 0 else 'top'
                y_offset = max_val * 0.01 if height >= 0 else -max_val * 0.01
                if height == 0:
                    label_text = '0.0%'
                    y_offset = -max_val * 0.02 if height >= 0 else max_val * 0.02
                else:
                    label_text = f'{height:.1f}%'
                ax2.text(bar.get_x() + bar.get_width() / 2, height + y_offset,
                         label_text, ha='center', va=va, fontsize=7, fontweight='bold')

    # Legend for top subplot
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    all_lines = lines1 + lines2
    all_labels = labels1 + labels2

    ax1.legend(all_lines, all_labels,
               loc='upper right',
               bbox_to_anchor=(0.98, 0.98),
               fontsize=9,
               framealpha=0.85,
               edgecolor='black',
               fancybox=False,
               handlelength=1.0,
               handletextpad=0.5,
               borderpad=0.5,
               labelspacing=0.4)

    # Add (a) label
    ax1.text(-0.08, 1.02, '(a)', transform=ax1.transAxes,
             fontsize=16, fontweight='bold', va='bottom', ha='right')

    ax1.set_title('Round-by-Round Performance Metrics', fontsize=14, fontweight='bold', pad=10)

    # ============================================================
    # (b) Confusion Matrices - Bottom subplot
    # ============================================================

    n_rounds = len(valid_rounds)

    # Create subplot for confusion matrices
    ax_cm = fig.add_subplot(gs[1])

    # For multiple rounds, we'll create side-by-side heatmaps
    if n_rounds == 1:
        # Single round - center it
        axs_cm = [ax_cm]
    else:
        # Multiple rounds - split the axes
        # Clear the subplot and create new ones
        fig.delaxes(ax_cm)
        # Create a grid for confusion matrices within the bottom area
        cm_gs = gs[1].subgridspec(1, n_rounds, wspace=0.3)
        axs_cm = [fig.add_subplot(cm_gs[i]) for i in range(n_rounds)]

    for idx, rm in enumerate(valid_rounds):
        ax = axs_cm[idx]
        cm = rm['confusion_matrix']
        labels = rm['confusion_labels']

        if cm.size > 0 and cm.sum() > 0:
            sns.heatmap(cm, annot=True, fmt='d', cmap='viridis',
                        xticklabels=labels, yticklabels=labels,
                        ax=ax, cbar=True, cbar_kws={'label': 'Count'})
        else:
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)

        ax.set_xlabel('Predicted Label', fontsize=11)
        ax.set_ylabel('True Label', fontsize=11)

        # Add description of sample set
        if rm['round'] == 1:
            sample_desc = f'All samples (N={rm["n_samples"]:,})'
        else:
            sample_desc = f'Errors from Round {rm["round"] - 1} (N={rm["n_samples"]:,})'

        ax.set_title(f'Round {rm["round"]}\n{sample_desc}\nAccuracy: {rm["accuracy"]:.1f}%',
                     fontsize=11, fontweight='bold')
        ax.set_xticklabels(labels, rotation=45, ha='right')

    # Add (b) label to the first confusion matrix subplot
    axs_cm[0].text(-0.12, 1.02, '(b)', transform=axs_cm[0].transAxes,
                   fontsize=16, fontweight='bold', va='bottom', ha='right')

    # Overall figure title
    fig.suptitle('Annotation-Enhanced LLM Classification: Performance Analysis',
                 fontsize=16, fontweight='bold', y=0.98)

    plt.tight_layout()
    plt.savefig(output_file, dpi=600, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✅ Saved: {output_file}")


def create_text_report(round_metrics, output_file, target_samples):
    """Generate text report"""

    print(f"\n📄 Generating text report...")

    # Filter out rounds with no data
    valid_rounds = [rm for rm in round_metrics if rm['n_samples'] > 0]

    lines = []
    lines.append("=" * 70)
    lines.append("ANNOTATION-ENHANCED LLM CLASSIFICATION - ROUND-BY-ROUND REPORT")
    lines.append("=" * 70)
    lines.append(f"Model: Doubao-seed-1.6-250615")
    lines.append(f"Role: Software Requirement Classification Expert")
    lines.append(f"Total Samples: {target_samples:,}")
    lines.append(f"Smart Retry: Max 3 rounds")
    lines.append("")
    lines.append("NOTE: Round 1 = Initial classification (all samples)")
    lines.append("      Round 2 = Rethinking (only samples incorrect in Round 1)")
    lines.append("      Round 3 = Final thinking (only samples incorrect in Round 2)")
    lines.append("")

    lines.append("-" * 60)
    lines.append("PERFORMANCE SUMMARY BY ROUND")
    lines.append("-" * 60)
    lines.append(f"{'Round':<10} {'Samples':<12} {'Accuracy':<12} {'Avg Conf':<12} {'ECE':<12} {'Pearson':<12}")
    lines.append("-" * 70)

    for rm in valid_rounds:
        pearson_str = f"{rm['pearson_corr']:.2f}%" if rm['pearson_corr'] != 0 else "0.00%"
        lines.append(f"Round {rm['round']:<5} {rm['n_samples']:<12} {rm['accuracy']:.2f}%       "
                     f"{rm['avg_confidence']:.2f}%     {rm['ece']:.2f}%     {pearson_str}")

    # Calculate cumulative accuracy
    lines.append("")
    lines.append("-" * 60)
    lines.append("CUMULATIVE PERFORMANCE")
    lines.append("-" * 60)

    if len(valid_rounds) >= 1:
        # Round 1: accuracy on all samples that participated in Round 1
        r1 = valid_rounds[0]
        base_samples = r1['n_samples']  # 实际参与第一轮的样本数（作为累计准确率分母）
        initial_correct = r1['n_samples'] * (r1['accuracy'] / 100)
        lines.append(f"Round 1: {r1['accuracy']:.2f}% accuracy on all {r1['n_samples']:,} samples")
        lines.append(
            f"  → {int(initial_correct):,} samples correct, {int(r1['n_samples'] - initial_correct):,} samples incorrect")

        # Calculate cumulative accuracy: errors fixed after each round
        if len(valid_rounds) >= 2:
            r2 = valid_rounds[1]
            # The samples in Round 2 are errors from Round 1
            # The number of samples that remain incorrect after Round 2
            r2_incorrect = r2['n_samples'] * (1 - r2['accuracy'] / 100)

            # Samples fixed in Round 2
            fixed_in_r2 = r2['n_samples'] * (r2['accuracy'] / 100)
            cumulative_correct = r1['n_samples'] * (r1['accuracy'] / 100) + fixed_in_r2
            cumulative_accuracy = (cumulative_correct / base_samples) * 100

            lines.append(f"Round 2: Fixed {int(fixed_in_r2):,} samples, "
                         f"cumulative accuracy = {cumulative_accuracy:.2f}%")
            lines.append(f"  → {int(cumulative_correct):,} samples correct, "
                         f"{int(base_samples - cumulative_correct):,} samples remain incorrect")

            if len(valid_rounds) >= 3:
                r3 = valid_rounds[2]
                fixed_in_r3 = r3['n_samples'] * (r3['accuracy'] / 100)
                cumulative_correct = r1['n_samples'] * (r1['accuracy'] / 100) + fixed_in_r2 + fixed_in_r3
                cumulative_accuracy = (cumulative_correct / base_samples) * 100
                lines.append(f"Round 3: Fixed {int(fixed_in_r3):,} samples, "
                             f"cumulative accuracy = {cumulative_accuracy:.2f}%")

    lines.append("")
    lines.append("-" * 60)
    lines.append("KEY OBSERVATIONS")
    lines.append("-" * 60)

    if valid_rounds:
        # Sample reduction
        if len(valid_rounds) >= 2:
            r1 = valid_rounds[0]
            r2 = valid_rounds[1]
            reduction = r1['n_samples'] - r2['n_samples']
            lines.append(f"• Sample reduction from Round 1 to Round 2: {reduction:,} samples "
                         f"({reduction / r1['n_samples'] * 100:.1f}% of total)")

        # Find best round by accuracy
        best_round = max(valid_rounds, key=lambda x: x['accuracy'])
        lines.append(f"• Highest accuracy: Round {best_round['round']} ({best_round['accuracy']:.2f}%)")

        # Find round with best calibration (lowest ECE)
        best_cal = min(valid_rounds, key=lambda x: x['ece'])
        lines.append(f"• Best calibration (lowest ECE): Round {best_cal['round']} ({best_cal['ece']:.2f}%)")

        # Find round with best confidence-accuracy correlation
        valid_pearson = [rm for rm in valid_rounds if rm['pearson_corr'] != 0]
        if valid_pearson:
            best_corr = max(valid_pearson, key=lambda x: x['pearson_corr'])
            lines.append(f"• Best confidence-accuracy correlation: Round {best_corr['round']} "
                         f"({best_corr['pearson_corr']:.2f}%)")
        else:
            lines.append("• No valid confidence-accuracy correlation found")

        # Improvement from Round 1 to Round 2
        if len(valid_rounds) >= 2:
            r1 = valid_rounds[0]
            r2 = valid_rounds[1]
            improvement = r2['accuracy'] - r1['accuracy']
            if improvement > 0:
                lines.append(f"• Accuracy improvement on re-evaluated samples: +{improvement:.2f}%")
            else:
                lines.append(f"• Accuracy change on re-evaluated samples: {improvement:.2f}%")

    lines.append("")
    lines.append("=" * 70)

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"✅ Saved: {output_file}")


# ============================================================================
# Main Function
# ============================================================================

def main():
    print("=" * 70)
    print("ANNOTATION-ENHANCED LLM CLASSIFICATION - ROUND-BY-ROUND ANALYSIS")
    print("=" * 70)

    print(f"\n📁 Working directory: {SCRIPT_DIR}")
    print(f"📁 Project root: {PROJECT_DIR}")
    print(f"📁 Input file: {INPUT_FILE}")
    print(f"📁 Output directory: {OUTPUT_DIR}")

    # Check if input file exists
    if not INPUT_FILE.exists():
        print(f"\n❌ Input file not found: {INPUT_FILE}")
        print("Please ensure the following directory structure:")
        print(f"   {PROJECT_DIR}/")
        print(f"   ├── analyze/")
        print(f"   │   └── this_script.py")
        print(f"   ├── classifyOutcome/")
        print(f"   │   └── Annotation-Enhanced LLM Classification.xlsx")
        print(f"   └── analyzeOutcome/")
        print(f"       └── Annotation-guided LLM Classification/")
        print(f"           └── (output files will be created here)")
        return

    # Load data
    print("\n📁 Loading Annotation-Enhanced data...")
    df = load_annotation_data(INPUT_FILE)

    if df is None:
        print("❌ Failed to load data. Exiting.")
        return

    # Determine target samples dynamically from the actual dataset
    target_samples = len(df)
    print(f"\n📊 Total samples in dataset: {target_samples:,}")

    # Check required columns
    required_cols = ['Human_Label', 'Model_Prediction', 'round',
                     'round1_pred', 'round1_conf']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"❌ Missing required columns: {missing_cols}")
        print(f"   Available columns: {list(df.columns)}")
        return

    # Detect available rounds
    available_rounds = detect_available_rounds(df)
    print(f"\n📊 Detected rounds: {available_rounds}")

    # Calculate metrics for each available round
    print("\n📊 Calculating round-by-round metrics...")
    round_metrics = []

    for round_num in available_rounds:
        rm = calculate_round_metrics(df, round_num)
        if rm is not None and rm['n_samples'] > 0:
            round_metrics.append(rm)
            pearson_str = f"{rm['pearson_corr']:.2f}%" if rm['pearson_corr'] != 0 else "0.00%"
            if round_num == 1:
                sample_desc = f"All {rm['n_samples']:,} samples"
            else:
                sample_desc = f"{rm['n_samples']:,} errors from Round {round_num - 1}"
            print(f"  Round {round_num}: {sample_desc}, "
                  f"Accuracy = {rm['accuracy']:.2f}%, "
                  f"ECE = {rm['ece']:.2f}%, "
                  f"Pearson = {pearson_str}")
        else:
            print(f"  Round {round_num}: No valid data")

    if not round_metrics:
        print("❌ No valid round data found!")
        return

    # Export results to Excel
    print("\n📁 Exporting results to Excel...")
    export_results_to_excel(round_metrics, OUTPUT_EXCEL)

    # Generate combined chart
    print("\n📊 Generating combined chart...")
    create_combined_chart(round_metrics, OUTPUT_CHART)

    # Generate text report
    print("\n📄 Generating text report...")
    create_text_report(round_metrics, OUTPUT_REPORT, target_samples)

    print("\n" + "=" * 70)
    print("✅ ANALYSIS COMPLETE!")
    print("=" * 70)
    print(f"\n📁 Output directory: {OUTPUT_DIR}")
    print(f"\nGenerated files:")
    print(f"  1. {OUTPUT_EXCEL.name}")
    print(f"  2. {OUTPUT_CHART.name}")
    print(f"  3. {OUTPUT_REPORT.name}")
    print("=" * 70)


if __name__ == "__main__":
    main()