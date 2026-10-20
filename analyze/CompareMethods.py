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

# ========== Set high-resolution figure parameters ==========
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 600
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# ========== Define custom color palette ==========
CUSTOM_COLORS = {
    'Accuracy': '#440154',  # Dark purple
    'Confidence': '#3B528B',  # Purple-blue
    'ECE': '#21908D',  # Teal
    'Pearson': '#35B779'  # Green
}

# ========== Define input and output paths ==========
script_dir = Path(__file__).parent.absolute()
project_dir = script_dir.parent
input_file = project_dir / "classifyOutcome" / "summarizedData.xlsx"
output_dir = project_dir / "analyzeOutcome" / "methodsCompare"
output_dir.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("Loading data...")
print("=" * 80)
print(f"Input file: {input_file}")

# ========== Read all sheets from the Excel file ==========
xl = pd.ExcelFile(input_file)
sheet_names = xl.sheet_names
print(f"Found {len(sheet_names)} sheets: {sheet_names}")

# Dictionary to store all dataframes
all_dfs = {}

for sheet in sheet_names:
    df = pd.read_excel(input_file, sheet_name=sheet)
    # Skip empty sheets
    if len(df) == 0:
        print(f"  {sheet}: {len(df)} records (skipped - empty)")
        continue
    all_dfs[sheet] = df
    print(f"  {sheet}: {len(df)} records")


# ========== Define metric calculation function ==========
def calculate_metrics(df, human_col='Human_Label', pred_col='Model_Prediction', conf_col='prediction_confidence'):
    """
    Calculate accuracy, average confidence, ECE, and Pearson correlation coefficient.
    """
    # Check if required columns exist
    if human_col not in df.columns or pred_col not in df.columns:
        return None

    # Clean data
    df_clean = df.dropna(subset=[human_col, pred_col])

    if len(df_clean) == 0:
        return None

    # Convert to lowercase strings
    y_true = df_clean[human_col].astype(str).str.strip().str.lower()
    y_pred = df_clean[pred_col].astype(str).str.strip().str.lower()

    # Filter out invalid predictions
    invalid_vals = ['unknown', '不通过', '通过', '', 'nan', 'none']
    valid_mask = ~y_pred.isin(invalid_vals)
    y_true = y_true[valid_mask]
    y_pred = y_pred[valid_mask]

    if len(y_true) == 0:
        return None

    # Calculate accuracy
    acc = accuracy_score(y_true, y_pred)

    # Get confidence column
    if conf_col in df_clean.columns:
        conf_vals = pd.to_numeric(df_clean[conf_col], errors='coerce')
        conf_vals = conf_vals[valid_mask]
    else:
        # Try to find a confidence column
        conf_col_found = None
        for col in df_clean.columns:
            if 'confidence' in col.lower() or 'conf' in col.lower():
                conf_col_found = col
                break
        if conf_col_found:
            conf_vals = pd.to_numeric(df_clean[conf_col_found], errors='coerce')
            conf_vals = conf_vals[valid_mask]
        else:
            conf_vals = pd.Series([np.nan] * len(y_true))

    # Average confidence (normalize to [0,1] if needed)
    avg_conf = conf_vals.mean() if not conf_vals.isna().all() else 0
    if avg_conf > 1:
        conf_vals = conf_vals / 100
        avg_conf = avg_conf / 100

    # Calculate correctness
    correctness = (y_true == y_pred).astype(float)

    # ECE calculation
    def calculate_ece(confidences, correctness, n_bins=10):
        if len(confidences) == 0 or confidences.isna().all():
            return 0

        confidences = np.array(confidences)
        correctness = np.array(correctness)

        bins = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(confidences, bins, right=False)
        bin_indices = np.clip(bin_indices, 1, n_bins)

        ece = 0
        total_samples = len(confidences)

        for i in range(1, n_bins + 1):
            mask = (bin_indices == i)
            bin_count = np.sum(mask)
            if bin_count == 0:
                continue
            bin_acc = np.mean(correctness[mask])
            bin_conf = np.mean(confidences[mask])
            ece += (bin_count / total_samples) * abs(bin_acc - bin_conf)

        return ece

    ece = calculate_ece(conf_vals, correctness)

    # Pearson correlation
    if len(conf_vals) > 1 and len(set(correctness)) > 1 and len(set(conf_vals)) > 1:
        pearson_corr, _ = pearsonr(conf_vals, correctness)
    else:
        pearson_corr = 0

    return {
        'accuracy': acc,
        'avg_confidence': avg_conf,
        'ece': ece,
        'pearson': pearson_corr,
        'n_samples': len(y_true)
    }


print("\n" + "=" * 80)
print("Calculating performance metrics for each method...")
print("=" * 80)

# ========== Calculate metrics for each sheet ==========
results = {}
for sheet_name, df in all_dfs.items():
    metrics = calculate_metrics(df)
    if metrics is not None:
        results[sheet_name] = metrics
        print(f"  {sheet_name}: Acc={metrics['accuracy'] * 100:.1f}%, N={metrics['n_samples']}")
    else:
        print(f"  {sheet_name}: Skipped (no valid data or missing columns)")

# ========== Check if we have any results ==========
if len(results) == 0:
    print("\n❌ No valid data found. Exiting.")
    exit()

# ========== Sort by accuracy (descending) ==========
sorted_methods = sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True)

# Extract sorted data
method_names = [item[0] for item in sorted_methods]
metrics_list = [item[1] for item in sorted_methods]

# Extract metric values
accuracies = [m['accuracy'] * 100 for m in metrics_list]
confidences = [m['avg_confidence'] * 100 for m in metrics_list]
eces = [m['ece'] * 100 for m in metrics_list]
pearsons = [m['pearson'] * 100 for m in metrics_list]

print("\n" + "=" * 80)
print("Sorted results by accuracy:")
print("=" * 80)
for i, (name, metrics) in enumerate(sorted_methods, 1):
    print(f"{i}. {name}: {metrics['accuracy'] * 100:.1f}%")

# ========== Generate grouped bar chart ==========
print("\n" + "=" * 80)
print("Generating grouped bar chart...")
print("=" * 80)

fig, ax = plt.subplots(figsize=(16, 10))

# Create short labels
method_labels = [f"{i + 1}. {name}" for i, name in enumerate(method_names)]

x = np.arange(len(method_labels))
width = 0.18

metric_configs = [
    ('Accuracy', accuracies, CUSTOM_COLORS['Accuracy'], 'Accuracy (%)'),
    ('Avg Confidence', confidences, CUSTOM_COLORS['Confidence'], 'Confidence (%)'),
    ('ECE', eces, CUSTOM_COLORS['ECE'], 'ECE (%)'),
    ('Pearson', pearsons, CUSTOM_COLORS['Pearson'], 'Pearson Correlation (%)')
]

# Plot each metric as a group of bars
for i, (metric_name, values, color, label) in enumerate(metric_configs):
    offset = width * (i - 1.5)
    bars = ax.bar(x + offset, values, width, label=label,
                  color=color, edgecolor='black', linewidth=0.8)

    # Add value labels on top of bars
    for bar, val in zip(bars, values):
        if metric_name == 'ECE':
            text = f'{val:.2f}'
        else:
            text = f'{val:.1f}'

        if val >= 0:
            va = 'bottom'
            y_offset = 0.5
        else:
            va = 'top'
            y_offset = -0.5

        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + y_offset,
                text, ha='center', va=va, fontsize=8, fontweight='bold')

# Configure axes
ax.set_xlabel('Methods', fontsize=14, fontweight='bold')
ax.set_ylabel('Percentage (%)', fontsize=14, fontweight='bold')
ax.set_title('Experimental Methods Comparison for Software Requirement Classification',
             fontsize=16, fontweight='bold', pad=20)

ax.set_xticks(x)
ax.set_xticklabels(method_labels, fontsize=10, ha='center')
ax.set_ylim(-20, 105)
ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8)

# Legend at upper right
ax.legend(loc='upper right', fontsize=11, framealpha=0.9, edgecolor='black')

plt.tight_layout()
plt.savefig(output_dir / 'Experimental_Methods_Comparison_Grouped.png',
            dpi=600, bbox_inches='tight', facecolor='white')
plt.close()

print(f"✅ Saved: {output_dir / 'Experimental_Methods_Comparison_Grouped.png'}")

# ========== Generate enhanced bar chart ==========
print("\nGenerating enhanced bar chart...")

fig2, ax2 = plt.subplots(figsize=(16, 10))

x = np.arange(len(method_labels))
width = 0.18

metric_configs_enhanced = [
    ('Accuracy', accuracies, CUSTOM_COLORS['Accuracy'], 'Accuracy (%)'),
    ('Confidence', confidences, CUSTOM_COLORS['Confidence'], 'Confidence (%)'),
    ('ECE', eces, CUSTOM_COLORS['ECE'], 'ECE (%)'),
    ('Pearson', pearsons, CUSTOM_COLORS['Pearson'], 'Pearson (%)')
]

for i, (name, values, color, label) in enumerate(metric_configs_enhanced):
    offset = width * (i - 1.5)
    bars = ax2.bar(x + offset, values, width, label=label,
                   color=color, edgecolor='black', linewidth=0.8, alpha=0.85)

    for bar, val in zip(bars, values):
        if name == 'ECE':
            text = f'{val:.2f}'
        else:
            text = f'{val:.1f}'

        y_pos = bar.get_height()
        if y_pos >= 0:
            va = 'bottom'
            offset_y = 0.8
        else:
            va = 'top'
            offset_y = -0.8

        ax2.text(bar.get_x() + bar.get_width() / 2, y_pos + offset_y,
                 text, ha='center', va=va, fontsize=8, fontweight='bold')

ax2.set_xlabel('Methods', fontsize=14, fontweight='bold')
ax2.set_ylabel('Percentage (%)', fontsize=14, fontweight='bold')
ax2.set_title('Experimental Methods Comparison for Software Requirement Classification',
              fontsize=16, fontweight='bold', pad=20)

ax2.set_xticks(x)
ax2.set_xticklabels(method_labels, fontsize=10, ha='center')
ax2.set_ylim(-25, 105)
ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
ax2.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8)

# Legend at upper right
ax2.legend(loc='upper right', fontsize=11, framealpha=0.9, edgecolor='black')

plt.tight_layout()
plt.savefig(output_dir / 'Experimental_Methods_Comparison_Enhanced.png',
            dpi=600, bbox_inches='tight', facecolor='white')
plt.close()

print(f"✅ Saved: {output_dir / 'Experimental_Methods_Comparison_Enhanced.png'}")

# ========== Generate summary table ==========
print("\n" + "=" * 80)
print("Performance Summary Table (sorted by accuracy):")
print("=" * 80)

summary_data = []
for i, (name, metrics) in enumerate(sorted_methods, 1):
    summary_data.append({
        'Rank': i,
        'Method': name,
        'Accuracy (%)': round(metrics['accuracy'] * 100, 2),
        'Avg Confidence (%)': round(metrics['avg_confidence'] * 100, 2),
        'ECE (%)': round(metrics['ece'] * 100, 4),
        'Pearson Correlation (%)': round(metrics['pearson'] * 100, 4),
        'N Samples': metrics['n_samples']
    })

df_summary = pd.DataFrame(summary_data)
print(df_summary.to_string(index=False))
df_summary.to_excel(output_dir / 'Experimental_Methods_Summary.xlsx', index=False)
print(f"\n✅ Saved: {output_dir / 'Experimental_Methods_Summary.xlsx'}")

print("\n" + "=" * 80)
print("Analysis complete!")
print("=" * 80)
print(f"\nOutput directory: {output_dir}")
print("\nGenerated files:")
print("  1. Experimental_Methods_Comparison_Grouped.png")
print("  2. Experimental_Methods_Comparison_Enhanced.png")
print("  3. Experimental_Methods_Summary.xlsx")
print("\nColor palette:")
print("  - Accuracy:   #440154 (Dark Purple)")
print("  - Confidence: #3B528B (Purple-blue)")
print("  - ECE:        #21908D (Teal)")
print("  - Pearson:    #35B779 (Green)")
print("\nLegend position: Upper right")
print("=" * 80)