import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix
from scipy.stats import pearsonr
import warnings
from pathlib import Path
from matplotlib.patches import Rectangle
from matplotlib.gridspec import GridSpec

warnings.filterwarnings('ignore')

# ============================================================
# 0. 图表样式设置
# ============================================================

plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9

# Viridis color palette
VIRIDIS_PALETTE = [
    '#440154', '#482878', '#3e4989', '#31688e',
    '#26828e', '#1f9e89', '#35b779', '#6ece58',
    '#b5de2b', '#fde725'
]

# ============================================================
# 1. 配置 - 路径设置
# ============================================================

# 获取当前脚本所在目录
try:
    SCRIPT_DIR = Path(__file__).parent.absolute()
except NameError:
    SCRIPT_DIR = Path(os.getcwd()).absolute()

# 根目录 = 当前脚本所在目录的上一级
BASE_DIR = SCRIPT_DIR.parent

# 输入文件路径
INPUT_DIR = BASE_DIR / "classifyOutcome" / "Single-model One-shot Inference"

# 输出文件路径
OUTPUT_DIR = BASE_DIR / "analyzeOutcome" / "Single-model One-shot Inference"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 图表输出目录
CHART_DIR = OUTPUT_DIR

# 模型列表（按正确顺序）
MODEL_NAMES = [
    "gpt",
    "doubao",
    "qwen",
    "hunyuan",
    "deepseek",
    "kimi",
    "glm"
]

# 模型显示名称映射
MODEL_DISPLAY_NAMES = {
    "gpt": "GPT-4.1 nano",
    "doubao": "Doubao-seed-1.6",
    "qwen": "Qwen-Plus",
    "hunyuan": "Hunyuan T1",
    "deepseek": "DeepSeek-V3.1",
    "kimi": "Kimi-K2-Instruct",
    "glm": "GLM-5.2"
}

# 文件名模板
FILE_TEMPLATE = "Single-Pass Classification-{model}.xlsx"

# 列名配置
REQUIREMENT_COL = "requirement"
HUMAN_LABEL_COL = "Human_Label"
PREDICTION_COL = "Model_Prediction"
CONFIDENCE_COL = "prediction_confidence"
CORRECT_COL = "是否正确"

# NOTE: TARGET_SAMPLES is no longer hardcoded.
# The analysis now uses the actual dataset size (len(df_valid)) without any rescaling.

# 模型成本数据
COST_DATA = {
    "GPT-4.1 nano": 2.88,
    "Doubao-seed-1.6": 2.00,
    "Qwen-Plus": 16.00,
    "Hunyuan T1": 4.00,
    "DeepSeek-V3.1": 7.92,
    "Kimi-K2-Instruct": 16.00,
    "GLM-5.2": 28.00,
}

INPUT_COST_DATA = {
    "GPT-4.1 nano": 0.72,
    "Doubao-seed-1.6": 0.80,
    "Qwen-Plus": 0.80,
    "Hunyuan T1": 1.00,
    "DeepSeek-V3.1": 1.94,
    "Kimi-K2-Instruct": 4.00,
    "GLM-5.2": 8.00,
}

print("=" * 100)
print("📊 Single-Pass Classification Model Analysis (Single-model One-shot Inference)")
print(f"   Script directory: {SCRIPT_DIR}")
print(f"   Base directory: {BASE_DIR}")
print(f"   Input directory: {INPUT_DIR}")
print(f"   Output directory: {OUTPUT_DIR}")
print("=" * 100)


# ============================================================
# 2. 工具函数
# ============================================================

def calculate_ece_pearson(y_true, y_pred, conf_vals):
    """计算ECE和Pearson相关系数"""
    n_bins = 10
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(conf_vals, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    ece_sum = 0
    for bin_idx in range(n_bins):
        mask = bin_indices == bin_idx
        if mask.sum() == 0:
            continue
        bin_acc = (y_true[mask] == y_pred[mask]).mean()
        bin_conf = conf_vals[mask].mean()
        ece_sum += (mask.sum() / len(conf_vals)) * abs(bin_acc - bin_conf)
    ece = ece_sum

    # Pearson correlation
    if len(conf_vals) > 1 and conf_vals.std() > 0:
        correctness = (y_true == y_pred).astype(float)
        pearson_corr, _ = pearsonr(conf_vals, correctness)
    else:
        pearson_corr = np.nan

    return ece, pearson_corr


# ============================================================
# 3. 加载数据
# ============================================================

data = {}
results = {}
confusion_matrices = {}

print("\n📁 Loading model data...")

# 检查输入目录是否存在
if not INPUT_DIR.exists():
    print(f"❌ 输入目录不存在: {INPUT_DIR}")
    print("请确保目录结构如下：")
    print(f"   {BASE_DIR}/")
    print(f"   ├── analyze/                    # 当前代码所在目录")
    print(f"   │   └── 本脚本.py")
    print(f"   ├── classifyOutcome/            # 数据目录")
    print(f"   │   └── Single-model One-shot Inference/")
    print(f"   │       ├── Single-Pass Classification-gpt.xlsx")
    print(f"   │       ├── Single-Pass Classification-doubao.xlsx")
    print(f"   │       └── ...")
    print(f"   └── analyzeOutcome/             # 输出目录（自动创建）")
    exit()

for model_key in MODEL_NAMES:
    file_path = INPUT_DIR / FILE_TEMPLATE.format(model=model_key)
    display_name = MODEL_DISPLAY_NAMES[model_key]

    if not file_path.exists():
        print(f"   ❌ {display_name}: File not found: {file_path}")
        continue

    try:
        df = pd.read_excel(file_path)

        # 检查必要列
        if HUMAN_LABEL_COL not in df.columns or PREDICTION_COL not in df.columns:
            print(f"   ❌ {display_name}: Missing required columns")
            print(f"      Available columns: {df.columns.tolist()}")
            continue

        # 清理数据
        df_valid = df.dropna(subset=[HUMAN_LABEL_COL, PREDICTION_COL])
        if df_valid.empty:
            print(f"   ❌ {display_name}: No valid data")
            continue

        # 处理标签
        y_true = df_valid[HUMAN_LABEL_COL].astype(str).str.strip().str.lower()
        y_pred = df_valid[PREDICTION_COL].astype(str).str.strip().str.lower()
        current_n = len(df_valid)

        # 计算准确率
        acc = accuracy_score(y_true, y_pred)

        # 混淆矩阵（直接用原始 cm，不缩放）
        labels = sorted(set(y_true) | set(y_pred))
        cm = confusion_matrix(y_true, y_pred, labels=labels)

        confusion_matrices[display_name] = {
            'cm': cm,
            'labels': labels,
            'current_n': current_n
        }

        # 处理置信度
        conf_col = None
        for col in df_valid.columns:
            if CONFIDENCE_COL in col.lower():
                conf_col = col
                break

        if conf_col is not None:
            conf_vals = pd.to_numeric(df_valid[conf_col], errors='coerce')
            if not conf_vals.isna().all() and conf_vals.mean() > 1:
                conf_vals = conf_vals / 100
            valid_idx = ~conf_vals.isna()

            if valid_idx.sum() > 0:
                y_true_valid = y_true[valid_idx]
                y_pred_valid = y_pred[valid_idx]
                conf_valid = conf_vals[valid_idx]

                avg_conf = conf_valid.mean()
                ece, pearson_corr = calculate_ece_pearson(y_true_valid, y_pred_valid, conf_valid)
            else:
                avg_conf = np.nan
                ece = np.nan
                pearson_corr = np.nan
        else:
            avg_conf = np.nan
            ece = np.nan
            pearson_corr = np.nan

        # 成本数据
        cost_per_million = COST_DATA.get(display_name, 10.00)
        input_cost = INPUT_COST_DATA.get(display_name, 1.00)
        cost_effectiveness = (acc / cost_per_million) * 1000 if cost_per_million > 0 else 0

        results[display_name] = {
            'accuracy': acc,
            'accuracy_percent': acc * 100,
            'ece': ece,
            'pearson': pearson_corr,
            'avg_confidence': avg_conf,
            'model_name': display_name,
            'cost_per_million': cost_per_million,
            'input_cost': input_cost,
            'cost_effectiveness': cost_effectiveness,
            'samples': current_n
        }

        print(f"   ✅ {display_name}: Acc={acc * 100:.2f}%, Samples={current_n}, Cost=¥{cost_per_million}/M")

    except Exception as e:
        print(f"   ❌ {display_name}: Error - {e}")
        import traceback
        traceback.print_exc()

print(f"\n📊 Total models loaded: {len(results)}")
print("=" * 100)

if len(results) == 0:
    print("❌ No data loaded. Please check the input files.")
    print(f"   Expected files in: {INPUT_DIR}")
    print("   Expected file pattern: Single-Pass Classification-{model}.xlsx")
    exit()

# ============================================================
# 4. 排序
# ============================================================

sorted_models = sorted(results.keys(), key=lambda k: results[k]['accuracy'], reverse=True)

# ============================================================
# 5. 打印汇总
# ============================================================

print("\n" + "=" * 100)
print("Detailed Metrics Summary (Sorted by Accuracy)")
print("=" * 100)

summary_df = pd.DataFrame({
    'Model': sorted_models,
    'Accuracy (%)': [results[k]['accuracy'] * 100 for k in sorted_models],
    'Pearson': [results[k]['pearson'] for k in sorted_models],
    'ECE': [results[k]['ece'] for k in sorted_models],
    'Avg Confidence': [results[k]['avg_confidence'] for k in sorted_models],
    'Output Cost (¥/M)': [results[k]['cost_per_million'] for k in sorted_models],
    'Cost-Effectiveness': [results[k]['cost_effectiveness'] for k in sorted_models],
    'Samples': [results[k]['samples'] for k in sorted_models]
})

print(summary_df.to_string(index=False))

# ============================================================
# 6. 导出到Excel
# ============================================================

print("\n" + "=" * 100)
print("📊 Exporting data to Excel...")
print("=" * 100)

excel_output = OUTPUT_DIR / "Single_Model_Baseline_Analysis.xlsx"

with pd.ExcelWriter(excel_output, engine='openpyxl') as writer:
    # Sheet 1: Model Performance Summary
    print("  ✓ Writing Sheet 1: Model Performance Summary...")
    summary_export = summary_df.copy()
    summary_export.to_excel(writer, sheet_name='Performance Summary', index=False)

    # Sheet 2: Detailed Metrics with Rankings
    print("  ✓ Writing Sheet 2: Detailed Metrics...")
    detailed_data = []
    for rank, model_name in enumerate(sorted_models, 1):
        res = results[model_name]
        detailed_data.append({
            'Rank': rank,
            'Model': res['model_name'],
            'Accuracy (%)': res['accuracy'] * 100,
            'ECE': res['ece'],
            'Pearson Correlation': res['pearson'],
            'Avg Confidence': res['avg_confidence'],
            'Output Cost (¥/M)': res['cost_per_million'],
            'Input Cost (¥/M)': res['input_cost'],
            'Cost-Effectiveness': res['cost_effectiveness'],
            'Samples': res['samples']
        })
    df_detailed = pd.DataFrame(detailed_data)
    df_detailed.to_excel(writer, sheet_name='Detailed Metrics', index=False)

    # Sheet 3: Per-Model Confusion Matrices（原始 cm）
    print("  ✓ Writing Sheet 3: Confusion Matrices...")
    for model_name, cm_data in confusion_matrices.items():
        cm = cm_data['cm']
        labels = cm_data['labels']
        df_cm = pd.DataFrame(cm, index=labels, columns=labels)
        sheet_name = f'CM_{model_name}'[:31]
        df_cm.to_excel(writer, sheet_name=sheet_name)

    # Sheet 4: Cost-Effectiveness Ranking
    print("  ✓ Writing Sheet 4: Cost-Effectiveness Ranking...")
    ce_ranked = sorted(detailed_data, key=lambda x: x['Cost-Effectiveness'], reverse=True)
    ce_df = pd.DataFrame(ce_ranked)
    ce_df['Rank_CE'] = range(1, len(ce_df) + 1)
    ce_df = ce_df[['Rank_CE', 'Model', 'Accuracy (%)', 'Output Cost (¥/M)', 'Cost-Effectiveness']]
    ce_df.to_excel(writer, sheet_name='Cost-Effectiveness Ranking', index=False)

    # Sheet 5: Calibration Metrics
    print("  ✓ Writing Sheet 5: Calibration Metrics...")
    cal_data = []
    for model_name in sorted_models:
        res = results[model_name]
        gap = res['avg_confidence'] - res['accuracy'] if not np.isnan(res['avg_confidence']) else np.nan
        if not np.isnan(gap):
            if gap > 0.2:
                cal_status = 'Severely Overconfident'
            elif gap > 0.1:
                cal_status = 'Overconfident'
            elif gap > -0.1:
                cal_status = 'Well Calibrated'
            elif gap > -0.2:
                cal_status = 'Underconfident'
            else:
                cal_status = 'Severely Underconfident'
        else:
            cal_status = 'N/A'

        cal_data.append({
            'Model': res['model_name'],
            'Accuracy': res['accuracy'],
            'Accuracy (%)': res['accuracy'] * 100,
            'Avg Confidence': res['avg_confidence'],
            'Confidence-Accuracy Gap': gap,
            'ECE': res['ece'],
            'Pearson Correlation': res['pearson'],
            'Calibration Status': cal_status
        })
    df_cal = pd.DataFrame(cal_data)
    df_cal.to_excel(writer, sheet_name='Calibration Metrics', index=False)

    # Sheet 6: Summary Statistics
    print("  ✓ Writing Sheet 6: Summary Statistics...")
    acc_values = [res['accuracy'] for res in results.values()]
    ece_values = [res['ece'] for res in results.values() if not np.isnan(res['ece'])]
    pearson_values = [res['pearson'] for res in results.values() if not np.isnan(res['pearson'])]
    ce_values = [res['cost_effectiveness'] for res in results.values()]
    sample_values = [res['samples'] for res in results.values()]

    stats_data = [
        {'Metric': 'Number of Models', 'Value': len(results)},
        {'Metric': 'Best Accuracy', 'Value': f"{max(acc_values) * 100:.2f}%"},
        {'Metric': 'Worst Accuracy', 'Value': f"{min(acc_values) * 100:.2f}%"},
        {'Metric': 'Average Accuracy', 'Value': f"{np.mean(acc_values) * 100:.2f}%"},
        {'Metric': 'Std of Accuracy', 'Value': f"{np.std(acc_values) * 100:.2f}%"},
        {'Metric': 'Best ECE (Lowest)', 'Value': f"{min(ece_values):.4f}" if ece_values else 'N/A'},
        {'Metric': 'Worst ECE (Highest)', 'Value': f"{max(ece_values):.4f}" if ece_values else 'N/A'},
        {'Metric': 'Average ECE', 'Value': f"{np.mean(ece_values):.4f}" if ece_values else 'N/A'},
        {'Metric': 'Best Pearson (Highest)', 'Value': f"{max(pearson_values):.4f}" if pearson_values else 'N/A'},
        {'Metric': 'Worst Pearson (Lowest)', 'Value': f"{min(pearson_values):.4f}" if pearson_values else 'N/A'},
        {'Metric': 'Average Pearson', 'Value': f"{np.mean(pearson_values):.4f}" if pearson_values else 'N/A'},
        {'Metric': 'Best Cost-Effectiveness', 'Value': f"{max(ce_values):.2f}"},
        {'Metric': 'Worst Cost-Effectiveness', 'Value': f"{min(ce_values):.2f}"},
        {'Metric': 'Average Cost-Effectiveness', 'Value': f"{np.mean(ce_values):.2f}"},
        {'Metric': 'Min Samples', 'Value': min(sample_values)},
        {'Metric': 'Max Samples', 'Value': max(sample_values)},
        {'Metric': 'Total Samples (sum)', 'Value': sum(sample_values)},
    ]
    df_stats = pd.DataFrame(stats_data)
    df_stats.to_excel(writer, sheet_name='Summary Statistics', index=False)

    # Sheet 7: Per-Class Performance（原始 cm）
    print("  ✓ Writing Sheet 7: Per-Class Performance...")
    class_data = []
    for model_name in sorted_models:
        cm_data = confusion_matrices[model_name]
        cm = cm_data['cm']
        labels = cm_data['labels']

        row = {'Model': model_name}
        for i, label in enumerate(labels):
            correct = cm[i, i]
            total = cm[i, :].sum()
            row[f'{label}_Correct'] = correct
            row[f'{label}_Total'] = total
            row[f'{label}_Accuracy'] = correct / total if total > 0 else 0
        class_data.append(row)
    df_class = pd.DataFrame(class_data)
    df_class.to_excel(writer, sheet_name='Per-Class Performance', index=False)

print(f"\n✅ Excel file saved to: {excel_output}")

# ============================================================
# 7. 绘制合并图表（可靠性指标 + 成本效益）- 上下布局
# ============================================================

print("\n📊 Generating combined chart (Metrics + Cost-Effectiveness) - Vertical layout...")

fig = plt.figure(figsize=(16, 14), dpi=300)

gs = GridSpec(2, 1, height_ratios=[1, 1], hspace=0.35)

# 上子图：可靠性指标
ax_top = fig.add_subplot(gs[0])

metrics = ['accuracy', 'avg_confidence', 'ece', 'pearson']
metric_labels = ['Accuracy', 'Avg Confidence', 'ECE', 'Pearson']
metric_colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(metrics)))

x = np.arange(len(sorted_models))
width = 0.2

for i, (metric, label, color) in enumerate(zip(metrics, metric_labels, metric_colors)):
    values = [results[k][metric] for k in sorted_models]

    if metric == 'ece':
        values = [v if not np.isnan(v) and v <= 1 else np.nan for v in values]

    offset = (i - len(metrics) / 2 + 0.5) * width
    bars = ax_top.bar(x + offset, values, width, label=label,
                      color=color, edgecolor='black', linewidth=0.8)

    for bar, v in zip(bars, values):
        if not np.isnan(v):
            y_pos = bar.get_height() + 0.01
            ax_top.text(bar.get_x() + bar.get_width() / 2, y_pos,
                        f'{v:.3f}', ha='center', va='bottom',
                        fontsize=8, fontweight='bold', rotation=0)

ax_top.set_xlabel('Models', fontsize=14, fontweight='bold')
ax_top.set_ylabel('Score', fontsize=14, fontweight='bold')
ax_top.set_title('(a) Single-Pass Classification Reliability Diagram', fontsize=16, fontweight='bold')

ax_top.set_xticks(x)
ax_top.set_xticklabels(sorted_models, fontsize=11, rotation=45, ha='right')

ax_top.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=11, framealpha=0.9)

ax_top.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8)
ax_top.set_ylim(-0.15, 1.1)
ax_top.axhline(y=0, color='gray', linestyle='-', alpha=0.3, linewidth=0.5)
ax_top.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5, linewidth=1)

for idx, model_name in enumerate(sorted_models):
    acc_val = results[model_name]['accuracy']
    if not np.isnan(acc_val):
        ax_top.text(idx, 1.05, f"#{idx + 1}", ha='center', va='bottom',
                    fontsize=10, fontweight='bold', color='darkred')

# 下子图：成本效益气泡图
ax_bottom = fig.add_subplot(gs[1])

model_names_display = []
costs = []
accuracies = []
cost_effectiveness = []
bubble_sizes = []

for model_name in sorted_models:
    res = results[model_name]
    model_names_display.append(res['model_name'])
    costs.append(res['cost_per_million'])
    accuracies.append(res['accuracy'] * 100)
    cost_eff = res['cost_effectiveness']
    cost_effectiveness.append(cost_eff)
    bubble_sizes.append(cost_eff * 50)

colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(model_names_display)))

scatter = ax_bottom.scatter(costs, accuracies,
                            s=bubble_sizes,
                            c=colors,
                            alpha=0.7,
                            edgecolors='black',
                            linewidth=1.5)

for i, name in enumerate(model_names_display):
    if name.upper() == 'GPT-4.1 NANO':
        offset_x = -10
        offset_y = -10
        ha = 'right'
        va = 'top'
    elif name.upper() == 'HUNYUAN T1':
        offset_x = 10
        offset_y = -10
        ha = 'left'
        va = 'top'
    elif name.upper() == 'GLM-5.2':
        offset_x = -8
        offset_y = 0
        ha = 'right'
        va = 'center'
    else:
        if costs[i] > 15:
            offset_x = -10
            ha = 'right'
        elif costs[i] < 3:
            offset_x = 10
            ha = 'left'
        else:
            offset_x = 5
            ha = 'left'

        if accuracies[i] > 70:
            offset_y = 5
            va = 'bottom'
        else:
            offset_y = -5
            va = 'top'

    bbox_props = dict(boxstyle='round,pad=0.3',
                      facecolor='white',
                      alpha=0.85,
                      edgecolor='gray',
                      linewidth=0.5)

    ax_bottom.annotate(f'{name}\n{accuracies[i]:.1f}%\n¥{costs[i]:.1f}/M',
                       (costs[i], accuracies[i]),
                       xytext=(offset_x, offset_y),
                       textcoords='offset points',
                       fontsize=10,
                       fontweight='normal',
                       ha=ha,
                       va=va,
                       bbox=bbox_props)

ax_bottom.axhline(y=50, color='gray', linestyle='--', alpha=0.4, linewidth=1)
ax_bottom.axvline(x=10, color='gray', linestyle='--', alpha=0.4, linewidth=1)

cbar = plt.colorbar(scatter, ax=ax_bottom)
cbar.set_label('Index (sorted by accuracy)', fontsize=12)

ax_bottom.text(0.98, 0.02,
               'Bubble size = Cost-Effectiveness\n(Accuracy / Output Cost × 1000)',
               transform=ax_bottom.transAxes,
               fontsize=11,
               ha='right',
               va='bottom',
               bbox=dict(boxstyle='round,pad=0.3',
                         facecolor='white',
                         alpha=0.8,
                         edgecolor='gray'))

ax_bottom.set_xlabel('Output Cost (¥ per 1M tokens)', fontsize=14, fontweight='bold')
ax_bottom.set_ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
ax_bottom.set_title('(b) Single-Pass Classification Cost Effectiveness Diagram', fontsize=16, fontweight='bold')

x_max = max(costs) * 1.25 if costs else 35
ax_bottom.set_xlim(-1, x_max)
ax_bottom.set_ylim(0, 105)
ax_bottom.grid(True, alpha=0.3, linestyle='--')
ax_bottom.tick_params(labelsize=11)

plt.tight_layout()
plt.savefig(CHART_DIR / 'Single_Model_Baseline_Combined_Metrics_Cost.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"✅ Saved: {CHART_DIR / 'Single_Model_Baseline_Combined_Metrics_Cost.png'}")

# ============================================================
# 9. 绘制混淆矩阵汇总（原始 cm）
# ============================================================

print("\n📊 Generating Confusion Matrices...")

n_models = len(confusion_matrices)
n_cols = min(2, n_models)
n_rows = (n_models + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5.5 * n_rows), dpi=300)
if n_models == 1:
    axes = np.array([axes])
else:
    axes = axes.flatten()

for idx, model_name in enumerate(sorted_models):
    ax = axes[idx]
    cm_data = confusion_matrices[model_name]
    cm = cm_data['cm']
    labels = cm_data['labels']

    sns.heatmap(cm,
                annot=True,
                fmt='d',
                cmap='viridis',
                xticklabels=labels,
                yticklabels=labels,
                ax=ax,
                cbar_kws={'label': 'Count'},
                annot_kws={'fontsize': 11, 'fontweight': 'bold'})

    acc = results[model_name]['accuracy'] * 100
    ax.set_title(f'{model_name}\nAccuracy: {acc:.1f}%, N={cm_data["current_n"]}',
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.tick_params(labelsize=10)

for idx in range(len(confusion_matrices), len(axes)):
    axes[idx].axis('off')

plt.tight_layout()
plt.savefig(CHART_DIR / 'Single_Model_Baseline_Confusion_Matrices.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"✅ Saved: {CHART_DIR / 'Single_Model_Baseline_Confusion_Matrices.png'}")

# ============================================================
# 10. 绘制单个混淆矩阵（原始 cm）
# ============================================================

print("\n📊 Generating individual confusion matrices...")

for model_name in sorted_models:
    cm_data = confusion_matrices[model_name]
    cm = cm_data['cm']
    labels = cm_data['labels']
    acc = results[model_name]['accuracy'] * 100
    n = cm_data['current_n']

    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)

    sns.heatmap(cm,
                annot=True,
                fmt='d',
                cmap='viridis',
                xticklabels=labels,
                yticklabels=labels,
                ax=ax,
                cbar_kws={'label': 'Count'},
                annot_kws={'fontsize': 13, 'fontweight': 'bold'})

    ax.set_title(f'{model_name} - Confusion Matrix\nAccuracy: {acc:.1f}%, N={n}',
                 fontsize=15, fontweight='bold')
    ax.set_xlabel('Predicted Label', fontsize=13)
    ax.set_ylabel('True Label', fontsize=13)
    ax.tick_params(labelsize=11)

    plt.tight_layout()
    plt.savefig(CHART_DIR / f'CM_{model_name.replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  ✅ Saved: CM_{model_name.replace(' ', '_')}.png")

# ============================================================
# 11. 成本效益排名
# ============================================================

print("\n" + "=" * 100)
print("Cost-Effectiveness Ranking (Accuracy / Output Cost × 1000)")
print("=" * 100)

ce_sorted = sorted(results.items(), key=lambda x: x[1]['cost_effectiveness'], reverse=True)

print(f"\n{'Rank':<6} {'Model':<20} {'Accuracy':<12} {'Output Cost (¥/M)':<18} {'CE Score':<15} {'Interpretation'}")
print("-" * 90)

for rank, (model_name, res) in enumerate(ce_sorted, 1):
    ce = res['cost_effectiveness']
    if ce > 200:
        interp = "Excellent value"
    elif ce > 100:
        interp = "Good value"
    elif ce > 50:
        interp = "Moderate value"
    else:
        interp = "Premium (low CE)"

    print(
        f"{rank:<6} {res['model_name']:<20} {res['accuracy'] * 100:>6.1f}%     ¥{res['cost_per_million']:>6.2f}        {ce:>10.2f}    {interp}")

# ============================================================
# 12. 关键发现
# ============================================================

print("\n" + "=" * 100)
print("Key Findings")
print("=" * 100)

best_acc = max(results.items(), key=lambda x: x[1]['accuracy'])
print(f"🏆 Best Accuracy: {best_acc[1]['model_name']} ({best_acc[1]['accuracy'] * 100:.1f}%)")

best_ce = max(results.items(), key=lambda x: x[1]['cost_effectiveness'])
print(f"💰 Best Cost-Effectiveness: {best_ce[1]['model_name']} (CE = {best_ce[1]['cost_effectiveness']:.2f})")

valid_ece = [(k, v) for k, v in results.items() if not np.isnan(v['ece'])]
if valid_ece:
    best_ece = min(valid_ece, key=lambda x: x[1]['ece'])
    print(f"🎯 Best Calibration (Lowest ECE): {best_ece[1]['model_name']} (ECE = {best_ece[1]['ece']:.3f})")

valid_pearson = [(k, v) for k, v in results.items() if not np.isnan(v['pearson'])]
if valid_pearson:
    best_pearson = max(valid_pearson, key=lambda x: x[1]['pearson'])
    print(
        f"📊 Best Confidence-Accuracy Correlation: {best_pearson[1]['model_name']} (r = {best_pearson[1]['pearson']:.3f})")

    negative_pearson = [(k, v) for k, v in valid_pearson if v['pearson'] < 0]
    if negative_pearson:
        print(f"\n⚠️ Models with NEGATIVE Pearson correlation:")
        for k, v in negative_pearson:
            print(f"   - {v['model_name']}: r = {v['pearson']:.3f}")

# ============================================================
# 13. 输出文件汇总
# ============================================================

print("\n" + "=" * 100)
print("📁 Output Files Summary")
print("=" * 100)
print(f"📊 Excel Files:")
print(f"   - {excel_output} (contains 7 sheets of analysis data)")
print(f"\n📈 Chart Files (300 DPI):")
print(f"   - {CHART_DIR / 'Single_Model_Baseline_Combined_Metrics_Cost.png'} (Combined chart with (a) top and (b) bottom)")
print(f"   - {CHART_DIR / 'Single_Model_Baseline_Confusion_Matrices.png'}")
print(f"   - Individual confusion matrices: CM_*.png")
print("=" * 100)
print("✅ Analysis completed successfully!")