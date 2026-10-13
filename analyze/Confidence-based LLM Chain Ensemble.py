"""
Confidence-based LLM Chain Ensemble Analysis
Reads the 3 pre-computed files in classifyOutcome/Confidence Cascade Routing/:
  - Chain_Ensemble_Simplified.xlsx       (final predictions per sample)
  - Chain_Ensemble_Link_Statistics.xlsx  (per-link statistics)
  - Chain_Ensemble_Metrics.xlsx          (overall metrics)

Outputs to analyzeOutcome/Confidence Cascade Routing/
  - Chain_Ensemble_Analysis_Results.xlsx
  - Chain_Ensemble_Stage_Analysis.png
  - Chain_Ensemble_Report.txt

Chart: 每个 link 两个柱子
  - 左柱：处理总数（堆叠：处理正确 / 处理错误）
  - 右柱：被选择数（堆叠：被选正确 / 被选错误）
  - 每段柱内写样本数，柱顶写正确率
  - 右轴两条折线：正确率 + 置信度
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import to_hex
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# ============================================================================
# Style
# ============================================================================
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 600
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['legend.fontsize'] = 10

VIRIDIS = mpl.colormaps['viridis']

# 颜色
COLOR_PROC_CORRECT = to_hex(VIRIDIS(0.15))   # 处理-正确
COLOR_PROC_WRONG   = to_hex(VIRIDIS(0.55))   # 处理-错误
COLOR_SEL_CORRECT  = to_hex(VIRIDIS(0.30))   # 被选-正确
COLOR_SEL_WRONG    = to_hex(VIRIDIS(0.75))   # 被选-错误
COLOR_ACC_LINE     = '#1f77b4'               # 正确率折线
COLOR_CONF_LINE    = '#d62728'               # 置信度折线

# Model order
MODEL_ORDER = ["gpt", "doubao", "qwen", "hunyuan", "deepseek", "kimi", "glm"]
MODEL_DISPLAY = {
    "gpt":      "GPT-4.1 nano",
    "doubao":   "Doubao-seed-1.6",
    "qwen":     "Qwen-Plus",
    "hunyuan":  "Hunyuan T1",
    "deepseek": "DeepSeek-V3.1",
    "kimi":     "Kimi-K2-Instruct",
    "glm":      "GLM-5.2",
}


# ============================================================================
# Paths
# ============================================================================
def get_paths():
    try:
        SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        SCRIPT_DIR = os.getcwd()

    PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
    INPUT_DIR   = os.path.join(PROJECT_DIR, "classifyOutcome",
                               "Confidence Cascade Routing")
    OUTPUT_DIR  = os.path.join(PROJECT_DIR, "analyzeOutcome",
                               "Confidence Cascade Routing")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    return {
        "project":     PROJECT_DIR,
        "input":       INPUT_DIR,
        "output":      OUTPUT_DIR,
        "simplified":  os.path.join(INPUT_DIR, "Chain_Ensemble_Simplified.xlsx"),
        "link_stats":  os.path.join(INPUT_DIR, "Chain_Ensemble_Link_Statistics.xlsx"),
        "metrics":     os.path.join(INPUT_DIR, "Chain_Ensemble_Metrics.xlsx"),
    }


# ============================================================================
# Helpers
# ============================================================================
def norm_model(x):
    x = str(x).strip().lower()
    for k in MODEL_ORDER:
        if k in x:
            return k
    return x


def pick_col(df, cands):
    for c in cands:
        if c in df.columns:
            return c
    return None


def find_pred_col(df):
    return pick_col(df, ["Final_Predicted_Label", "Predicted_Label", "Final_Prediction"])


def find_conf_col(df):
    return pick_col(df, ["Final_Confidence", "Confidence"])


def find_model_col(df):
    return pick_col(df, ["Model", "model", "Model_Name", "Final_Model"])


# ============================================================================
# Loaders
# ============================================================================
def load_simplified(path):
    print(f"\n[LOAD] Simplified: {path}")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Not found: {path}")
    df = pd.read_excel(path)
    df.columns = [str(c).strip() for c in df.columns]
    print(f"       shape={df.shape}")
    print(f"       columns={list(df.columns)}")
    return df


def load_link_stats(path):
    print(f"\n[LOAD] Link Statistics: {path}")
    if not os.path.exists(path):
        print(f"       ⚠️ Not found, will try to derive from Simplified")
        return None
    df = pd.read_excel(path)
    df.columns = [str(c).strip() for c in df.columns]
    print(f"       shape={df.shape}")
    print(f"       columns={list(df.columns)}")
    print(df.to_string(index=False))
    return df


def load_metrics(path):
    print(f"\n[LOAD] Metrics: {path}")
    if not os.path.exists(path):
        print(f"       ⚠️ Not found, will compute from Simplified")
        return None
    df = pd.read_excel(path)
    df.columns = [str(c).strip() for c in df.columns]
    print(f"       shape={df.shape}")
    print(f"       columns={list(df.columns)}")
    print(df.to_string(index=False))
    return df


# ============================================================================
# Build per-link stats
# ============================================================================
def build_link_table(simplified, link_stats):
    """
    生成每个 link 需要的字段：
      Processed_Total / Processed_Correct / Processed_Wrong / Processed_Accuracy
      Selected_Total  / Selected_Correct  / Selected_Wrong  / Selected_Accuracy
      Mean_Confidence
    优先使用 link_stats 里已有的字段，否则从 simplified 计算。
    """
    df = simplified.copy()
    df["_model_"] = df["Final_Model"].apply(norm_model) if "Final_Model" in df.columns else None

    pred_col = find_pred_col(df)
    conf_col = find_conf_col(df)
    gt_col = "Human_Label"

    # 计算每条样本的 correct（被选中情况下）
    if pred_col and gt_col in df.columns:
        y_pred = df[pred_col].astype(str).str.strip().str.lower()
        y_true = df[gt_col].astype(str).str.strip().str.lower()
        df["_correct_"] = (y_pred == y_true).astype(int)
    else:
        df["_correct_"] = np.nan

    # 按模型聚合 = "被选择" 的统计
    agg = df.groupby("_model_").agg(
        Selected_Total=("_correct_", "count"),
        Selected_Correct=("_correct_", "sum"),
    ).reset_index()
    agg["Selected_Wrong"] = agg["Selected_Total"] - agg["Selected_Correct"]
    agg["Selected_Accuracy"] = np.where(
        agg["Selected_Total"] > 0,
        agg["Selected_Correct"] / agg["Selected_Total"],
        np.nan
    )

    # 置信度
    if conf_col:
        conf_agg = df.groupby("_model_")[conf_col].mean().reset_index()
        conf_agg.columns = ["_model_", "Mean_Confidence"]
        agg = agg.merge(conf_agg, on="_model_", how="left")
    else:
        agg["Mean_Confidence"] = np.nan

    # 合并 link_stats 里可能的"处理总数"信息
    proc_map = {}
    if link_stats is not None and len(link_stats) > 0:
        ls = link_stats.copy()
        model_col = find_model_col(ls)
        if model_col is not None:
            ls["_model_"] = ls[model_col].apply(norm_model)
            for _, row in ls.iterrows():
                m = row["_model_"]
                # 尽量找处理相关的列
                p_total = None
                p_correct = None
                p_wrong = None
                for c in ["Processed_Total", "Samples_Processed", "Processed",
                          "Total_Processed", "Num_Processed"]:
                    if c in ls.columns and not pd.isna(row[c]):
                        p_total = float(row[c]); break
                for c in ["Processed_Correct", "Correct_Processed",
                          "Num_Correct_Processed"]:
                    if c in ls.columns and not pd.isna(row[c]):
                        p_correct = float(row[c]); break
                for c in ["Processed_Wrong", "Wrong_Processed",
                          "Num_Wrong_Processed"]:
                    if c in ls.columns and not pd.isna(row[c]):
                        p_wrong = float(row[c]); break
                if p_total is not None:
                    proc_map[m] = (p_total, p_correct, p_wrong)

    # 装配最终表
    rows = []
    for i, m in enumerate(MODEL_ORDER):
        sub = agg[agg["_model_"] == m]
        if len(sub) == 0:
            continue
        sub = sub.iloc[0]

        sel_total   = int(sub["Selected_Total"])
        sel_correct = int(sub["Selected_Correct"])
        sel_wrong   = int(sub["Selected_Wrong"])
        sel_acc     = float(sub["Selected_Accuracy"]) if not pd.isna(sub["Selected_Accuracy"]) else np.nan
        mean_conf   = float(sub["Mean_Confidence"]) if not pd.isna(sub["Mean_Confidence"]) else np.nan

        # 处理总数：优先 link_stats
        if m in proc_map:
            p_total, p_correct, p_wrong = proc_map[m]
            if p_correct is None or p_wrong is None:
                # 只有总数时，按被选比例拆分（近似）
                p_correct = sel_correct / sel_total * p_total if sel_total > 0 else 0
                p_wrong   = p_total - p_correct
        else:
            # 没有处理总数信息时，默认 = 被选择数
            p_total   = sel_total
            p_correct = sel_correct
            p_wrong   = sel_wrong

        p_total   = int(round(p_total))
        p_correct = int(round(p_correct))
        p_wrong   = int(round(p_wrong))
        if p_correct + p_wrong != p_total:
            p_wrong = p_total - p_correct

        proc_acc = p_correct / p_total if p_total > 0 else np.nan

        rows.append({
            "Link": i + 1,
            "Model": m,
            "Model_Display": MODEL_DISPLAY.get(m, m),
            "Processed_Total": p_total,
            "Processed_Correct": p_correct,
            "Processed_Wrong": p_wrong,
            "Processed_Accuracy": proc_acc,
            "Selected_Total": sel_total,
            "Selected_Correct": sel_correct,
            "Selected_Wrong": sel_wrong,
            "Selected_Accuracy": sel_acc,
            "Mean_Confidence": mean_conf,
            "Net_Contribution": sel_correct - sel_wrong,
        })

    return pd.DataFrame(rows)


# ============================================================================
# Metrics
# ============================================================================
def compute_ece(confidences, correct, n_bins=10):
    confidences = np.asarray(confidences, dtype=float)
    correct     = np.asarray(correct, dtype=float)
    bins = np.linspace(0, 1, n_bins + 1)
    bin_idx = np.digitize(confidences, bins, right=False)
    ece = 0.0
    n = len(confidences)
    for i in range(1, n_bins + 1):
        mask = bin_idx == i
        if mask.sum() == 0:
            continue
        ece += (mask.sum() / n) * abs(correct[mask].mean()
                                     - confidences[mask].mean())
    return ece


def compute_metrics(simplified):
    print("\n[COMPUTE] Overall metrics from Simplified ...")
    pred_col = find_pred_col(simplified)
    conf_col = find_conf_col(simplified)
    gt_col = "Human_Label"
    if pred_col is None or conf_col is None or gt_col not in simplified.columns:
        raise KeyError("Simplified 缺少必要列")

    y_pred = simplified[pred_col].astype(str).str.strip().str.lower()
    y_true = simplified[gt_col].astype(str).str.strip().str.lower()
    conf   = pd.to_numeric(simplified[conf_col], errors="coerce").fillna(0.0).values
    correct = (y_pred == y_true).astype(int).values

    acc = float(correct.mean())
    avg_conf = float(np.mean(conf))
    ece = compute_ece(conf, correct)
    pearson = (float(np.corrcoef(conf, correct)[0, 1])
               if len(np.unique(correct)) > 1 and len(np.unique(conf)) > 1 else 0.0)

    return {
        "N": len(simplified),
        "Accuracy": acc,
        "Average_Confidence": avg_conf,
        "ECE": ece,
        "Pearson": pearson,
    }


# ============================================================================
# Chart
# ============================================================================
def create_stage_chart(link_table, metrics, output_path):
    """
    每个 link 两个柱子：
      - 左柱：Routed Total      —— 路由到该节点的样本总量（正确/错误堆叠）
      - 右柱：Selected (Final)  —— 该节点结果被最终采纳的样本（正确/错误堆叠）
    柱内写样本数，柱顶写正确率 + 总数；
    右轴：仅保留 Mean Confidence（红色虚线）折线。
    """
    print(f"\n[CHART] Stage-level analysis (double bars per link) ...")
    if link_table is None or len(link_table) == 0:
        print("       ⚠️ Empty link table, skip chart")
        return None

    df = link_table.copy().reset_index(drop=True)
    models_disp = df["Model_Display"].tolist()
    n = len(df)

    x = np.arange(n)
    bar_w = 0.36
    gap = 0.02

    proc_correct = df["Processed_Correct"].values.astype(float)
    proc_wrong   = df["Processed_Wrong"].values.astype(float)
    proc_total   = df["Processed_Total"].values.astype(float)
    proc_acc     = df["Processed_Accuracy"].values.astype(float) * 100

    sel_correct  = df["Selected_Correct"].values.astype(float)
    sel_wrong    = df["Selected_Wrong"].values.astype(float)
    sel_total    = df["Selected_Total"].values.astype(float)
    sel_acc      = df["Selected_Accuracy"].values.astype(float) * 100
    mean_conf    = df["Mean_Confidence"].values.astype(float) * 100

    fig, ax1 = plt.subplots(figsize=(15, 8))

    x_proc = x - bar_w/2 - gap/2
    x_sel  = x + bar_w/2 + gap/2

    # ---- 左柱：Routed Total ----
    b_pc = ax1.bar(x_proc, proc_correct, width=bar_w,
                   color=COLOR_PROC_CORRECT, edgecolor='black', linewidth=0.8,
                   label='Routed Total - Correct', zorder=2)
    b_pw = ax1.bar(x_proc, proc_wrong, width=bar_w, bottom=proc_correct,
                   color=COLOR_PROC_WRONG, edgecolor='black', linewidth=0.8,
                   label='Routed Total - Wrong', zorder=2)

    # ---- 右柱：Selected (Final) ----
    b_sc = ax1.bar(x_sel, sel_correct, width=bar_w,
                   color=COLOR_SEL_CORRECT, edgecolor='black', linewidth=0.8,
                   label='Selected (Final) - Correct', zorder=2)
    b_sw = ax1.bar(x_sel, sel_wrong, width=bar_w, bottom=sel_correct,
                   color=COLOR_SEL_WRONG, edgecolor='black', linewidth=0.8,
                   label='Selected (Final) - Wrong', zorder=2)

    # ---- 柱内标注样本数 ----
    def annotate_bar(ax, bars, bottoms, values, color='white', fs=8):
        for b, bot, v in zip(bars, bottoms, values):
            if v <= 0:
                continue
            ax.text(b.get_x() + b.get_width()/2,
                    bot + v/2,
                    f'{int(v):,}',
                    ha='center', va='center',
                    fontsize=fs, fontweight='bold', color=color)

    annotate_bar(ax1, b_pc, np.zeros_like(proc_correct), proc_correct, 'white')
    annotate_bar(ax1, b_pw, proc_correct, proc_wrong, 'white')
    annotate_bar(ax1, b_sc, np.zeros_like(sel_correct), sel_correct, 'white')
    annotate_bar(ax1, b_sw, sel_correct, sel_wrong, 'black')

    # ---- 柱顶标注：柱名 + 正确率 + 总数 ----
    ymax = max(np.nanmax(proc_total), np.nanmax(sel_total), 1) * 1.38
    ax1.set_ylim(0, ymax)

    for i in range(n):
        # 左柱（路由总量）
        top_p = proc_total[i]
        ax1.text(x_proc[i], top_p + ymax * 0.10,
                 'Routed\nTotal',
                 ha='center', va='bottom', fontsize=8,
                 color=COLOR_PROC_CORRECT, fontweight='bold')
        ax1.text(x_proc[i], top_p + ymax * 0.015,
                 f'Acc: {proc_acc[i]:.1f}%' if not np.isnan(proc_acc[i]) else 'Acc: -',
                 ha='center', va='bottom', fontsize=8,
                 color='black', fontweight='bold')
        ax1.text(x_proc[i], top_p + ymax * 0.055,
                 f'Total: {int(proc_total[i]):,}',
                 ha='center', va='bottom', fontsize=8, color='#333333')

        # 右柱（最终筛选）
        top_s = sel_total[i]
        ax1.text(x_sel[i], top_s + ymax * 0.10,
                 'Selected\n(Final)',
                 ha='center', va='bottom', fontsize=8,
                 color=COLOR_SEL_CORRECT, fontweight='bold')
        ax1.text(x_sel[i], top_s + ymax * 0.015,
                 f'Acc: {sel_acc[i]:.1f}%' if not np.isnan(sel_acc[i]) else 'Acc: -',
                 ha='center', va='bottom', fontsize=8,
                 color='black', fontweight='bold')
        ax1.text(x_sel[i], top_s + ymax * 0.055,
                 f'Total: {int(sel_total[i]):,}',
                 ha='center', va='bottom', fontsize=8, color='#333333')

    # ---- 轴设置 ----
    ax1.set_xlabel('Link (Model)', fontsize=13)
    ax1.set_ylabel('Number of Samples', fontsize=13)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'Link {i+1}\n{m}' for i, m in enumerate(models_disp)],
                        fontsize=9)
    ax1.grid(axis='y', alpha=0.3, zorder=0)

    # ---- 右轴：仅保留置信度折线 ----
    ax2 = ax1.twinx()
    if not np.all(np.isnan(mean_conf)):
        ax2.plot(x, mean_conf, 's--', color=COLOR_CONF_LINE, linewidth=2,
                 markersize=7, markerfacecolor='white', markeredgewidth=2,
                 label='Mean Confidence (%)', zorder=4)
        for i, v in enumerate(mean_conf):
            if not np.isnan(v):
                ax2.text(x[i], v - 2.5, f'{v:.1f}%',
                         ha='center', va='top', fontsize=8,
                         color=COLOR_CONF_LINE, fontweight='bold')

    ax2.set_ylabel('Mean Confidence (%)', fontsize=13, color=COLOR_CONF_LINE)
    ax2.set_ylim(0, 115)
    ax2.tick_params(axis='y', labelcolor=COLOR_CONF_LINE)

    # ---- 图例 ----
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2,
               loc='upper right', fontsize=8, framealpha=0.9, ncol=2)

    # ---- 整体指标 ----
    info = (f"Overall Accuracy: {metrics['Accuracy']*100:.1f}%\n"
            f"Pearson: {metrics['Pearson']*100:.2f}%\n"
            f"ECE: {metrics['ECE']*100:.2f}%\n"
            f"Total Samples: {metrics['N']:,}")
    ax1.text(0.985, 0.72, info, transform=ax1.transAxes,
             fontsize=9, ha='right', va='top',
             bbox=dict(boxstyle='round,pad=0.4',
                       facecolor='white', edgecolor='black', alpha=0.9))

    ax1.set_title('Confidence-based Cascade Routing: Stage-level Analysis\n'
                  '(Routed Total vs Selected Final per Link, with Confidence)',
                  fontsize=13, fontweight='bold', pad=12)

    plt.tight_layout()
    plt.savefig(output_path, dpi=600, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"       ✅ Saved: {output_path}")
    return output_path

# ============================================================================
# Excel export
# ============================================================================
def export_excel(simplified, link_table, metrics_df, metrics_dict, output_path):
    print(f"\n[EXCEL] Exporting to {output_path} ...")
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        overall = pd.DataFrame([
            {"Metric": "N",                  "Value": metrics_dict["N"]},
            {"Metric": "Accuracy",           "Value": f"{metrics_dict['Accuracy']*100:.2f}%"},
            {"Metric": "Average_Confidence", "Value": f"{metrics_dict['Average_Confidence']*100:.2f}%"},
            {"Metric": "ECE",                "Value": f"{metrics_dict['ECE']*100:.2f}%"},
            {"Metric": "Pearson",            "Value": f"{metrics_dict['Pearson']*100:.2f}%"},
        ])
        overall.to_excel(writer, sheet_name="Overall Metrics", index=False)

        if link_table is not None and len(link_table) > 0:
            link_table.to_excel(writer, sheet_name="Link Statistics", index=False)

        simplified.to_excel(writer, sheet_name="Full Predictions", index=False)

        if metrics_df is not None and len(metrics_df) > 0:
            metrics_df.to_excel(writer, sheet_name="Raw Metrics", index=False)

    print(f"       ✅ Saved: {output_path}")


# ============================================================================
# Report
# ============================================================================
def write_report(link_table, metrics_dict, output_path):
    print(f"\n[REPORT] Writing {output_path} ...")
    lines = []
    lines.append("=" * 70)
    lines.append("CONFIDENCE-BASED LLM CHAIN ENSEMBLE - ANALYSIS REPORT")
    lines.append("=" * 70)
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Total Samples: {metrics_dict['N']:,}")
    lines.append("")
    lines.append("-" * 50)
    lines.append("OVERALL METRICS")
    lines.append("-" * 50)
    lines.append(f"Accuracy:            {metrics_dict['Accuracy']*100:.2f}%")
    lines.append(f"Average Confidence:  {metrics_dict['Average_Confidence']*100:.2f}%")
    lines.append(f"ECE:                 {metrics_dict['ECE']*100:.2f}%")
    lines.append(f"Pearson Correlation: {metrics_dict['Pearson']*100:.2f}%")
    lines.append("")

    if link_table is not None and len(link_table) > 0:
        lines.append("-" * 50)
        lines.append("STAGE-BY-STAGE STATISTICS")
        lines.append("-" * 50)
        lines.append("Processed_* = 该节点处理总数（其正确/错误）；"
                     "Selected_* = 该节点结果被最终采纳的数量（其正确/错误）")
        lines.append("Net_Contribution = Selected_Correct - Selected_Wrong")
        lines.append("")
        lines.append(link_table.to_string(index=False))
        lines.append("")

    lines.append("=" * 70)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"       ✅ Saved: {output_path}")


# ============================================================================
# Main
# ============================================================================
def main():
    print("=" * 70)
    print("Confidence-based LLM Chain Ensemble Analysis")
    print("=" * 70)

    paths = get_paths()
    print(f"\nInput dir:  {paths['input']}")
    print(f"Output dir: {paths['output']}")

    simplified = load_simplified(paths["simplified"])
    link_stats = load_link_stats(paths["link_stats"])
    metrics_df = load_metrics(paths["metrics"])

    link_table = build_link_table(simplified, link_stats)
    print("\n[LINK TABLE]")
    print(link_table.to_string(index=False))

    metrics_dict = compute_metrics(simplified)
    print("\n[METRICS]")
    for k, v in metrics_dict.items():
        print(f"  {k}: {v:,}" if k == "N" else f"  {k}: {v:.4f}")

    chart_path  = os.path.join(paths["output"], "Chain_Ensemble_Stage_Analysis.png")
    excel_path  = os.path.join(paths["output"], "Chain_Ensemble_Analysis_Results.xlsx")
    report_path = os.path.join(paths["output"], "Chain_Ensemble_Report.txt")

    create_stage_chart(link_table, metrics_dict, chart_path)
    export_excel(simplified, link_table, metrics_df, metrics_dict, excel_path)
    write_report(link_table, metrics_dict, report_path)

    print("\n" + "=" * 70)
    print("✅ DONE")
    print("=" * 70)
    print(f"Outputs in: {paths['output']}")
    for p in [chart_path, excel_path, report_path]:
        print(f"  - {os.path.basename(p)}")


if __name__ == "__main__":
    main()