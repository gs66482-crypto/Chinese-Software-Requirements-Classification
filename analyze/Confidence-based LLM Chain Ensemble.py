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
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import to_hex, to_rgb
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# ============================================================================
# Style
# ============================================================================
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 13
plt.rcParams['axes.labelsize'] = 15
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['legend.fontsize'] = 12

# Viridis palette; correct segments are stacked on top.
VIRIDIS = mpl.colormaps['viridis']
COLOR_PROC_CORRECT = to_hex(VIRIDIS(0.35))
COLOR_PROC_WRONG = to_hex(VIRIDIS(0.90))
COLOR_SEL_CORRECT = to_hex(VIRIDIS(0.55))
COLOR_SEL_WRONG = to_hex(VIRIDIS(0.75))
COLOR_CONF_LINE = to_hex(VIRIDIS(0.05))

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
    model_col = pick_col(df, ["Final_Model", "Model", "model", "Model_Name"])
    if model_col is None:
        raise KeyError("Missing final selection model column")
    if df[model_col].isna().any():
        raise ValueError("Missing final selection model")
    df["_model_"] = df[model_col].apply(norm_model)
    unknown = set(df["_model_"]) - set(MODEL_ORDER)
    if unknown:
        raise ValueError(f"Unknown final models: {sorted(unknown)}")

    pred_col = find_pred_col(df)
    conf_col = find_conf_col(df)
    gt_col = "Human_Label"

    # 计算每条样本的 correct（被选中情况下）
    if pred_col and gt_col in df.columns:
        if df[[pred_col, gt_col]].isna().any().any():
            raise ValueError("Missing final prediction or reference label")
        y_pred = df[pred_col].astype(str).str.strip().str.lower()
        y_true = df[gt_col].astype(str).str.strip().str.lower()
        df["_correct_"] = (y_pred == y_true).astype(int)
    else:
        raise KeyError("Selected accuracy requires final predictions and Human_Label")

    # 按模型聚合 = "被选择" 的统计
    agg = df.groupby("_model_").agg(
        Selected_Total=("_correct_", "size"),
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
        df[conf_col] = pd.to_numeric(df[conf_col], errors="coerce")
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
            if m not in proc_map:
                continue
            sub = pd.Series({"Selected_Total": 0, "Selected_Correct": 0,
                             "Selected_Wrong": 0, "Selected_Accuracy": np.nan,
                             "Mean_Confidence": np.nan})
        else:
            sub = sub.iloc[0]

        sel_total   = int(sub["Selected_Total"])
        sel_correct = int(sub["Selected_Correct"])
        sel_wrong   = int(sub["Selected_Wrong"])
        sel_acc     = float(sub["Selected_Accuracy"]) if not pd.isna(sub["Selected_Accuracy"]) else np.nan
        mean_conf   = float(sub["Mean_Confidence"]) if not pd.isna(sub["Mean_Confidence"]) else np.nan

        # Processed correctness requires actual stage predictions/counts.
        # Never estimate it from the selected subset or final predictions.
        p_total, p_correct, p_wrong = proc_map.get(m, (np.nan, None, None))
        if p_correct is None and p_wrong is not None and np.isfinite(p_total):
            p_correct = p_total - p_wrong
        if p_wrong is None and p_correct is not None and np.isfinite(p_total):
            p_wrong = p_total - p_correct
        if p_correct is None or p_wrong is None:
            p_correct = p_wrong = np.nan
        if np.isfinite(p_total):
            if p_total < sel_total:
                raise ValueError(f"{m}: processed total is smaller than selected total")
            if np.isfinite(p_correct) and (p_correct < 0 or p_wrong < 0 or
                    not np.isclose(p_correct + p_wrong, p_total)):
                raise ValueError(f"{m}: inconsistent processed counts")
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
    """Each bar uses its own denominator; correct samples are stacked on top."""
    if link_table is None or link_table.empty:
        return None
    from matplotlib.patches import Patch
    df = link_table.reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(16, 8))
    width = 0.34
    x = np.arange(len(df))
    for prefix, offset, correct_colour, wrong_colour, name in [
        ("Processed", -0.19, COLOR_PROC_CORRECT, COLOR_PROC_WRONG, "Routed"),
        ("Selected", 0.19, COLOR_SEL_CORRECT, COLOR_SEL_WRONG, "Selected")
    ]:
        for i, row in df.iterrows():
            total = float(row[f"{prefix}_Total"])
            correct = float(row[f"{prefix}_Correct"])
            wrong = float(row[f"{prefix}_Wrong"])
            xpos = x[i] + offset
            known = np.isfinite(correct) and np.isfinite(wrong)
            if not np.isfinite(total):
                ax.text(xpos, 0, "N/A", ha="center", va="bottom", fontsize=12)
                continue
            if known:
                # Recalculate from the counts for THIS bar, never overall accuracy.
                accuracy = correct / total if total > 0 else np.nan
                ax.bar(xpos, wrong, width, color=wrong_colour, edgecolor="white")
                ax.bar(xpos, correct, width, bottom=wrong,
                       color=correct_colour, edgecolor="white")
                for value, bottom, background in [(wrong, 0, wrong_colour),
                                                   (correct, wrong, correct_colour)]:
                    rgb = np.asarray(to_rgb(background))
                    linear = np.where(rgb <= 0.04045, rgb / 12.92,
                                      ((rgb + 0.055) / 1.055) ** 2.4)
                    luminance = float(linear @ np.array([0.2126, 0.7152, 0.0722]))
                    text_colour = "white" if luminance < 0.179 else "#172B4D"
                    if value > 0:
                        ax.text(xpos, bottom + value / 2, f"{value:,.0f}",
                                ha="center", va="center", fontsize=12,
                                fontweight="bold", color=text_colour)
                acc_text = f"{accuracy:.1%}" if np.isfinite(accuracy) else "N/A"
            else:
                ax.bar(xpos, total, width, color="#D9DEE7", hatch="//", edgecolor="#7A869A")
                acc_text = "N/A"
            ax.annotate(f"{name}\nAcc: {acc_text}\nn={total:,.0f}",
                        (xpos, total), xytext=(0, 6), textcoords="offset points",
                        ha="center", va="bottom", fontsize=11, fontweight="bold")
    maximum = df[["Processed_Total", "Selected_Total"]].max().max()
    ax.set_ylim(0, max(float(maximum), 1) * 1.32)
    ax.set_xticks(x)
    ax.set_xticklabels([f"Link {r.Link}\n{r.Model_Display}" for r in df.itertuples()], fontsize=12)
    ax.set_xlabel("Link (Model)")
    ax.set_ylabel("Number of samples")
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=0.18)
    ax2 = ax.twinx()
    confidence = df["Mean_Confidence"].to_numpy(dtype=float) * 100
    handles = [Patch(facecolor=c, label=label) for c, label in [
        (COLOR_PROC_CORRECT, "Routed: correct"), (COLOR_PROC_WRONG, "Routed: wrong"),
        (COLOR_SEL_CORRECT, "Selected: correct"), (COLOR_SEL_WRONG, "Selected: wrong")]]
    if np.isfinite(confidence).any():
        line, = ax2.plot(x, confidence, "D--", color=COLOR_CONF_LINE,
                         linewidth=2, label="Selected mean confidence")
        handles.append(line)
        for xpos, value in zip(x, confidence):
            if np.isfinite(value):
                ax2.annotate(f"{value:.1f}%", (xpos, value),
                             xytext=(0, 9), textcoords="offset points",
                             ha="center", va="bottom", fontsize=12,
                             fontweight="bold", color=COLOR_CONF_LINE,
                             bbox=dict(facecolor="white", edgecolor="none",
                                       alpha=0.85, pad=1.5), zorder=6)
    ax2.set_ylim(0, 110)
    ax2.set_ylabel("Selected mean confidence (%)", color=COLOR_CONF_LINE)
    ax2.tick_params(axis="y", labelcolor=COLOR_CONF_LINE)
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.94),
               ncol=3, frameon=False, fontsize=12)
    fig.suptitle("Confidence Cascade Routing: Routed vs Selected Samples", fontweight="bold", fontsize=18, y=0.99)
    fig.text(0.5, 0.015,
             "Accuracy = correct / total within each bar. Hatched grey = processed correctness unavailable.",
             ha="center", fontsize=12)
    fig.tight_layout(rect=(0, 0.04, 1, 0.86))
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
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
