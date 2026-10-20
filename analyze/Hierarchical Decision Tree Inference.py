import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

LABELS = ("composite", "interactive", "sequential", "conditional", "structural")
STEP_TARGETS = {1: "composite", 2: "interactive", 3: "sequential", 4: "conditional"}
STEP_NAMES = {1: "Composite Check", 2: "User Action Check",
              3: "System Condition Check", 4: "Conditional Check"}
STEP_DECISIONS = {1: "composite_marker_check", 2: "interactive_check",
                  3: "sequential_check", 4: "conditional_check"}


DECISION_TREE_RULES = {
    step: {label: label == target for label in LABELS}
    for step, target in STEP_TARGETS.items()
}
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = SCRIPT_DIR.parent / "classifyOutcome" / "Hierarchical Binary Decision Routing.xlsx"
DEFAULT_OUTPUT = SCRIPT_DIR.parent / "analyzeOutcome" / "Hierarchical Decision Tree Inference"


def parse_binary(value, context="result"):

    if pd.isna(value):
        raise ValueError(f"Missing {context}")
    key = str(value).strip().lower()
    if key in {"pass", "true", "yes", "1", "1.0", "通过"}:
        return True
    if key in {"fail", "reject", "rejection", "false", "no", "0", "0.0", "不通过"}:
        return False
    raise ValueError(f"Invalid {context}: {value!r}")


def normalize_label(value):
    label = str(value).strip().lower()
    if label not in LABELS:
        raise ValueError(f"Unknown class label: {value!r}")
    return label


def validate_confidences(values, context="confidence"):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise ValueError(f"{context} must contain finite numeric values")
    if np.any((values < 0) | (values > 1)):
        raise ValueError(f"{context} must be probabilities in [0, 1], not percentages")
    return values


def load_data(file_path):
    df = pd.read_excel(Path(file_path))
    required = {"true_label", "predicted_label", "prediction_confidence"}
    for step in STEP_TARGETS:
        required.update({f"step{step}_decision", f"step{step}_result", f"step{step}_confidence"})
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    if df.empty:
        raise ValueError("Input workbook contains no samples")
    for col in ("true_label", "predicted_label"):
        df[col] = df[col].map(normalize_label)
    if "sample_id" in df and (df.sample_id.isna().any() or df.sample_id.duplicated().any()):
        raise ValueError("sample_id must be present and unique for every row")
    validate_routing(df)
    return df


def validate_routing(df):

    for index, row in df.iterrows():
        active = True
        final = "structural"
        reached = 0
        for step, target in STEP_TARGETS.items():
            cols = [f"step{step}_{suffix}" for suffix in ("decision", "result", "confidence")]
            present = [pd.notna(row[col]) and str(row[col]).strip() != "" for col in cols]
            if active:
                if not all(present):
                    raise ValueError(f"Excel row {index + 2}: incomplete reached step {step}")
                decision = str(row[cols[0]]).strip().lower()
                if decision != STEP_DECISIONS[step]:
                    raise ValueError(f"Excel row {index + 2}: unexpected step {step} name {decision!r}")
                validate_confidences([row[cols[2]]], f"row {index + 2}, step {step}")
                reached += 1
                if parse_binary(row[cols[1]], f"row {index + 2}, step {step}"):
                    final = target
                    active = False
            elif any(present):
                raise ValueError(f"Excel row {index + 2}: step {step} recorded after a terminal PASS")
        if final != row.predicted_label:
            raise ValueError(f"Excel row {index + 2}: path gives {final}, final prediction is {row.predicted_label}")
        if "step_count" in df and int(row.step_count) != reached:
            raise ValueError(f"Excel row {index + 2}: step_count does not match the path")
        if "is_correct" in df:
            logged = parse_binary(row.is_correct, "is_correct")
            if logged != (row.true_label == row.predicted_label):
                raise ValueError(f"Excel row {index + 2}: is_correct disagrees with labels")
    validate_confidences(df.prediction_confidence, "prediction_confidence")


def get_ground_truth_binary_label(final_label, round_num):
    if round_num not in STEP_TARGETS:
        raise ValueError(f"Unknown step: {round_num}")
    return normalize_label(final_label) == STEP_TARGETS[round_num]


def get_round_binary_predictions(df, round_num):

    result_col = f"step{round_num}_result"
    reached = df[result_col].notna() & df[result_col].astype(str).str.strip().ne("")
    subset = df.loc[reached]
    predictions = [parse_binary(v, result_col) for v in subset[result_col]]
    confidences = validate_confidences(subset[f"step{round_num}_confidence"])
    return subset.index.tolist(), predictions, confidences


def calculate_ece(confidences, correctness, n_bins=10):

    if n_bins < 1:
        raise ValueError("n_bins must be positive")
    c = validate_confidences(confidences)
    y = np.asarray(correctness)
    if y.ndim != 1 or len(c) != len(y) or np.any((y != 0) & (y != 1)):
        raise ValueError("Correctness must be a matching vector of zeros and ones")
    y = y.astype(int)
    if len(c) == 0:
        return float("nan"), []
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    indices = np.clip(np.searchsorted(bins, c, side="right") - 1, 0, n_bins - 1)
    stats = []
    for i in range(n_bins):
        mask = indices == i
        if mask.any():
            stats.append({"bin_lower": float(bins[i]), "bin_upper": float(bins[i+1]),
                          "includes_upper": i == n_bins-1,
                          "bin_center": float((bins[i]+bins[i+1])/2),
                          "accuracy": float(y[mask].mean()), "confidence": float(c[mask].mean()),
                          "count": int(mask.sum()), "weight": float(mask.mean())})
    assert sum(s["count"] for s in stats) == len(c)
    ece = sum(s["weight"] * abs(s["accuracy"] - s["confidence"]) for s in stats)
    assert ece + 1e-12 >= abs(y.mean() - c.mean())
    return float(ece), stats


def calculate_metrics(confidences, correctness, n_bins=10):
    c = validate_confidences(confidences)
    y = np.asarray(correctness, dtype=bool)
    if len(c) != len(y) or len(y) == 0:
        raise ValueError("Metrics require a nonempty, aligned sample set")
    ece, stats = calculate_ece(c, y, n_bins)
    r = float(pearsonr(c, y.astype(int)).statistic) if len(y)>1 and np.ptp(c)>0 and np.ptp(y.astype(int))>0 else float("nan")
    return {"n_samples": len(y), "n_correct": int(y.sum()),
            "accuracy": float(y.mean()*100), "avg_confidence": float(c.mean()*100),
            "ece": ece*100, "pearson_corr": r*100, "bin_stats": stats}


def calculate_round_binary_metrics(df, round_num, true_label_col="true_label", n_bins=10):
    indices, predictions, confidences = get_round_binary_predictions(df, round_num)
    if not indices:
        return None
    true = np.array([get_ground_truth_binary_label(df.loc[i,true_label_col], round_num) for i in indices], bool)
    pred = np.asarray(predictions, bool)
    result = calculate_metrics(confidences, true == pred, n_bins)
    result.update({"round": round_num, "step_name": STEP_NAMES[round_num],
                   "target_class": STEP_TARGETS[round_num],
                   "n_true_pass": int(true.sum()), "n_true_reject": int((~true).sum()),
                   "n_pred_pass": int(pred.sum()), "n_pred_reject": int((~pred).sum()),
                   "tp": int((true&pred).sum()), "tn": int((~true&~pred).sum()),
                   "fp": int((~true&pred).sum()), "fn": int((true&~pred).sum()),
                   "true_labels": true, "predictions": pred, "confidences": confidences,
                   "sample_indices": indices})
    result["n_binary_correct"] = result["n_correct"]
    result["binary_accuracy"] = result["accuracy"]
    result["n_correct"] = result["tp"]
    result["accuracy"] = result["tp"] / result["n_samples"] * 100
    return result


def final_metrics(df, n_bins=10):


    return calculate_metrics(df.prediction_confidence, df.true_label == df.predicted_label, n_bins)


def summary_row(m):
    row = {"Samples": m["n_samples"], "Correct": m["n_correct"],
            "Accuracy (%)": m["accuracy"], "Avg Confidence (%)": m["avg_confidence"],
            "ECE (%)": m["ece"], "Pearson Correlation (%)": m["pearson_corr"]}
    if "n_binary_correct" in m:
        row.update({"Binary correct (TP+TN)": m["n_binary_correct"],
                    "Binary decision accuracy (%)": m["binary_accuracy"],
                    "Accuracy definition": "TP / actual node samples",
                    "Calibration basis": "Binary PASS/FAIL decision correctness"})
    else:
        row.update({"Accuracy definition": "Correct final category predictions / all samples",
                    "Calibration basis": "Final predicted category matches reference label"})
    return row


def export_round_results_to_excel(round_metrics, overall, df, output_file):

    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        summary = [{"Step": m["step_name"], "Round": m["round"], "PASS target": m["target_class"],
                    **summary_row(m), "True Pass": m["n_true_pass"], "True Reject": m["n_true_reject"],
                    "Pred Pass": m["n_pred_pass"], "Pred Reject": m["n_pred_reject"],
                    "TP": m["tp"], "TN": m["tn"], "FP": m["fp"], "FN": m["fn"]} for m in round_metrics]
        pd.DataFrame(summary).to_excel(writer, sheet_name="Round Performance Summary", index=False)
        pd.DataFrame([summary_row(overall)]).to_excel(writer, sheet_name="Final Classification Summary", index=False)
        calib = []
        for name,m in [(m["step_name"],m) for m in round_metrics]+[("Final classification",overall)]:
            for b in m["bin_stats"]:
                calib.append({"Step": name, **b})
        pd.DataFrame(calib).to_excel(writer, sheet_name="Calibration Statistics", index=False)
        for m in round_metrics:
            details=[]
            for k,idx in enumerate(m["sample_indices"]):
                t,p = bool(m["true_labels"][k]), bool(m["predictions"][k])
                details.append({"Excel source row": int(idx)+2,
                                "sample_id": df.loc[idx,"sample_id"] if "sample_id" in df else idx,
                                "True category": df.loc[idx,"true_label"],
                                "PASS target": m["target_class"],
                                "True binary": "PASS" if t else "FAIL",
                                "Predicted binary": "PASS" if p else "FAIL",
                                "Confidence": float(m["confidences"][k]),
                                "Correct target assignment (TP)": t and p,
                                "Binary decision correct": t==p,
                                "Confusion cell": "TP" if t and p else "TN" if not t and not p else "FP" if p else "FN"})
            pd.DataFrame(details).to_excel(writer,sheet_name=f"Step{m['round']}_binary_details",index=False)
        keep=[c for c in ("sample_id","true_label","predicted_label","prediction_confidence","step_count") if c in df]
        final=df[keep].copy();final["Recomputed correct"]=df.true_label==df.predicted_label
        final.to_excel(writer,sheet_name="Final prediction details",index=False)
        pd.DataFrame({"Definition":[
            "PASS targets are Composite, Interactive, Sequential and Conditional, in order.",
            "Each node uses all actual visits, including samples misrouted by earlier checks.",
            "Node Accuracy (%) = TP / all actual visits to this node, multiplied by 100.",
            "Correct in the node summary counts only TP. Correctly forwarded negatives (TN) are not included.",
            "Binary decision accuracy (%) = (TP + TN) / all actual node visits, multiplied by 100.",
            "Node ECE and Pearson use binary PASS/FAIL correctness, not TP-only correctness.",
            "Node confidence belongs to the binary decision. A confident correct FAIL is not a correct final target-category assignment.",
            "Final accuracy uses one final prediction per requirement, not a weighted average of node accuracies.",
            "Correct final predictions include correct Structural assignments from the default branch.",
            "Final confidence uses prediction_confidence as recorded in the source log.",
            "Calibration statistics contain proportions in [0,1]; summary columns marked (%) are multiplied by 100.",
            "The final ECE bin includes confidence 1.0. No confidence observations are discarded.",
            "Undefined Pearson correlations are exported as N/A."]}).to_excel(writer,sheet_name="Metric Definitions",index=False)

        for sheet in writer.book.worksheets:
            sheet.freeze_panes="A2"
            sheet.auto_filter.ref=sheet.dimensions
            for col in sheet.columns:
                sheet.column_dimensions[col[0].column_letter].width=25
            for row in sheet.iter_rows(min_row=2):
                for cell in row:
                    if isinstance(cell.value,float):cell.number_format="0.0000"
        for name in ("Round Performance Summary","Final Classification Summary"):
            sheet=writer.book[name]
            header={c.value:c.column for c in sheet[1]}
            for row in range(2,sheet.max_row+1):
                c=sheet.cell(row,header["Pearson Correlation (%)"])
                if c.value is None:c.value="N/A"


def create_performance_bar_charts(round_metrics, overall, output_file, dpi=300):
    plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":11,
                         "axes.unicode_minus":False})
    cmap=plt.get_cmap("viridis")
    colors={"total":cmap(.15), "tp":cmap(.38), "confidence":cmap(.63),
            "pearson":cmap(.84), "accuracy":cmap(.04)}
    short_names={1:"Composite Check",2:"Interactive Check",3:"Sequential Check",4:"Conditional Check"}
    names=[short_names[m["round"]] for m in round_metrics]
    x=np.arange(len(round_metrics))
    totals=np.array([m["n_samples"] for m in round_metrics])
    correct=np.array([m["tp"] for m in round_metrics])
    accuracy=correct/totals*100
    confidence=np.array([m["avg_confidence"] for m in round_metrics])
    pearson=np.array([m["pearson_corr"]/100 for m in round_metrics])
    finite=np.isfinite(pearson)
    pearson_scaled=np.where(finite,pearson*100,0)

    fig,ax_count=plt.subplots(figsize=(17,8.5))
    ax_rate=ax_count.twinx()
    width=.17
    total_bars=ax_count.bar(x-.30,totals,width,color=colors["total"],
                            label="Processed requirements",zorder=3)
    tp_bars=ax_count.bar(x-.10,correct,width,color=colors["tp"],
                         label="Correct classifications",zorder=3)
    conf_bars=ax_rate.bar(x+.10,confidence,width,color=colors["confidence"],
                          label="Mean confidence (%)",zorder=3)
    r_bars=ax_rate.bar(x+.30,pearson_scaled,width,color=colors["pearson"],
                       label="Pearson correlation",zorder=3)
    line,=ax_rate.plot(x,accuracy,color=colors["accuracy"],marker="o",markersize=7,
                       linewidth=2.5,label="Accuracy (%)",zorder=5)

    right_min=min(0,float(pearson_scaled.min())*1.2)
    right_max=115
    left_max=max(float(totals.max()),1)*1.20
    ax_rate.set_ylim(right_min,right_max)
    ax_count.set_ylim(left_max*right_min/right_max,left_max)
    ax_count.set_ylabel("Number of requirements",fontsize=12)
    ax_rate.set_ylabel("Accuracy / confidence (%) and Pearson r x 100",fontsize=12)
    ax_count.set_xticks(x,names)
    ax_count.set_xlabel("Decision node",fontsize=12,labelpad=10)
    ax_count.set_axisbelow(True)
    ax_count.grid(axis="y",alpha=.20)
    ax_count.spines["top"].set_visible(False)
    ax_rate.spines["top"].set_visible(False)

    ax_count.bar_label(total_bars,labels=[str(v) for v in totals],padding=5,fontsize=11,fontweight="bold")
    ax_count.bar_label(tp_bars,labels=[str(v) for v in correct],padding=5,fontsize=11,fontweight="bold")
    ax_rate.bar_label(conf_bars,labels=[f"{v:.2f}%" for v in confidence],padding=5,fontsize=10,fontweight="bold")
    for i,bar in enumerate(r_bars):
        value=pearson_scaled[i]
        ax_rate.annotate(f"r={pearson[i]:.4f}" if finite[i] else "N/A",
                         (bar.get_x()+bar.get_width()/2,bar.get_height()),
                         xytext=(0,5 if value>=0 else -15),textcoords="offset points",
                         ha="center",fontsize=10,fontweight="bold")
    for pos,value in zip(x,accuracy):
        ax_rate.annotate(f"{value:.2f}%",(pos,value),xytext=(0,17),
                         textcoords="offset points",ha="center",va="bottom",
                         fontsize=10,fontweight="bold",color=colors["accuracy"],
                         bbox={"facecolor":"white","edgecolor":"none","alpha":.88,"pad":1.5})

    fig.suptitle("Hierarchical Classification: Node-level Performance",fontsize=17,fontweight="bold",y=.98)
    fig.legend(handles=[total_bars,tp_bars,conf_bars,r_bars,line],
               loc="upper center",bbox_to_anchor=(.5,.93),ncol=3,frameon=False,fontsize=11)
    fig.subplots_adjust(top=.78,bottom=.12,left=.07,right=.92)
    fig.savefig(output_file,dpi=dpi,bbox_inches="tight",facecolor="white")
    plt.close(fig)


def print_summary(round_metrics, overall):
    print("Node accuracy = TP / actual node samples; ECE and r use binary decision correctness")
    for m in round_metrics:
        print(f"{m['step_name']}: {m['n_correct']}/{m['n_samples']} = {m['accuracy']:.2f}%; "
              f"TP={m['tp']}, TN={m['tn']}, FP={m['fp']}, FN={m['fn']}; "
              f"binary accuracy={m['binary_accuracy']:.2f}%; "
              f"confidence={m['avg_confidence']:.2f}%, binary ECE={m['ece']:.2f}%, binary r={m['pearson_corr']/100:.4f}")
    print(f"Final classification: {overall['n_correct']}/{overall['n_samples']} = {overall['accuracy']:.2f}%; "
          f"confidence={overall['avg_confidence']:.2f}%, ECE={overall['ece']:.2f}%, r={overall['pearson_corr']/100:.4f}")


def main():
    parser=argparse.ArgumentParser(description="Single Viridis figure: node counts, TP/N accuracy, confidence and Pearson r",formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input",type=Path,default=DEFAULT_INPUT)
    parser.add_argument("--output-dir",type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument("--bins",type=int,default=10)
    parser.add_argument("--dpi",type=int,default=300)
    args=parser.parse_args()
    if args.bins<1 or args.dpi<1:parser.error("--bins and --dpi must be positive")
    df=load_data(args.input)
    rounds=[calculate_round_binary_metrics(df,i,n_bins=args.bins) for i in STEP_TARGETS]
    rounds=[m for m in rounds if m is not None]
    overall=final_metrics(df,args.bins)
    structural_correct=int(((df.true_label=="structural") & (df.predicted_label=="structural")).sum())
    assert sum(m["tp"] for m in rounds)+structural_correct==overall["n_correct"]
    print_summary(rounds,overall)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    export_round_results_to_excel(rounds,overall,df,args.output_dir/"Hierarchical_Binary_Decision_Tree_Analysis.xlsx")
    create_performance_bar_charts(rounds,overall,args.output_dir/"Hierarchical_Node_Performance_Single_Figure.png",args.dpi)
    public=[]
    for m in rounds:
        public.append({"step":m["step_name"],"target_class":m["target_class"],**summary_row(m),
                       "TP":m["tp"],"TN":m["tn"],"FP":m["fp"],"FN":m["fn"]})
    payload={"input":args.input.name,"ece_bins":args.bins,"nodes":public,
             "correct_structural_default":structural_correct,"final":summary_row(overall)}

    def clean(value):
        if isinstance(value,float) and not np.isfinite(value):return None
        if isinstance(value,dict):return {k:clean(v) for k,v in value.items()}
        if isinstance(value,list):return [clean(v) for v in value]
        return value
    (args.output_dir/"metrics.json").write_text(json.dumps(clean(payload),ensure_ascii=False,indent=2),encoding="utf-8")
    report=["Node accuracy = TP / actual node samples. Binary accuracy is reported separately."]
    for m in rounds:
        report.append(f"{m['step_name']}: {m['tp']}/{m['n_samples']} = {m['accuracy']:.2f}%")
    report.append(f"Correct Structural predictions from the default branch: {structural_correct}")
    report.append(f"Final accuracy: {overall['n_correct']}/{overall['n_samples']} = {overall['accuracy']:.2f}%")
    report.append("Node ECE and Pearson r evaluate binary decisions and are not calculated from TP-only correctness.")
    (args.output_dir/"analysis_report.txt").write_text("\n".join(report)+"\n",encoding="utf-8")
    print(f"Outputs saved to {args.output_dir.resolve()}")


if __name__=="__main__":
    main()
