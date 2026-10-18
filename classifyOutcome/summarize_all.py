"""
Summarize results from 5 classification methods into a single Excel file.

Inputs (all in classifyOutcome\):
  - BERT:            BERT_output/BERT/bert_test_predictions.xlsx
  - Cascade:         Confidence Cascade Routing/Chain_Ensemble_Simplified.xlsx
  - Single-Model:    Single-model One-shot Inference/Single-Pass Classification-doubao.xlsx
  - Hierarchical:    Hierarchical Binary Decision Routing.xlsx
  - Annotation:      Annotation-Enhanced LLM Classification.xlsx

Output: classifyOutcome\summarizedData.xlsx
"""

import os
import pandas as pd
import numpy as np

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
BASE_DIR = r"C:\Users\1\Desktop\SIT上传的数据\SupplementaryMaterials\Supplementary materials\classifyOutcome"

BERT_XLSX    = os.path.join(BASE_DIR, "BERT_output", "BERT", "bert_test_predictions.xlsx")
CASCADE_XLSX = os.path.join(BASE_DIR, "Confidence Cascade Routing",
                            "Chain_Ensemble_Simplified.xlsx")
SINGLE_XLSX  = os.path.join(BASE_DIR, "Single-model One-shot Inference",
                            "Single-Pass Classification-doubao.xlsx")
HIER_XLSX    = os.path.join(BASE_DIR, "Hierarchical Binary Decision Routing.xlsx")
ANNO_XLSX    = os.path.join(BASE_DIR, "Annotation-Enhanced LLM Classification.xlsx")

OUTPUT_XLSX  = os.path.join(BASE_DIR, "summarizedData.xlsx")


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def to_binary(x):
    if pd.isna(x):
        return 0
    if isinstance(x, (bool, np.bool_)):
        return int(x)
    if isinstance(x, (int, np.integer)):
        return int(x != 0)
    if isinstance(x, (float, np.floating)):
        return int(x != 0)
    s = str(x).strip().lower()
    if s in ("yes", "y", "true", "t", "1", "是", "correct"):
        return 1
    if s in ("no", "n", "false", "f", "0", "否", "wrong", "incorrect"):
        return 0
    try:
        return int(float(s) != 0)
    except Exception:
        return 0


def normalize_sample_id(series):
    return pd.to_numeric(series, errors="coerce").fillna(-1).astype(int)


def clean_columns(df):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def pick_col(df, candidates, name="", required=True):
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise KeyError(f"[{name}] 未找到候选列 {candidates}，实际列：{list(df.columns)}")
    return None


# ----------------------------------------------------------------------------
# 1. BERT (xlsx)
# ----------------------------------------------------------------------------
print(f"Reading BERT: {BERT_XLSX}")
bert_df = clean_columns(pd.read_excel(BERT_XLSX))
print(f"  Columns: {list(bert_df.columns)}")

bert_id_col = pick_col(bert_df, ["sample_id", "Sample_ID", "id", "ID"], name="BERT")
bert_df[bert_id_col] = normalize_sample_id(bert_df[bert_id_col])
bert_df = bert_df.drop_duplicates(subset=bert_id_col, keep="first").reset_index(drop=True)

bert_sheet = pd.DataFrame({
    "sample_id":             bert_df[bert_id_col].astype(int),
    "requirement":           bert_df["requirement"].astype(str),
    "Human_Label":           bert_df["true_label"].astype(str).str.strip().str.lower(),
    "Model_Prediction":      bert_df["bert_pred"].astype(str).str.strip().str.lower(),
    "prediction_confidence": pd.to_numeric(bert_df["bert_confidence"], errors="coerce").fillna(0.0),
    "Is_Correct":            bert_df["bert_correct"].apply(to_binary),
})
print(f"  BERT sheet: {len(bert_sheet)} rows")


# ----------------------------------------------------------------------------
# 2. Confidence Cascade Routing  ->  Chain_Ensemble_Simplified.xlsx
# 列：Sample_Index, Requirement_Content, Human_Label, Final_Predicted_Label,
#      Final_Confidence, Final_Model, Final_Model_Name, Final_Link,
#      Rank, Normalized_Rank, IsCorrect
# ----------------------------------------------------------------------------
print(f"\nReading Cascade: {CASCADE_XLSX}")
casc = clean_columns(pd.read_excel(CASCADE_XLSX))
print(f"  Columns: {list(casc.columns)}")

casc_id_col   = pick_col(casc, ["Sample_Index", "sample_id", "Sample_ID", "id", "ID"], name="Cascade")
casc_pred_col = pick_col(casc, ["Final_Predicted_Label", "Predicted_Label", "Model_Prediction"], name="Cascade")
casc_conf_col = pick_col(casc, ["Final_Confidence", "Confidence", "prediction_confidence"], name="Cascade")

casc[casc_id_col] = normalize_sample_id(casc[casc_id_col])
casc = casc.drop_duplicates(subset=casc_id_col, keep="first").reset_index(drop=True)

cascade_sheet = pd.DataFrame({
    "sample_id":             casc[casc_id_col].astype(int),
    "requirement":           casc["Requirement_Content"].astype(str),
    "Human_Label":           casc["Human_Label"].astype(str).str.strip().str.lower(),
    "Model_Prediction":      casc[casc_pred_col].astype(str).str.strip().str.lower(),
    "prediction_confidence": pd.to_numeric(casc[casc_conf_col], errors="coerce").fillna(0.0),
    "Final_Model":           casc["Final_Model"].astype(str) if "Final_Model" in casc.columns else "",
    "Is_Correct":            casc["IsCorrect"].apply(to_binary),
})
print(f"  Cascade sheet: {len(cascade_sheet)} rows")


# ----------------------------------------------------------------------------
# 3. Single-Model One-Shot (doubao)
# 列：sample_id, requirement, Human_Label, Model_Prediction,
#      prediction_confidence, IsCorrect
# ----------------------------------------------------------------------------
print(f"\nReading Single-Model: {SINGLE_XLSX}")
single = clean_columns(pd.read_excel(SINGLE_XLSX))
print(f"  Columns: {list(single.columns)}")

single_id_col = pick_col(single, ["sample_id", "Sample_ID", "Sample_Index", "id", "ID"], name="Single")
single[single_id_col] = normalize_sample_id(single[single_id_col])
single = single.drop_duplicates(subset=single_id_col, keep="first").reset_index(drop=True)

single_sheet = pd.DataFrame({
    "sample_id":             single[single_id_col].astype(int),
    "requirement":           single["requirement"].astype(str),
    "Human_Label":           single["Human_Label"].astype(str).str.strip().str.lower(),
    "Model_Prediction":      single["Model_Prediction"].astype(str).str.strip().str.lower(),
    "prediction_confidence": pd.to_numeric(single["prediction_confidence"], errors="coerce").fillna(0.0),
    "Is_Correct":            single["IsCorrect"].apply(to_binary),
})
print(f"  Single-Model sheet: {len(single_sheet)} rows")


# ----------------------------------------------------------------------------
# 4. Hierarchical Binary Decision Routing
# ----------------------------------------------------------------------------
print(f"\nReading Hierarchical: {HIER_XLSX}")
hier = clean_columns(pd.read_excel(HIER_XLSX))
print(f"  Columns: {list(hier.columns)}")

hier_id_col = pick_col(hier, ["sample_id", "Sample_ID", "Sample_Index", "id", "ID", "requirement_id"], name="Hier")
hier[hier_id_col] = normalize_sample_id(hier[hier_id_col])
hier = hier.drop_duplicates(subset=hier_id_col, keep="first").reset_index(drop=True)


def infer_steps_hier(row):
    sc = row.get("step_count", None)
    if pd.isna(sc):
        return 1
    try:
        return int(sc)
    except Exception:
        return 1


hier_sheet = pd.DataFrame({
    "sample_id":             hier[hier_id_col].astype(int),
    "requirement":           hier["requirement_text"].astype(str),
    "Human_Label":           hier["true_label"].astype(str).str.strip().str.lower(),
    "Model_Prediction":      hier["predicted_label"].astype(str).str.strip().str.lower(),
    "prediction_confidence": pd.to_numeric(hier["prediction_confidence"], errors="coerce").fillna(0.0),
    "Steps Used":            hier.apply(infer_steps_hier, axis=1),
    "Is_Correct":            hier["is_correct"].apply(to_binary),
})
print(f"  Hierarchical sheet: {len(hier_sheet)} rows")


# ----------------------------------------------------------------------------
# 5. Annotation-Enhanced LLM Classification
# ----------------------------------------------------------------------------
print(f"\nReading Annotation-Enhanced: {ANNO_XLSX}")
anno = clean_columns(pd.read_excel(ANNO_XLSX))
print(f"  Columns: {list(anno.columns)}")

anno_id_col = pick_col(anno, ["sample_id", "Sample_ID", "Sample_Index", "id", "ID"], name="Anno")
anno[anno_id_col] = normalize_sample_id(anno[anno_id_col])
anno = anno.drop_duplicates(subset=anno_id_col, keep="first").reset_index(drop=True)


def infer_steps_anno(row):
    if "round" in row.index and pd.notna(row["round"]):
        try:
            return int(row["round"])
        except Exception:
            pass
    r2 = row.get("round2_pred", None)
    if pd.notna(r2):
        return 2
    return 1


anno_sheet = pd.DataFrame({
    "sample_id":             anno[anno_id_col].astype(int),
    "requirement":           anno["requirement"].astype(str),
    "Human_Label":           anno["Human_Label"].astype(str).str.strip().str.lower(),
    "Model_Prediction":      anno["Model_Prediction"].astype(str).str.strip().str.lower(),
    "prediction_confidence": pd.to_numeric(anno["prediction_confidence"], errors="coerce").fillna(0.0),
    "Steps Used":            anno.apply(infer_steps_anno, axis=1),
    "Is_Correct":            anno["IsCorrect"].apply(to_binary),
})
print(f"  Annotation sheet: {len(anno_sheet)} rows")


# ----------------------------------------------------------------------------
# Sanity report
# ----------------------------------------------------------------------------
print("\n================ Summary ================")
print(f"  BERT sheet:         {len(bert_sheet)} rows")
print(f"  Cascade sheet:      {len(cascade_sheet)} rows")
print(f"  Single-Model sheet: {len(single_sheet)} rows")
print(f"  Hierarchical sheet: {len(hier_sheet)} rows")
print(f"  Annotation sheet:   {len(anno_sheet)} rows")

ref_ids = set(bert_sheet["sample_id"])
for name, df in [
    ("Cascade",      cascade_sheet),
    ("Single-Model", single_sheet),
    ("Hierarchical", hier_sheet),
    ("Annotation",   anno_sheet),
]:
    ids = set(df["sample_id"])
    missing = ref_ids - ids
    extra   = ids - ref_ids
    if missing or extra:
        print(f"  ⚠️ {name}: missing={len(missing)}, extra={len(extra)}")
    else:
        print(f"  ✅ {name}: fully aligned with BERT")
print("=========================================\n")


# ----------------------------------------------------------------------------
# Write out
# ----------------------------------------------------------------------------
print(f"Writing to {OUTPUT_XLSX}")
with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl") as writer:
    bert_sheet.to_excel(writer, sheet_name="BERT", index=False)
    cascade_sheet.to_excel(writer, sheet_name="Confidence Cascade Routing", index=False)
    single_sheet.to_excel(writer, sheet_name="Single-Model One-Shot (doubao)", index=False)
    hier_sheet.to_excel(writer, sheet_name="Hierarchical Binary Routing", index=False)
    anno_sheet.to_excel(writer, sheet_name="Annotation-Enhanced", index=False)

print("Done.")