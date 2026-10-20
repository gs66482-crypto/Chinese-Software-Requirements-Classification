import os
import time
import re
import random
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


class Config:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
    DATA_DIR = os.path.join(PROJECT_ROOT, "data")

    CONCEPT_FILE = os.path.join(DATA_DIR, "Concept.xlsx")
    EXAMPLE_FILE = os.path.join(DATA_DIR, "RequirementExamples.xlsx")

    BERT_XLSX = os.path.join(
        PROJECT_ROOT, "classifyOutcome", "BERT_output", "BERT",
        "bert_test_predictions.xlsx"
    )

    OUTPUT_DIR = os.path.join(PROJECT_ROOT, "classifyOutcome")
    OUTPUT_FILE = os.path.join(
        OUTPUT_DIR, "Annotation-Enhanced LLM Classification.xlsx"
    )

    MODEL_NAME = os.getenv("DOUBAO_MODEL", "doubao-seed-2-0-lite-260428")
    API_KEY_ENV = "DOUBAO_API_KEY"
    BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"

    TEMPERATURE = 0.3
    MAX_TOKENS = 300
    SEED = 42

    MAX_EXAMPLES_PER_CLASS = None

    REQUEST_INTERVAL = 1.0
    MAX_RETRIES = 5
    BASE_BACKOFF = 2.0

    MAX_ROUNDS = 3


class KnowledgeBaseLoader:
    def __init__(self, concept_path: str, examples_path: str,
                 max_examples_per_class: Optional[int] = None):
        self.concept_path = concept_path
        self.examples_path = examples_path
        self.max_examples_per_class = max_examples_per_class

        self.category_definitions: Dict[str, str] = {}
        self.few_shot_examples: Dict[str, List[str]] = {}

        self._load_concept()
        self._load_examples()

    def _load_concept(self):
        df = pd.read_excel(self.concept_path, sheet_name=0)
        df.columns = [str(c).strip() for c in df.columns]
        cat_col = df.columns[0]
        exp_col = df.columns[1]
        for _, row in df.iterrows():
            cat = str(row[cat_col]).strip().lower()
            exp = str(row[exp_col]).strip()
            if cat and cat != "nan":
                self.category_definitions[cat] = exp
        print(f"[KB] Loaded {len(self.category_definitions)} category definitions")

    def _load_examples(self):
        df = pd.read_excel(self.examples_path, sheet_name=0, header=None)
        for i in range(1, len(df)):
            cat = str(df.iloc[i, 0]).strip().lower()
            if not cat or cat == "nan":
                continue
            examples = []
            for j in range(1, df.shape[1]):
                val = df.iloc[i, j]
                if pd.isna(val):
                    continue
                text = str(val).strip()
                if text and text.lower() != "nan":
                    examples.append(text)
            if self.max_examples_per_class is None:
                self.few_shot_examples[cat] = examples
            else:
                self.few_shot_examples[cat] = examples[:self.max_examples_per_class]

        total = sum(len(v) for v in self.few_shot_examples.values())
        print(f"[KB] Loaded {total} few-shot examples across "
              f"{len(self.few_shot_examples)} categories")
        for cat, exs in self.few_shot_examples.items():
            print(f"     - {cat}: {len(exs)} examples")

    def get_category_list_text(self) -> str:
        lines = []
        for cat, exp in self.category_definitions.items():
            lines.append(f"- {cat}: {exp}")
        return "\n".join(lines)

    def get_few_shot_text(self) -> str:
        lines = []
        for cat, examples in self.few_shot_examples.items():
            if not examples:
                continue
            lines.append(f"【{cat}】")
            for k, ex in enumerate(examples, 1):
                lines.append(f"  示例{k}: {ex}")
        return "\n".join(lines)


def build_system_prompt(kb: KnowledgeBaseLoader) -> str:
    base_role = ("你是一位专业的软件需求分类专家。"
                 "请根据给定的类别定义对需求进行分类。")
    category_text = kb.get_category_list_text()
    few_shot_text = kb.get_few_shot_text()
    return f"""{base_role}

【类别定义】（来源：Concept.xlsx）
{category_text}

【分类示例】（来源：RequirementExamples.xlsx）
{few_shot_text}"""


class PromptBuilder:
    @staticmethod
    def build_round1_user_prompt(text: str) -> str:
        return f"""请对以下软件需求进行分类。

## 待分类需求：
{text}

## 输出格式：
分类: [标签名称]
置信度: [0-1之间的数值]

请严格按照上述格式输出。"""

    @staticmethod
    def build_round2_user_prompt(text: str, human_label: str,
                                 previous_label: str) -> str:
        return f"""你需要重新思考刚才的分类。

## 待分类需求：
{text}

## 你刚才的分类是：
{previous_label}

## 人类专家的标注是：
{human_label}

## 请从软件需求分类专家的专业角度重新分析：
1. 为什么人类专家会有不同的标注？
2. 重新评估后，给出你认为最合适的分类标签
3. 给出你的分类置信度（0-1之间的数值）

## 输出格式：
【重新分析】[你的分析]
【分类标签】[标签名称]
【置信度】[0-1之间的数值]

请严格按照上述格式输出。"""

    @staticmethod
    def build_round3_user_prompt(text: str, human_label: str,
                                 previous_label: str) -> str:
        return f"""这是你最后一次重新思考的机会。

## 待分类需求：
{text}

## 你之前的分类是：
{previous_label}

## 人类专家的标注一直是：
{human_label}

## 请从软件需求分类专家的专业角度：
1. 给出你认为最合适的分类标签
2. 分析为什么你的分类与人类专家不同
3. 给出你的分类置信度（0-1之间的分数）

## 输出格式：
【最终分析】[你的分析，包括可能的视角差异]
【分类标签】[标签名称]
【置信度】[0-1之间的数值]
【原因分析】[解释为什么与人类标注不同，50字以内]

请严格按照上述格式输出。"""


class ModelClient:
    def __init__(self, kb: KnowledgeBaseLoader):
        self.model_name = Config.MODEL_NAME
        self.api_key = os.getenv(Config.API_KEY_ENV)
        self.base_url = Config.BASE_URL

        if not self.api_key:
            raise ValueError(f"API key not found: {Config.API_KEY_ENV}")

        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

        self.system_prompt = build_system_prompt(kb)

        self.max_retries = Config.MAX_RETRIES
        self.base_backoff = Config.BASE_BACKOFF
        self.request_interval = Config.REQUEST_INTERVAL
        self.last_request_time = 0

    def _wait_if_needed(self):
        elapsed = time.time() - self.last_request_time
        if elapsed < self.request_interval:
            time.sleep(self.request_interval - elapsed)
        self.last_request_time = time.time()

    def _call_model_with_retry(self, user_prompt: str) -> str:
        for attempt in range(self.max_retries):
            try:
                self._wait_if_needed()

                messages = [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_prompt},
                ]

                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=Config.TEMPERATURE,
                    max_tokens=Config.MAX_TOKENS,
                )
                return response.choices[0].message.content.strip()

            except Exception as e:
                error_msg = str(e).lower()
                if "rate" in error_msg or "429" in error_msg or "throttle" in error_msg:
                    wait = self.base_backoff * (2 ** attempt) + random.uniform(0, 0.5)
                    print(f"   ⚠️ Rate limit, retry in {wait:.1f}s "
                          f"({attempt + 1}/{self.max_retries})")
                    time.sleep(wait)
                else:
                    print(f"   ❌ API error: {e}")
                    if attempt < self.max_retries - 1:
                        wait = self.base_backoff * (2 ** attempt)
                        print(f"   🔄 Retry in {wait:.1f}s "
                              f"({attempt + 1}/{self.max_retries})")
                        time.sleep(wait)
                    else:
                        raise

        raise Exception(f"Max retries exceeded ({self.max_retries})")

    def classify_with_retry(self, text: str,
                            human_label: Optional[str] = None) -> Dict[str, Any]:
        labels = self._extract_labels_from_prompt()

        print(f"   🔄 Round 1: Initial classification...")
        user1 = PromptBuilder.build_round1_user_prompt(text)
        resp1 = self._call_model_with_retry(user1)
        label1, conf1 = self._parse_response(resp1, labels)

        round1_correct = None
        if human_label:
            round1_correct = "Yes" if label1 == human_label else "No"

        print(f"      Prediction: {label1} (confidence: {conf1:.4f})")
        if human_label:
            print(f"      Human label: {human_label} - {round1_correct}")

        if not human_label or label1 == human_label:
            return {
                "final_label": label1,
                "final_confidence": conf1,
                "total_rounds": 1,
                "round1_pred": label1,
                "round1_conf": conf1,
                "round1_correct": round1_correct,
                "round2_pred": None,
                "round2_conf": None,
                "round2_correct": None,
                "round3_pred": None,
                "round3_conf": None,
                "round3_correct": None,
                "disagreement_reason": "",
            }

        print(f"   🔄 Round 2: Rethinking...")
        time.sleep(1)
        user2 = PromptBuilder.build_round2_user_prompt(text, human_label, label1)
        resp2 = self._call_model_with_retry(user2)
        label2, conf2 = self._parse_response(resp2, labels)

        round2_correct = "Yes" if label2 == human_label else "No"
        print(f"      Prediction: {label2} (confidence: {conf2:.4f})")
        print(f"      Human label: {human_label} - {round2_correct}")

        if label2 == human_label:
            return {
                "final_label": label2,
                "final_confidence": conf2,
                "total_rounds": 2,
                "round1_pred": label1,
                "round1_conf": conf1,
                "round1_correct": round1_correct,
                "round2_pred": label2,
                "round2_conf": conf2,
                "round2_correct": round2_correct,
                "round3_pred": None,
                "round3_conf": None,
                "round3_correct": None,
                "disagreement_reason": "",
            }

        print(f"   🔄 Round 3: Final thinking...")
        time.sleep(1)
        user3 = PromptBuilder.build_round3_user_prompt(text, human_label, label2)
        resp3 = self._call_model_with_retry(user3)
        label3, conf3 = self._parse_response(resp3, labels)
        reason = self._extract_disagreement_reason(resp3)

        round3_correct = "Yes" if label3 == human_label else "No"
        print(f"      Final prediction: {label3} (confidence: {conf3:.4f})")
        print(f"      Human label: {human_label} - {round3_correct}")
        if reason:
            print(f"      Disagreement reason: {reason[:100]}...")

        return {
            "final_label": label3,
            "final_confidence": conf3,
            "total_rounds": 3,
            "round1_pred": label1,
            "round1_conf": conf1,
            "round1_correct": round1_correct,
            "round2_pred": label2,
            "round2_conf": conf2,
            "round2_correct": round2_correct,
            "round3_pred": label3,
            "round3_conf": conf3,
            "round3_correct": round3_correct,
            "disagreement_reason": reason,
        }

    def _extract_labels_from_prompt(self) -> List[str]:
        try:
            block = self.system_prompt.split("【类别定义】")[1].split("【分类示例】")[0]
            labels = []
            for line in block.strip().split("\n"):
                line = line.strip()
                if line.startswith("- "):
                    label = line[2:].split(":")[0].strip()
                    if label:
                        labels.append(label)
            return labels
        except Exception:
            return []

    def _parse_response(self, response: str, labels: List[str]) -> Tuple[str, float]:
        if not response or not labels:
            return (labels[0] if labels else "Unknown"), 0.0

        response_clean = response.strip()
        label = ""
        confidence = 0.0

        label_patterns = [
            r'【分类标签】\s*[:：]?\s*(.+?)(?:\n|$)',
            r'分类标签[:：]\s*(.+?)(?:\n|$)',
            r'分类[:：]\s*(.+?)(?:\n|$)',
        ]
        for pattern in label_patterns:
            m = re.search(pattern, response_clean, re.IGNORECASE | re.MULTILINE)
            if m:
                extracted = m.group(1).strip().strip('.,!?;:"\'')
                for cand in labels:
                    if extracted.lower() == cand.lower():
                        label = cand
                        break
                    elif cand.lower() in extracted.lower() or extracted.lower() in cand.lower():
                        label = cand
                        break
                if label:
                    break

        if not label:
            for cand in labels:
                if cand.lower() in response_clean.lower():
                    label = cand
                    break
        if not label:
            label = labels[0] if labels else "Unknown"

        confidence_patterns = [
            r'【置信度】\s*[:：]?\s*([0-9]*\.?[0-9]+)',
            r'置信度[:：]\s*([0-9]*\.?[0-9]+)',
        ]
        for pattern in confidence_patterns:
            m = re.search(pattern, response_clean, re.IGNORECASE | re.MULTILINE)
            if m:
                try:
                    confidence = max(0.0, min(1.0, float(m.group(1))))
                    break
                except ValueError:
                    continue
        return label, confidence

    def _extract_disagreement_reason(self, response: str) -> str:
        if not response:
            return ""
        for pattern in [
            r'【原因分析】\s*[:：]?\s*(.+?)(?:\n|$)',
            r'原因分析[:：]\s*(.+?)(?:\n|$)',
            r'不一致原因[:：]\s*(.+?)(?:\n|$)',
            r'视角差异[:：]\s*(.+?)(?:\n|$)',
        ]:
            m = re.search(pattern, response.strip(), re.IGNORECASE | re.MULTILINE)
            if m:
                reason = m.group(1).strip()
                if len(reason) > 5:
                    return reason
        return ""


def main():
    print("=" * 80)
    print("Annotation-Enhanced LLM Classification (Unified)")
    print(f"   Model: {Config.MODEL_NAME}")
    print("   Adaptive Multi-Round Retry (Max 3 rounds)")
    print("=" * 80)

    print(f"\nPath Information:")
    print(f"  Script Directory: {Config.SCRIPT_DIR}")
    print(f"  Project Root:     {Config.PROJECT_ROOT}")
    print(f"  BERT XLSX:        {Config.BERT_XLSX}")
    print(f"  Concept XLSX:     {Config.CONCEPT_FILE}")
    print(f"  Examples XLSX:    {Config.EXAMPLE_FILE}")
    print(f"  Output File:      {Config.OUTPUT_FILE}")

    for path, name in [(Config.BERT_XLSX, "BERT XLSX"),
                       (Config.CONCEPT_FILE, "Concept.xlsx"),
                       (Config.EXAMPLE_FILE, "RequirementExamples.xlsx")]:
        if not os.path.exists(path):
            print(f"\n  [ERROR] {name} not found: {path}")
            return
        print(f"  [OK] {name} found")

    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)

    print(f"\nLoading knowledge base...")
    kb = KnowledgeBaseLoader(
        concept_path=Config.CONCEPT_FILE,
        examples_path=Config.EXAMPLE_FILE,
        max_examples_per_class=Config.MAX_EXAMPLES_PER_CLASS,
    )

    print(f"\nLoading BERT test split...")
    bert_df = pd.read_excel(Config.BERT_XLSX)
    bert_df.columns = [c.strip() for c in bert_df.columns]
    print(f"Loaded {len(bert_df)} samples")

    print(f"\nInitializing model with unified System Prompt...")
    client = ModelClient(kb)
    print(f"   Model: {Config.MODEL_NAME}")
    print(f"   System Prompt length: {len(client.system_prompt)} chars")
    print(f"   Request interval: {Config.REQUEST_INTERVAL}s")
    print(f"   Max rounds: {Config.MAX_ROUNDS}")

    results = []
    total = len(bert_df)
    start_time = datetime.now()

    for i, (_, row) in enumerate(bert_df.iterrows(), 1):
        text = str(row['requirement'])
        gold = str(row['true_label']).strip().lower()
        sample_id = int(row['sample_id'])

        print(f"\n[{i}/{total}] sample_id={sample_id}")
        print(f"   Requirement: {text[:100]}{'...' if len(text) > 100 else ''}")
        print(f"   Ground Truth: {gold}")

        result = client.classify_with_retry(text, gold)

        is_correct = "Yes" if result["final_label"] == gold else "No"

        results.append({
            "sample_id": sample_id,
            "requirement": text,
            "Human_Label": gold,
            "Model_Prediction": result["final_label"],
            "prediction_confidence": round(result["final_confidence"], 4),
            "IsCorrect": is_correct,
            "round": result["total_rounds"],
            "round1_pred": result["round1_pred"],
            "round1_conf": round(result["round1_conf"], 4) if result["round1_conf"] is not None else None,
            "round1_correct": result["round1_correct"] if result["round1_correct"] else "N/A",
            "round2_pred": result["round2_pred"] if result["round2_pred"] else "N/A",
            "round2_conf": round(result["round2_conf"], 4) if result["round2_conf"] is not None else None,
            "round2_correct": result["round2_correct"] if result["round2_correct"] else "N/A",
            "round3_pred": result["round3_pred"] if result["round3_pred"] else "N/A",
            "round3_conf": round(result["round3_conf"], 4) if result["round3_conf"] is not None else None,
            "round3_correct": result["round3_correct"] if result["round3_correct"] else "N/A",
        })

        print(f"   ✅ Final: {result['final_label']} "
              f"(conf={result['final_confidence']:.4f}, "
              f"rounds={result['total_rounds']}, "
              f"{'correct' if is_correct == 'Yes' else 'incorrect'})")

        if i % 5 == 0:
            elapsed = datetime.now() - start_time
            avg = elapsed / i
            remain = (total - i) * avg
            print(f"\n   Progress: {i}/{total} - "
                  f"Elapsed: {str(elapsed).split('.')[0]} - "
                  f"Remaining: {str(remain).split('.')[0]}")

        if i < total:
            time.sleep(Config.REQUEST_INTERVAL)

    print("\n" + "=" * 80)
    print("Saving results...")
    print("=" * 80)

    df = pd.DataFrame(results)
    df.to_excel(Config.OUTPUT_FILE, index=False)
    print(f"✅ Results saved to: {Config.OUTPUT_FILE}")

    correct_count = (df["IsCorrect"] == "Yes").sum()
    accuracy = correct_count / len(df) * 100
    print(f"\n📊 Classification Statistics:")
    print(f"   Total samples: {len(df)}")
    print(f"   Overall accuracy: {correct_count}/{len(df)} = {accuracy:.2f}%")

    print(f"\n   Round distribution:")
    for r, c in df['round'].value_counts().sort_index().items():
        print(f"      Round {r}: {c} ({c / len(df) * 100:.1f}%)")

    print(f"\n   Accuracy by round:")
    for r in sorted(df['round'].unique()):
        sub = df[df['round'] == r]
        sub_c = (sub["IsCorrect"] == "Yes").sum()
        print(f"      Round {r}: {sub_c}/{len(sub)} = {sub_c / len(sub) * 100:.2f}%")

    print(f"\n   Prediction distribution:")
    for label, count in df['Model_Prediction'].value_counts().items():
        print(f"      - {label}: {count}")

    print("\n" + "=" * 80)
    print("✅ Completed")
    print(f"   Output: {Config.OUTPUT_FILE}")
    print("=" * 80)


if __name__ == "__main__":
    main()