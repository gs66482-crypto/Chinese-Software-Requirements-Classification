import os
import time
import random
import pandas as pd
from typing import List, Dict, Tuple, Optional
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

class Config:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
    DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
    CONCEPT_FILE = os.path.join(DATA_DIR, 'Concept.xlsx')
    EXAMPLE_FILE = os.path.join(DATA_DIR, 'RequirementExamples.xlsx')
    BERT_XLSX = os.path.join(PROJECT_ROOT, 'classifyOutcome', 'BERT_output', 'BERT', 'bert_test_predictions.xlsx')
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'classifyOutcome', 'Single-model One-shot Inference')
    OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'Single-Pass Classification-glm.xlsx')
    MODEL_NAME = 'glm-4.5'
    API_KEY_ENV = 'DASHSCOPE_API_KEY'
    BASE_URL = 'https://dashscope.aliyuncs.com/compatible-mode/v1'
    RPM = 20
    MIN_INTERVAL = 60.0 / RPM
    MAX_RETRIES = 5
    BASE_BACKOFF = 2.0
    TEMPERATURE = 0.3
    MAX_TOKENS = 300
    SEED = 42
    MAX_EXAMPLES_PER_CLASS = None
    REQUEST_INTERVAL = 1.0

class KnowledgeBaseLoader:

    def __init__(self, concept_path: str, examples_path: str, max_examples_per_class: Optional[int]=None):
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
            if cat and cat != 'nan':
                self.category_definitions[cat] = exp
        print(f'[KB] Loaded {len(self.category_definitions)} category definitions')

    def _load_examples(self):
        df = pd.read_excel(self.examples_path, sheet_name=0, header=None)
        for i in range(1, len(df)):
            cat = str(df.iloc[i, 0]).strip().lower()
            if not cat or cat == 'nan':
                continue
            examples = []
            for j in range(1, df.shape[1]):
                val = df.iloc[i, j]
                if pd.isna(val):
                    continue
                text = str(val).strip()
                if text and text.lower() != 'nan':
                    examples.append(text)
            if self.max_examples_per_class is None:
                self.few_shot_examples[cat] = examples
            else:
                self.few_shot_examples[cat] = examples[:self.max_examples_per_class]
        total = sum((len(v) for v in self.few_shot_examples.values()))
        print(f'[KB] Loaded {total} few-shot examples across {len(self.few_shot_examples)} categories')

    def get_category_list_text(self) -> str:
        lines = []
        for cat, exp in self.category_definitions.items():
            lines.append(f'- {cat}: {exp}')
        return '\n'.join(lines)

    def get_few_shot_text(self) -> str:
        lines = []
        for cat, examples in self.few_shot_examples.items():
            if not examples:
                continue
            lines.append(f'【{cat}】')
            for k, ex in enumerate(examples, 1):
                lines.append(f'  示例{k}: {ex}')
        return '\n'.join(lines)

def build_system_prompt(kb: KnowledgeBaseLoader) -> str:
    base_role = '你是一位专业的软件需求分类专家。请根据给定的类别定义对需求进行分类。'
    category_text = kb.get_category_list_text()
    few_shot_text = kb.get_few_shot_text()
    return f'{base_role}\n\n【类别定义】（来源：Concept.xlsx）\n{category_text}\n\n【分类示例】（来源：RequirementExamples.xlsx）\n{few_shot_text}'

class PromptBuilder:

    @staticmethod
    def build_user_prompt(text: str) -> str:
        return f'请对以下软件需求进行分类。\n\n## 待分类需求：\n{text}\n\n## 输出格式：\n分类: [标签名称]\n置信度: [0-1之间的数值]\n\n请严格按照上述格式输出。'

class RateLimiter:

    def __init__(self, rpm: int=20, min_interval: float=3.0):
        self.min_interval = max(min_interval, 60.0 / rpm)
        self.last_request_time = 0

    def wait_if_needed(self):
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self.last_request_time = time.time()

class ModelClient:

    def __init__(self, kb: KnowledgeBaseLoader):
        self.model_name = Config.MODEL_NAME
        self.api_key = os.getenv(Config.API_KEY_ENV)
        self.base_url = Config.BASE_URL
        if not self.api_key:
            raise ValueError(f'API key not found: {Config.API_KEY_ENV}')
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        self.rate_limiter = RateLimiter(Config.RPM, Config.MIN_INTERVAL)
        self.max_retries = Config.MAX_RETRIES
        self.base_backoff = Config.BASE_BACKOFF
        self.system_prompt = build_system_prompt(kb)

    def _call_model_with_retry(self, user_prompt: str) -> str:
        for attempt in range(self.max_retries):
            try:
                self.rate_limiter.wait_if_needed()
                messages = [{'role': 'system', 'content': self.system_prompt}, {'role': 'user', 'content': user_prompt}]
                response = self.client.chat.completions.create(model=self.model_name, messages=messages, temperature=Config.TEMPERATURE, max_tokens=Config.MAX_TOKENS)
                return response.choices[0].message.content.strip()
            except Exception as e:
                error_msg = str(e).lower()
                if 'rate' in error_msg or '429' in error_msg or 'throttle' in error_msg:
                    wait_time = self.base_backoff * 2 ** attempt + random.uniform(0, 0.5)
                    print(f'   ⚠️ Rate limit, retry in {wait_time:.1f}s ({attempt + 1}/{self.max_retries})')
                    time.sleep(wait_time)
                else:
                    print(f'   ❌ API error: {e}')
                    if attempt < self.max_retries - 1:
                        wait_time = self.base_backoff * 2 ** attempt
                        print(f'   🔄 Retry in {wait_time:.1f}s ({attempt + 1}/{self.max_retries})')
                        time.sleep(wait_time)
                    else:
                        raise
        raise Exception(f'Max retries exceeded ({self.max_retries})')

    def classify(self, text: str) -> Tuple[str, float]:
        user_prompt = PromptBuilder.build_user_prompt(text)
        try:
            response = self._call_model_with_retry(user_prompt)
            return self._parse_response(response)
        except Exception as e:
            print(f'   ❌ Classification failed: {e}')
            return ('Unknown', 0.0)

    def _parse_response(self, response: str) -> Tuple[str, float]:
        lines = response.strip().split('\n')
        label = ''
        confidence = 0.0
        for line in lines:
            line = line.strip()
            if line.startswith('分类:') or line.startswith('分类：'):
                label = line.split(':', 1)[-1].strip()
                if label.startswith('：'):
                    label = label[1:].strip()
            elif line.startswith('置信度:') or line.startswith('置信度：'):
                try:
                    c = line.split(':', 1)[-1].strip()
                    if c.startswith('：'):
                        c = c[1:].strip()
                    confidence = max(0.0, min(1.0, float(c)))
                except Exception:
                    confidence = 0.0
        if not label:
            label = 'Unknown'
        return (label, confidence)

def main():
    print('=' * 80)
    print('Single-Pass Classification with GLM')
    print(f'   Model: {Config.MODEL_NAME}')
    print('=' * 80)
    print(f'\nPath Information:')
    print(f'  Script Directory: {Config.SCRIPT_DIR}')
    print(f'  Project Root:     {Config.PROJECT_ROOT}')
    print(f'  BERT XLSX:         {Config.BERT_XLSX}')
    print(f'  Concept XLSX:     {Config.CONCEPT_FILE}')
    print(f'  Examples XLSX:    {Config.EXAMPLE_FILE}')
    print(f'  Output File:      {Config.OUTPUT_FILE}')
    for path, name in [(Config.BERT_XLSX, 'BERT XLSX'), (Config.CONCEPT_FILE, 'Concept.xlsx'), (Config.EXAMPLE_FILE, 'RequirementExamples.xlsx')]:
        if not os.path.exists(path):
            print(f'\n  [ERROR] {name} not found: {path}')
            return
        print(f'  [OK] {name} found')
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    print(f'\nLoading knowledge base...')
    kb = KnowledgeBaseLoader(concept_path=Config.CONCEPT_FILE, examples_path=Config.EXAMPLE_FILE, max_examples_per_class=Config.MAX_EXAMPLES_PER_CLASS)
    print(f'\nLoading BERT test split...')
    bert_df = pd.read_excel(Config.BERT_XLSX, sheet_name=0)
    bert_df.columns = [c.strip() for c in bert_df.columns]
    print(f'Loaded {len(bert_df)} samples')
    print(f'\nInitializing model...')
    client = ModelClient(kb)
    print(f'   Model: {Config.MODEL_NAME}')
    print(f'   System Prompt length: {len(client.system_prompt)} chars')
    print(f'   Rate limit: {Config.RPM} RPM')
    print('\n' + '=' * 80)
    print('Starting classification...')
    print('=' * 80)
    results = []
    total = len(bert_df)
    start_time = datetime.now()
    for i, (_, row) in enumerate(bert_df.iterrows(), 1):
        text = str(row['requirement'])
        gold = str(row['true_label']).strip().lower()
        sample_id = int(row['sample_id'])
        print(f'\n[{i}/{total}] sample_id={sample_id}')
        print(f"   Requirement: {text[:100]}{('...' if len(text) > 100 else '')}")
        label, confidence = client.classify(text)
        is_correct = 'Yes' if label == gold else 'No'
        results.append({'sample_id': sample_id, 'requirement': text, 'Human_Label': gold, 'Model_Prediction': label, 'prediction_confidence': round(confidence, 4), 'IsCorrect': is_correct})
        print(f'   Prediction: {label} ({confidence:.4f}) | gold={gold} | {is_correct}')
        if i % 5 == 0:
            elapsed = datetime.now() - start_time
            avg = elapsed / i
            remain = (total - i) * avg
            print(f"\n   Progress: {i}/{total} - Elapsed: {str(elapsed).split('.')[0]} - Remaining: {str(remain).split('.')[0]}")
        if i < total:
            time.sleep(Config.REQUEST_INTERVAL)
    print('\n' + '=' * 80)
    print('Saving results...')
    print('=' * 80)
    df = pd.DataFrame(results)
    df.to_excel(Config.OUTPUT_FILE, index=False)
    print(f'✅ Results saved to: {Config.OUTPUT_FILE}')
    correct_count = (df['IsCorrect'] == 'Yes').sum()
    accuracy = correct_count / len(df) * 100
    print(f'\n📊 Classification Statistics:')
    print(f'   Total samples: {len(df)}')
    print(f'   Correct predictions: {correct_count}')
    print(f'   Accuracy: {accuracy:.2f}%')
    print(f'\n   Prediction distribution:')
    for label, count in df['Model_Prediction'].value_counts().items():
        print(f'      - {label}: {count}')
    print('\n' + '=' * 80)
    print('✅ Classification completed successfully!')
    print(f'   Output file: {Config.OUTPUT_FILE}')
    print('=' * 80)
if __name__ == '__main__':
    main()
