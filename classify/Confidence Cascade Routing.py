import os
import re
import time
import json
import random
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

LABELS = ('composite', 'interactive', 'sequential', 'structural', 'conditional')
MODEL_CONFIGS = [
    dict(alias='gpt', model='gpt-4.1-nano', env='OPENAI_API_KEY', url='https://api.openai-proxy.org/v1'),
    dict(alias='doubao', model='doubao-seed-1-6-250615', env='DOUBAO_API_KEY', url=None),
    dict(alias='qwen', model='qwen-plus-2025-07-28', env='DASHSCOPE_API_KEY', url='https://dashscope.aliyuncs.com/compatible-mode/v1'),
    dict(alias='hunyuan', model='hunyuan-turbos-latest', env='HUNYUAN_API_KEY', url='https://api.hunyuan.cloud.tencent.com/v1'),
    dict(alias='deepseek', model='deepseek-chat', env='DEEPSEEK_API_KEY', url='https://api.deepseek.com/v1'),
    dict(alias='kimi', model='kimi-k2-turbo-preview', env='MOONSHOT_API_KEY', url='https://api.moonshot.cn/v1'),
    dict(alias='glm', model='glm-4.5', env='GLM_API_KEY', url='https://open.bigmodel.cn/api/paas/v4/'),
]
MODEL_NAMES = {
    'gpt': 'GPT-4.1 nano',
    'doubao': 'Doubao-seed-1.6-250615',
    'qwen': 'Qwen-Plus',
    'hunyuan': 'Hunyuan T1',
    'deepseek': 'DeepSeek-V3.1',
    'kimi': 'Kimi-K2-Instruct',
    'glm': 'GLM-5.2',
}
TEMPERATURE = 0.3
MAX_TOKENS = 300
MIN_INTERVAL_SECONDS = 3.5
MAX_RETRIES = 5


def load_knowledge(concept_file: Path, example_file: Path) -> str:
    concepts = pd.read_excel(concept_file)
    definitions = []
    for _, row in concepts.iterrows():
        label = str(row.iloc[0]).strip().lower()
        if label in LABELS:
            definitions.append(f'- {label}: {str(row.iloc[1]).strip()}')
    if {line.split(':')[0][2:] for line in definitions} != set(LABELS):
        raise ValueError('Concept.xlsx must define exactly the five required categories')
    examples = pd.read_excel(example_file, header=None)
    example_lines = []
    for i in range(1, len(examples)):
        label = str(examples.iloc[i, 0]).strip().lower()
        if label not in LABELS:
            continue
        entries = [str(x).strip() for x in examples.iloc[i, 1:] if pd.notna(x) and str(x).strip()]
        example_lines.append(f'【{label}】')
        example_lines.extend(f'示例{k}: {entry}' for k, entry in enumerate(entries, 1))
    return ('你是一位专业的软件需求分类专家。请根据类别定义分类。\n'
            '【类别定义】\n' + '\n'.join(definitions) + '\n'
            '【分类示例】\n' + '\n'.join(example_lines))


def build_user_prompt(requirement: str) -> str:
    return (f'请对以下软件需求进行分类：\n{requirement}\n\n'
            '请估计该需求属于五个类别的相对概率。'
            '仅输出一个 JSON 对象，键必须是 composite, interactive, sequential, structural, conditional，'
            '值必须是 0 到 1 的数值，五个值之和为 1。不要输出其他内容。')


def parse_probabilities(content: str) -> Dict[str, float]:
    content = content.strip()
    match = re.search(r'\{[\s\S]*\}', content)
    if not match:
        raise ValueError(f'No JSON probability object: {content[:160]}')
    raw = json.loads(match.group())
    if set(raw) != set(LABELS):
        raise ValueError(f'Expected exactly {LABELS}, received {list(raw)}')
    values = np.array([float(raw[k]) for k in LABELS], dtype=float)
    if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
        raise ValueError('Probabilities must be finite numbers in [0,1]')
    if not np.isclose(values.sum(), 1.0, atol=0.02):
        raise ValueError(f'Probability sum {values.sum():.4f} differs from 1')
    values = values / values.sum()
    return dict(zip(LABELS, map(float, values)))


def classify(client: OpenAI, model: str, system_prompt: str, requirement: str):
    messages = [{'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': build_user_prompt(requirement)}]
    for attempt in range(MAX_RETRIES):
        try:
            result = client.chat.completions.create(
                model=model, messages=messages, temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS)
            content = result.choices[0].message.content or ''
            probabilities = parse_probabilities(content)
            ordered = sorted(probabilities.items(), key=lambda kv: (-kv[1], LABELS.index(kv[0])))
            return {'label': ordered[0][0], 'confidence': ordered[0][1] - ordered[1][1],
                    'probabilities': probabilities}
        except Exception:
            if attempt == MAX_RETRIES - 1:
                raise
            time.sleep(2 ** attempt + random.uniform(0, 0.5))


def balanced_quotas(n: int, m: int):
    q, remainder = divmod(n, m)
    return [q + (1 if i < remainder else 0) for i in range(m)]


def ece(scores, correct, bins=10):
    scores = np.asarray(scores, float)
    correct = np.asarray(correct, float)
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for k in range(bins):
        lo, hi = edges[k], edges[k + 1]
        if k == bins - 1:
            mask = (scores >= lo) & (scores <= hi)
        else:
            mask = (scores >= lo) & (scores < hi)
        if mask.any():
            total += mask.mean() * abs(scores[mask].mean() - correct[mask].mean())
    return float(total)


def run(dataset: pd.DataFrame, system_prompt: str, clients: Dict[str, OpenAI], output_dir: Path):
    n, m = len(dataset), len(MODEL_CONFIGS)
    quotas = balanced_quotas(n, m)
    active = list(range(n))
    records = {i: [] for i in range(n)}
    stage_stats = []
    last_request = {c['alias']: 0.0 for c in MODEL_CONFIGS}
    output_dir.mkdir(parents=True, exist_ok=True)

    for stage, cfg in enumerate(MODEL_CONFIGS):
        alias = cfg['alias']
        if not active:
            break
        batch = []
        for i in active:
            delay = MIN_INTERVAL_SECONDS - (time.monotonic() - last_request[alias])
            if delay > 0:
                time.sleep(delay)
            last_request[alias] = time.monotonic()
            prediction = classify(clients[alias], cfg['model'], system_prompt,
                                  str(dataset.iloc[i]['requirement']))
            batch.append((i, prediction))

        high_to_low = sorted(batch, key=lambda x: (-x[1]['confidence'], x[0]))
        confidence_series = pd.Series({i: p['confidence'] for i, p in batch})
        normalized_ranks = confidence_series.rank(method='average', ascending=True) / len(batch)
        for i, pred in batch:
            records[i].append({'stage': stage + 1, 'model': alias, **pred,
                               'rank': float(normalized_ranks.loc[i] * len(batch)),
                               'normalized_rank': float(normalized_ranks.loc[i])})

        k = quotas[stage] if stage < m - 1 else len(active)
        k = min(k, len(active))
        retained = [i for i, _ in high_to_low[:k]]
        active = [i for i, _ in high_to_low[k:]]

        processed = len(batch)
        processed_correct = sum(1 for i, p in batch
                                if p['label'] == dataset.iloc[i]['true_label'])
        retained_correct = sum(1 for i in retained
                               if dict(batch)[i]['label'] == dataset.iloc[i]['true_label'])
        mean_conf = float(np.mean([p['confidence'] for _, p in batch]))

        stage_stats.append({
            'Link': stage + 1,
            'Model': alias,
            'Model_Name': MODEL_NAMES[alias],
            'Samples_Processed': processed,
            'Processed_Correct': processed_correct,
            'Processed_Accuracy': processed_correct / processed,
            'Mean_Confidence': mean_conf,
            'Retained': len(retained),
            'Retained_Correct': retained_correct,
            'Retained_Accuracy': retained_correct / len(retained) if retained else 0.0,
            'Forwarded': len(active),
        })
        print(f'{stage + 1}/{m} {alias}: processed={processed}, retained={len(retained)}, forwarded={len(active)}', flush=True)

    if any(not records[i] for i in range(n)):
        raise RuntimeError('Some samples have no model predictions')

    rows = []
    for i in range(n):
        selected = max(records[i], key=lambda p: (p['normalized_rank'], -p['stage']))
        sid = dataset.iloc[i]['sample_id']
        try:
            sid_out = int(sid)
        except (ValueError, TypeError):
            sid_out = sid
        rows.append({
            'Sample_Index': sid_out,
            'Requirement_Content': dataset.iloc[i]['requirement'],
            'Human_Label': dataset.iloc[i]['true_label'],
            'Final_Predicted_Label': selected['label'],
            'Final_Confidence': selected['confidence'],
            'Final_Model': selected['model'],
            'Final_Model_Name': MODEL_NAMES[selected['model']],
            'Final_Link': selected['stage'],
            'Rank': selected['rank'],
            'Normalized_Rank': selected['normalized_rank'],
            'IsCorrect': 1 if selected['label'] == dataset.iloc[i]['true_label'] else 0,
        })

    simplified = pd.DataFrame(rows)
    stages_df = pd.DataFrame(stage_stats)

    conf = simplified['Final_Confidence'].to_numpy(float)
    correct = simplified['IsCorrect'].to_numpy(float)
    accuracy = float(correct.mean())
    avg_conf = float(conf.mean())
    ece_val = ece(conf, correct)
    pearson = (float(np.corrcoef(conf, correct)[0, 1])
               if np.std(conf) > 0 and np.std(correct) > 0 else float('nan'))

    metrics = pd.DataFrame([{
        'Test_Set_Size': len(dataset),
        'Accuracy': accuracy,
        'Average_Confidence': avg_conf,
        'ECE': ece_val,
        'Pearson': pearson,
    }])

    simplified.to_excel(output_dir / 'Chain_Ensemble_Simplified.xlsx', index=False)
    stages_df.to_excel(output_dir / 'Chain_Ensemble_Link_Statistics.xlsx', index=False)
    metrics.to_excel(output_dir / 'Chain_Ensemble_Metrics.xlsx', index=False)

    print(f'Accuracy: {accuracy:.2%} ({int(correct.sum())}/{n})')
    return simplified


def main():
    SCRIPT_DIR = Path(__file__).resolve().parent
    root = SCRIPT_DIR.parent
    data_dir = root / 'data'

    concept_file = data_dir / 'Concept.xlsx'
    example_file = data_dir / 'RequirementExamples.xlsx'
    if not concept_file.exists():
        raise FileNotFoundError(f'找不到 {concept_file}')
    if not example_file.exists():
        raise FileNotFoundError(f'找不到 {example_file}')

    bert_path = root / 'classifyOutcome' / 'BERT_output' / 'BERT' / 'bert_test_predictions.csv'
    if not bert_path.exists():
        bert_path = root / 'bert_test_predictions.xlsx'
    if not bert_path.exists():
        raise FileNotFoundError(
            f'找不到测试集，尝试过:\n'
            f'  {root / "classifyOutcome" / "BERT_output" / "BERT" / "bert_test_predictions.csv"}\n'
            f'  {root / "bert_test_predictions.xlsx"}'
        )

    if bert_path.suffix == '.csv':
        dataset = pd.read_csv(bert_path, encoding='utf-8-sig')
    else:
        dataset = pd.read_excel(bert_path)

    required = {'sample_id', 'requirement', 'true_label'}
    if not required.issubset(dataset.columns):
        raise ValueError(f'Test data missing columns: {required - set(dataset.columns)}')

    dataset['true_label'] = dataset['true_label'].astype(str).str.lower().str.strip()
    if not dataset['true_label'].isin(LABELS).all():
        raise ValueError('Test set contains labels outside the five-class taxonomy')
    if dataset['sample_id'].duplicated().any():
        raise ValueError('Test set contains duplicate sample_id values')

    system_prompt = load_knowledge(concept_file, example_file)

    clients = {}
    for cfg in MODEL_CONFIGS:
        key = os.getenv(cfg['env'])
        if not key:
            raise EnvironmentError(f'Missing {cfg["env"]} for {cfg["alias"]}')
        clients[cfg['alias']] = OpenAI(api_key=key, base_url=cfg['url'])

    output_dir = root / 'classifyOutcome' / 'Confidence Cascade Routing'
    run(dataset, system_prompt, clients, output_dir)


if __name__ == '__main__':
    main()