import os
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
from dotenv import load_dotenv
from openai import OpenAI
import time
import re
import datetime
from collections import defaultdict

load_dotenv()




class Config:
    """Shared configuration."""


    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
    DATA_DIR = os.path.join(PROJECT_ROOT, "data")

    BERT_CSV = os.path.join(
        PROJECT_ROOT, "classifyOutcome", "BERT_output", "BERT",
        "bert_test_predictions.csv"
    )
    CONCEPT_FILE = os.path.join(DATA_DIR, "Concept.xlsx")
    EXAMPLE_FILE = os.path.join(DATA_DIR, "RequirementExamples.xlsx")

    OUTPUT_DIR = os.path.join(PROJECT_ROOT, "classifyOutcome")
    OUTPUT_FILE = os.path.join(
        OUTPUT_DIR,
        "Hierarchical Binary Decision Routing.xlsx"
    )


    MODEL_NAME = "doubao-seed-2-0-mini-260428"
    API_KEY_ENV = "DOUBAO_API_KEY"
    BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"


    TEMPERATURE = 0.3
    MAX_TOKENS = 300
    SEED = 42


    MAX_EXAMPLES_PER_CLASS = None


    REQUEST_INTERVAL = 1.0


    CSV_ENCODINGS = ["utf-8-sig", "utf-8", "gbk", "gb18030", "cp936", "latin1"]




class ClassificationStep(Enum):
    """决策步骤枚举（顺序与最新定义对齐）"""
    COMPOSITE_MARKER_CHECK = "composite_marker_check"
    INTERACTIVE_CHECK      = "interactive_check"
    SEQUENTIAL_CHECK       = "sequential_check"
    CONDITIONAL_CHECK      = "conditional_check"
    FINAL_CLASSIFICATION   = "final_classification"


class RequirementType(Enum):
    COMPOSITE = "composite"
    CONDITIONAL = "conditional"
    INTERACTIVE = "interactive"
    SEQUENTIAL = "sequential"
    STRUCTURAL = "structural"
    UNKNOWN = "unknown"


class ActionType(Enum):
    CHECK_COMPOSITE_MARKER = "check_composite_marker"
    CHECK_INTERACTIVE      = "check_interactive"
    CHECK_SEQUENTIAL       = "check_sequential"
    CHECK_CONDITIONAL      = "check_conditional"
    MAKE_FINAL_CLASSIFICATION = "make_final_classification"




@dataclass
class ClassificationState:
    step: ClassificationStep
    requirement_text: str
    intermediate_results: Dict[str, Any] = field(default_factory=dict)
    history: List[Dict] = field(default_factory=list)
    is_complete: bool = False
    final_classification: Optional[RequirementType] = None


@dataclass
class StepDecision:
    step: ClassificationStep
    action: ActionType
    decision: bool
    confidence: float
    reasoning: str
    evidence: List[str]
    timestamp: str


@dataclass
class ClassificationEpisode:
    requirement_id: str
    requirement_text: str
    steps: List[StepDecision] = field(default_factory=list)
    final_classification: Optional[RequirementType] = None
    ground_truth: Optional[RequirementType] = None
    total_reward: float = 0.0
    step_rewards: List[float] = field(default_factory=list)
    efficiency_metrics: Dict[str, float] = field(default_factory=dict)
    sample_id: Optional[int] = None




class KnowledgeBaseLoader:
    """
    加载 Concept.xlsx（类别定义）和 RequirementExamples.xlsx（少样本示例）。
    提供按类别取定义 / 取示例的方法，供提示词动态插入。
    """

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


    def get_definition(self, category: str) -> str:
        """按类别名取定义文本；找不到返回空串。"""
        return self.category_definitions.get(category.strip().lower(), "")

    def get_examples(self, category: str) -> List[str]:
        """按类别名取示例列表；找不到返回空列表。"""
        return self.few_shot_examples.get(category.strip().lower(), [])


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




class DoubaoLLMClient:
    """
    Doubao LLM Client with unified System Prompt.
    System Prompt: Role + Category definitions + Few-shot
    User Prompt:   Step-specific decision node logic（定义从 Concept 动态取）
    """

    def __init__(self, model_name: str = Config.MODEL_NAME,
                 seed: int = Config.SEED,
                 knowledge_base: Optional[KnowledgeBaseLoader] = None):
        self.model_name = model_name
        self.seed = seed
        self.knowledge_base = knowledge_base

        api_key = os.getenv(Config.API_KEY_ENV)
        if not api_key:
            raise ValueError(f"{Config.API_KEY_ENV} environment variable not set")

        self.client = OpenAI(api_key=api_key, base_url=Config.BASE_URL)

        self.system_prompt = self._build_system_prompt()

        self.inference_params = {
            "temperature": Config.TEMPERATURE,
            "max_tokens": Config.MAX_TOKENS,
            "seed": seed,
            "stream": False,
            "top_p": 1.0,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
        }

    def _build_system_prompt(self) -> str:
        base_role = (
            "你是一位专业的软件需求分类专家。"
            "请根据给定的类别定义对需求进行分类。"
        )

        if self.knowledge_base is None:
            return base_role

        category_text = self.knowledge_base.get_category_list_text()
        few_shot_text = self.knowledge_base.get_few_shot_text()

        return f"""{base_role}

【类别定义】（来源：Concept.xlsx）
{category_text}

【分类示例】（来源：RequirementExamples.xlsx）
{few_shot_text}"""

    def query_decision(self, user_prompt: str,
                       step: ClassificationStep) -> str:
        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self.inference_params["temperature"],
                max_tokens=self.inference_params["max_tokens"],
                seed=self.inference_params["seed"],
                stream=self.inference_params["stream"],
                top_p=self.inference_params["top_p"],
                frequency_penalty=self.inference_params["frequency_penalty"],
                presence_penalty=self.inference_params["presence_penalty"],
            )
            return completion.choices[0].message.content
        except Exception as e:
            print(f"LLM query failed: {e}")
            return """【分析】技术问题导致无法完成分析。
【决策】false
【置信度】0.5
【证据】无"""




class RewardCalculator:
    def __init__(self, config=None):
        self.config = config or self._default_config()

    def _default_config(self):
        return {
            "step_rewards": {"evidence_found": 0.05},
            "final_rewards": {
                "correct_classification": 2.0,
                "incorrect_classification": -1.5,
                "early_correct": 0.5,
                "optimal_path": 0.3,
            },
            "thresholds": {
                "max_steps": 5,
                "optimal_step_counts": {
                    "composite": 1, "interactive": 2, "sequential": 3,
                    "conditional": 4, "structural": 4,
                },
            },
        }

    def calculate_episode_reward(self, episode):
        step_rewards = [self._calc_step(s, i, episode)
                        for i, s in enumerate(episode.steps)]
        final_reward = self._calc_final(episode)
        total = sum(step_rewards) + final_reward
        return self._normalize(total, episode), step_rewards

    def _calc_step(self, step, step_index, episode):
        reward = 0.0
        if 0.4 <= step.confidence <= 0.7:
            reward += 0.1
        elif 0.7 < step.confidence <= 0.9:
            reward += 0.05
        elif step.confidence > 0.9:
            reward -= 0.1
        else:
            reward -= 0.05
        if step.evidence:
            reward += self.config["step_rewards"]["evidence_found"]
        if len(step.reasoning) > 50:
            reward += 0.05
        return reward

    def _calc_final(self, episode):
        if episode.ground_truth is None:
            return 0.0
        reward = 0.0
        if episode.final_classification == episode.ground_truth:
            reward += self.config["final_rewards"]["correct_classification"]
            optimal = self.config["thresholds"]["optimal_step_counts"].get(
                episode.ground_truth.value, 3)
            if len(episode.steps) <= optimal:
                reward += self.config["final_rewards"]["optimal_path"]
            if len(episode.steps) < optimal:
                reward += self.config["final_rewards"]["early_correct"]
        else:
            reward += self.config["final_rewards"]["incorrect_classification"]
        if len(episode.steps) > self.config["thresholds"]["max_steps"]:
            reward -= 0.5
        return reward

    def _normalize(self, reward, episode):
        max_possible = (len(episode.steps) * 0.5 +
                        self.config["final_rewards"]["correct_classification"] +
                        self.config["final_rewards"]["optimal_path"] +
                        self.config["final_rewards"]["early_correct"])
        min_possible = (len(episode.steps) * -0.5 +
                        self.config["final_rewards"]["incorrect_classification"] - 0.5)
        if max_possible > min_possible:
            normalized = -1 + 3 * (reward - min_possible) / (max_possible - min_possible)
            return max(-1.0, min(2.0, normalized))
        return max(-1.0, min(2.0, reward))




class ClassificationAgent:
    def __init__(self, llm_client, reward_calculator=None,
                 knowledge_base: Optional[KnowledgeBaseLoader] = None):
        self.llm_client = llm_client
        self.reward_calculator = reward_calculator or RewardCalculator()
        self.knowledge_base = knowledge_base
        self.state: Optional[ClassificationState] = None
        self.current_episode: Optional[ClassificationEpisode] = None


        self.decision_tree = {
            ClassificationStep.COMPOSITE_MARKER_CHECK: {
                "decision_true": {
                    "classification": RequirementType.COMPOSITE,
                    "is_terminal": True,
                },
                "decision_false": {
                    "next_step": ClassificationStep.INTERACTIVE_CHECK,
                    "action": ActionType.CHECK_INTERACTIVE,
                },
            },
            ClassificationStep.INTERACTIVE_CHECK: {
                "decision_true": {
                    "classification": RequirementType.INTERACTIVE,
                    "is_terminal": True,
                },
                "decision_false": {
                    "next_step": ClassificationStep.SEQUENTIAL_CHECK,
                    "action": ActionType.CHECK_SEQUENTIAL,
                },
            },
            ClassificationStep.SEQUENTIAL_CHECK: {
                "decision_true": {
                    "classification": RequirementType.SEQUENTIAL,
                    "is_terminal": True,
                },
                "decision_false": {
                    "next_step": ClassificationStep.CONDITIONAL_CHECK,
                    "action": ActionType.CHECK_CONDITIONAL,
                },
            },
            ClassificationStep.CONDITIONAL_CHECK: {
                "decision_true": {
                    "classification": RequirementType.CONDITIONAL,
                    "is_terminal": True,
                },
                "decision_false": {
                    "classification": RequirementType.STRUCTURAL,
                    "is_terminal": True,
                },
            },
        }

    def reset(self, requirement_text, requirement_id=None, sample_id=None):
        if requirement_id is None:
            requirement_id = (f"req_"
                              f"{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}_"
                              f"{hash(requirement_text) % 10000}")
        self.state = ClassificationState(
            step=ClassificationStep.COMPOSITE_MARKER_CHECK,
            requirement_text=requirement_text,
            intermediate_results={}, history=[]
        )
        self.current_episode = ClassificationEpisode(
            requirement_id=requirement_id,
            requirement_text=requirement_text,
            steps=[],
            final_classification=RequirementType.UNKNOWN,
            sample_id=sample_id
        )
        return self.state

    def step(self):
        if self.state.is_complete:
            return None, self.state, True
        current_config = self.decision_tree.get(self.state.step, {})
        if not current_config:
            raise ValueError(f"Unknown step: {self.state.step}")

        step_decision = self._execute_step_decision(self.state.step)
        self.state.history.append({
            "step": step_decision.step.value,
            "decision": step_decision.decision,
            "confidence": step_decision.confidence,
            "evidence": step_decision.evidence,
        })
        self.current_episode.steps.append(step_decision)
        is_terminal = self._process_decision_result(step_decision)

        if is_terminal:
            self.state.is_complete = True
            self.current_episode.final_classification = self.state.final_classification
            if self.reward_calculator:
                total, sr = self.reward_calculator.calculate_episode_reward(
                    self.current_episode)
                self.current_episode.total_reward = total
                self.current_episode.step_rewards = sr
        return step_decision, self.state, is_terminal

    def _execute_step_decision(self, step):
        user_prompt = self._build_user_prompt(step, self.state.requirement_text)
        response = self.llm_client.query_decision(user_prompt, step)
        decision, confidence, reasoning, evidence = self._parse_llm_response(
            response, step)

        action_map = {
            ClassificationStep.COMPOSITE_MARKER_CHECK: ActionType.CHECK_COMPOSITE_MARKER,
            ClassificationStep.INTERACTIVE_CHECK:      ActionType.CHECK_INTERACTIVE,
            ClassificationStep.SEQUENTIAL_CHECK:       ActionType.CHECK_SEQUENTIAL,
            ClassificationStep.CONDITIONAL_CHECK:      ActionType.CHECK_CONDITIONAL,
            ClassificationStep.FINAL_CLASSIFICATION:   ActionType.MAKE_FINAL_CLASSIFICATION,
        }
        return StepDecision(
            step=step,
            action=action_map.get(step, ActionType.CHECK_COMPOSITE_MARKER),
            decision=decision,
            confidence=confidence,
            reasoning=reasoning,
            evidence=evidence,
            timestamp=datetime.datetime.now().strftime("%H:%M:%S"),
        )

    def _process_decision_result(self, step_decision):
        current_config = self.decision_tree.get(self.state.step, {})
        decision_key = "decision_true" if step_decision.decision else "decision_false"
        if decision_key in current_config:
            next_config = current_config[decision_key]
            if next_config.get("is_terminal", False):
                self.state.final_classification = next_config["classification"]
                self.state.step = ClassificationStep.FINAL_CLASSIFICATION
                return True
            else:
                self.state.step = next_config["next_step"]
                return False
        else:
            if "next_step" in current_config:
                self.state.step = current_config["next_step"]
                return False
            else:
                self.state.is_complete = True
                return True




    def _get_def(self, category: str) -> str:
        if self.knowledge_base is None:
            return ""
        return self.knowledge_base.get_definition(category)

    def _build_user_prompt(self, step, requirement_text) -> str:
        base = f"""请分析以下需求描述：

需求描述："{requirement_text}"

"""


        if step == ClassificationStep.COMPOSITE_MARKER_CHECK:
            definition = self._get_def("composite")
            return base + f"""【当前决策节点：composite 检查】

请根据下方【当前类别定义】判断这个需求是否属于 "composite"。
请严格以该定义为唯一判断依据，不要自行添加或替换标准。

【当前类别定义】
{definition}

请分析：
1. 依据上述定义，逐一核对该需求是否符合其中描述的每一类特征；
2. 明确指出需求文本中支持你判断的关键词、短语或标点；
3. 给出最终是否属于 composite 的结论。

请按照以下格式回复：
【分析】你的分析过程
【决策】true/false（true表示属于composite，false表示不属于）
【置信度】0.0-1.0之间的数字
【证据】从需求文本中提取的支持你决策的关键词或短语，用逗号分隔"""


        elif step == ClassificationStep.INTERACTIVE_CHECK:
            definition = self._get_def("interactive")
            return base + f"""【当前决策节点：interactive 检查】

请根据下方【当前类别定义】判断这个需求是否属于 "interactive"。
请严格以该定义为唯一判断依据，不要自行添加或替换标准。

【当前类别定义】
{definition}

请分析：
1. 依据上述定义，逐一核对该需求是否符合其中描述的每一类特征；
2. 明确指出需求文本中支持你判断的关键词、短语或标点；
3. 给出最终是否属于 interactive 的结论。

请按照以下格式回复：
【分析】你的分析过程
【决策】true/false（true表示属于interactive，false表示不属于）
【置信度】0.0-1.0之间的数字
【证据】从需求文本中提取的支持你决策的关键词或短语，用逗号分隔"""


        elif step == ClassificationStep.SEQUENTIAL_CHECK:
            definition = self._get_def("sequential")
            return base + f"""【当前决策节点：sequential 检查】

请根据下方【当前类别定义】判断这个需求是否属于 "sequential"。
请严格以该定义为唯一判断依据，不要自行添加或替换标准。

【当前类别定义】
{definition}

请分析：
1. 依据上述定义，逐一核对该需求是否符合其中描述的每一类特征；
2. 明确指出需求文本中支持你判断的关键词、短语或标点；
3. 给出最终是否属于 sequential 的结论。

请按照以下格式回复：
【分析】你的分析过程
【决策】true/false（true表示属于sequential，false表示不属于）
【置信度】0.0-1.0之间的数字
【证据】从需求文本中提取的支持你决策的关键词或短语，用逗号分隔"""


        elif step == ClassificationStep.CONDITIONAL_CHECK:
            definition = self._get_def("conditional")
            return base + f"""【当前决策节点：conditional 检查】

请根据下方【当前类别定义】判断这个需求是否属于 "conditional"。
请严格以该定义为唯一判断依据，不要自行添加或替换标准。

【当前类别定义】
{definition}

请分析：
1. 依据上述定义，逐一核对该需求是否符合其中描述的每一类特征；
2. 明确指出需求文本中支持你判断的关键词、短语或标点；
3. 给出最终是否属于 conditional 的结论。

请按照以下格式回复：
【分析】你的分析过程
【决策】true/false（true表示属于conditional，false表示不属于）
【置信度】0.0-1.0之间的数字
【证据】从需求文本中提取的支持你决策的关键词或短语，用逗号分隔"""

        else:
            return base + "请对这个需求进行分类。"

    def _parse_llm_response(self, response, step):
        try:
            analysis_match = re.search(r'【分析】\s*([^【】]+)', response, re.DOTALL)
            reasoning = analysis_match.group(1).strip() if analysis_match else ""

            if step == ClassificationStep.FINAL_CLASSIFICATION:
                decision = True
            else:
                decision_match = re.search(r'【决策】\s*(true|false)',
                                           response, re.IGNORECASE)
                decision = (decision_match.group(1).lower() == "true"
                            if decision_match else False)

            confidence_match = re.search(r'【置信度】\s*([0-9]*\.?[0-9]+)',
                                         response)
            if confidence_match:
                confidence = float(confidence_match.group(1))
                confidence = max(0.0, min(1.0, confidence))
            else:
                confidence = 0.5

            evidence_match = re.search(r'【证据】\s*([^【】]+)', response)
            if evidence_match:
                evidence = [e.strip()
                            for e in evidence_match.group(1).split(',')
                            if e.strip()]
            else:
                evidence = []

            return decision, confidence, reasoning, evidence
        except Exception as e:
            print(f"Failed to parse LLM response: {e}")
            return False, 0.5, f"Parse failed: {str(e)}", []




class SequentialClassificationProcessor:
    def __init__(self, llm_client=None, agent=None):
        self.llm_client = llm_client or DoubaoLLMClient()
        self.agent = agent or ClassificationAgent(
            self.llm_client, knowledge_base=getattr(llm_client, "knowledge_base", None))
        self.results = []
        self.statistics = defaultdict(list)

    def classify_requirement(self, requirement_text, requirement_id=None,
                             ground_truth=None, sample_id=None):
        print(f"\n{'=' * 60}")
        print(f"[sample_id={sample_id}] Classifying: {requirement_text[:80]}...")
        if ground_truth:
            print(f"Ground Truth: {ground_truth.value}")
        print(f"{'=' * 60}")

        state = self.agent.reset(requirement_text, requirement_id, sample_id)
        step_count = 0
        is_complete = False
        while not is_complete and step_count < 10:
            step_count += 1
            print(f"\nStep {step_count}: {state.step.value}")
            step_decision, new_state, is_complete = self.agent.step()
            if step_decision:
                print(f"  Decision: "
                      f"{'PASS' if step_decision.decision else 'FAIL'}")
                print(f"  Confidence: {step_decision.confidence:.3f}")
                print(f"  Evidence: "
                      f"{', '.join(step_decision.evidence[:3])}"
                      f"{'...' if len(step_decision.evidence) > 3 else ''}")
            state = new_state
            if is_complete:
                print(f"\nClassification Complete!")
                print(f"  Final: {state.final_classification.value}")
                print(f"  Total Steps: {step_count}")
                break

        episode = self.agent.current_episode
        episode.ground_truth = ground_truth
        if self.agent.reward_calculator:
            total, sr = self.agent.reward_calculator.calculate_episode_reward(
                episode)
            episode.total_reward = total
            episode.step_rewards = sr
            print(f"  Total Reward: {total:.3f}")

        self._record_statistics(episode)
        return episode

    def _record_statistics(self, episode):
        self.results.append(episode)
        if episode.ground_truth:
            is_correct = episode.final_classification == episode.ground_truth
            self.statistics["accuracy"].append(1 if is_correct else 0)
        self.statistics["steps"].append(len(episode.steps))
        self.statistics["rewards"].append(episode.total_reward)
        self.statistics["confidences"].append(
            np.mean([s.confidence for s in episode.steps])
            if episode.steps else 0)

    def process_dataset(self, dataset):
        episodes = []
        print(f"\nProcessing {len(dataset)} requirements (BERT test split)")
        print(f"{'=' * 60}")
        for i, item in enumerate(dataset):
            requirement_text = item.get('requirement', '')
            sample_id = item.get('sample_id', None)

            ground_truth = None
            if 'label' in item and item['label']:
                label_str = str(item['label']).strip().lower()
                for req_type in RequirementType:
                    if req_type.value == label_str:
                        ground_truth = req_type
                        break

            episode = self.classify_requirement(
                requirement_text=requirement_text,
                requirement_id=f"req_{i + 1:04d}",
                ground_truth=ground_truth,
                sample_id=sample_id
            )
            episodes.append(episode)

            if (i + 1) % 5 == 0:
                print(f"\nProgress: {i + 1}/{len(dataset)}")
                if self.statistics.get("accuracy"):
                    acc = np.mean(self.statistics["accuracy"]) * 100
                    print(f"  Current Accuracy: {acc:.1f}%")

            if i < len(dataset) - 1:
                time.sleep(Config.REQUEST_INTERVAL)
        return episodes

    def save_results(self, episodes, output_file, dataset):
        results_data = []
        for i, episode in enumerate(episodes):
            true_label_original = ""
            if dataset and i < len(dataset) and 'label' in dataset[i]:
                true_label_original = str(
                    dataset[i].get('label', '')).lower().strip()

            is_correct = False
            if episode.ground_truth:
                is_correct = (episode.final_classification
                              == episode.ground_truth)

            error_type = "correct" if is_correct else (
                "misclassification" if episode.ground_truth else "unknown")


            decision_path = []
            path_evidence = []
            for step in episode.steps:
                result = 'PASS' if step.decision else 'FAIL'
                decision_path.append(f"{step.step.value}:{result}")
                path_evidence.extend(step.evidence)

            row = {
                'sample_id': episode.sample_id,
                'id': i + 1,
                'requirement_id': episode.requirement_id,
                'requirement_text': episode.requirement_text,
                'true_label': (episode.ground_truth.value
                               if episode.ground_truth else ""),
                'predicted_label': (episode.final_classification.value
                                    if episode.final_classification else ""),
                'prediction_confidence': (
                    np.mean([s.confidence for s in episode.steps])
                    if episode.steps else 0.0),
                'is_correct': is_correct,
                'error_type': error_type,
                'step_count': len(episode.steps),
                'step_details': ' -> '.join(decision_path),
                'decision_path_evidence': ' | '.join(path_evidence[:10]),
                'final_classification_original': (
                    episode.final_classification.value
                    if episode.final_classification else ""),
                'true_label_original': true_label_original,
                'is_consistent': 'Yes' if is_correct else 'No',
                'total_reward': round(episode.total_reward, 3),
                'avg_confidence': round(
                    np.mean([s.confidence for s in episode.steps])
                    if episode.steps else 0, 3),
                'evidence_count': sum(len(s.evidence)
                                      for s in episode.steps),
                'processing_time': datetime.datetime.now().strftime(
                    '%Y-%m-%d %H:%M:%S'),
            }

            step_names = ['step1', 'step2', 'step3', 'step4']
            for j, step in enumerate(episode.steps[:4]):
                row[f'{step_names[j]}_decision'] = step.step.value
                row[f'{step_names[j]}_result'] = ('PASS'
                                                  if step.decision else 'FAIL')
                row[f'{step_names[j]}_confidence'] = step.confidence
                row[f'{step_names[j]}_evidence'] = (
                        ', '.join(step.evidence[:3])
                        + ('...' if len(step.evidence) > 3 else ''))
                row[f'{step_names[j]}_reasoning'] = step.reasoning
            for j in range(len(episode.steps), 4):
                row[f'{step_names[j]}_decision'] = ''
                row[f'{step_names[j]}_result'] = ''
                row[f'{step_names[j]}_confidence'] = ''
                row[f'{step_names[j]}_evidence'] = ''
                row[f'{step_names[j]}_reasoning'] = ''

            results_data.append(row)

        df = pd.DataFrame(results_data)
        os.makedirs(os.path.dirname(output_file)
                    if os.path.dirname(output_file) else '.', exist_ok=True)
        df.to_excel(output_file, index=False, engine='openpyxl')
        print(f"\nResults saved to: {output_file}")
        return df




def read_csv_auto(path: str, encodings=None) -> pd.DataFrame:
    """依次尝试多种编码读取 CSV，返回第一个成功的 DataFrame。"""
    if encodings is None:
        encodings = Config.CSV_ENCODINGS

    last_err = None
    for enc in encodings:
        try:
            df = pd.read_csv(path, encoding=enc)
            print(f"[CSV] 使用编码 '{enc}' 成功读取")
            return df
        except UnicodeDecodeError as e:
            print(f"[CSV] 编码 '{enc}' 失败：{e}")
            last_err = e
    raise last_err if last_err else RuntimeError("CSV 读取失败")




def main():
    print("=" * 80)
    print("Hierarchical Binary Decision Routing - TEST SPLIT ONLY")
    print("Decision Flow: composite -> interactive -> sequential -> conditional -> structural")
    print("=" * 80)
    print(f"\nPath Information:")
    print(f"  Script Directory: {Config.SCRIPT_DIR}")
    print(f"  Project Root:     {Config.PROJECT_ROOT}")
    print(f"  Data Directory:   {Config.DATA_DIR}")
    print(f"  BERT CSV:         {Config.BERT_CSV}")
    print(f"  Concept XLSX:     {Config.CONCEPT_FILE}")
    print(f"  Examples XLSX:    {Config.EXAMPLE_FILE}")
    print(f"  Output File:      {Config.OUTPUT_FILE}")

    api_key = os.getenv(Config.API_KEY_ENV)
    if not api_key:
        print(f"\n  [ERROR] {Config.API_KEY_ENV} is not set")
        return
    print(f"\n  [OK] {Config.API_KEY_ENV} is set")

    for path, name in [(Config.BERT_CSV, "BERT test CSV"),
                       (Config.CONCEPT_FILE, "Concept.xlsx"),
                       (Config.EXAMPLE_FILE, "RequirementExamples.xlsx")]:
        if not os.path.exists(path):
            print(f"\n  [ERROR] {name} not found: {path}")
            return
        print(f"  [OK] {name} found")

    try:
        print(f"\nLoading knowledge base...")
        kb = KnowledgeBaseLoader(
            concept_path=Config.CONCEPT_FILE,
            examples_path=Config.EXAMPLE_FILE,
            max_examples_per_class=Config.MAX_EXAMPLES_PER_CLASS,
        )

        print(f"\nLoading BERT test split...")


        xlsx_path = os.path.splitext(Config.BERT_CSV)[0] + ".xlsx"
        if os.path.exists(xlsx_path):
            print(f"[DATA] 使用 xlsx：{xlsx_path}")
            bert_df = pd.read_excel(xlsx_path)
        else:
            print(f"[DATA] 未找到 xlsx，回退读取 csv：{Config.BERT_CSV}")
            bert_df = read_csv_auto(Config.BERT_CSV)
        bert_df.columns = [c.strip() for c in bert_df.columns]
        print(f"Loaded {len(bert_df)} samples")
        print(f"  Columns: {list(bert_df.columns)}")

        dataset = []
        for _, row in bert_df.iterrows():
            item = {
                'sample_id': int(row['sample_id']),
                'requirement': str(row['requirement']),
                'label': str(row['true_label']).lower().strip(),
            }
            dataset.append(item)
        print(f"\nDataset prepared: {len(dataset)} items")

        print(f"\nBuilding LLM client with unified System Prompt...")
        llm_client = DoubaoLLMClient(
            model_name=Config.MODEL_NAME,
            seed=Config.SEED,
            knowledge_base=kb,
        )
        print(f"  Model: {Config.MODEL_NAME}")
        print(f"  System Prompt length: {len(llm_client.system_prompt)} chars")
        print(f"  Categories: {len(kb.category_definitions)}")
        print(f"  Few-shot classes: {len(kb.few_shot_examples)}")

        print(f"\nStarting hierarchical binary decision routing...")
        processor = SequentialClassificationProcessor(llm_client=llm_client)
        episodes = processor.process_dataset(dataset)

        df_result = processor.save_results(episodes, Config.OUTPUT_FILE,
                                            dataset)

        print(f"\n{'=' * 60}")
        print("Classification Statistics")
        print(f"{'=' * 60}")
        if processor.statistics.get("accuracy"):
            acc = np.mean(processor.statistics["accuracy"]) * 100
            print(f"Accuracy: {acc:.1f}%")
        if processor.statistics.get("steps"):
            print(f"Average Steps: "
                  f"{np.mean(processor.statistics['steps']):.1f}")
        if processor.statistics.get("rewards"):
            print(f"Average Reward: "
                  f"{np.mean(processor.statistics['rewards']):.3f}")

        print(f"\nLabel Distribution:")
        for label, count in df_result['predicted_label'].value_counts().items():
            if label:
                print(f"   {label}: {count} "
                      f"({count / len(df_result) * 100:.1f}%)")

        print(f"\n{'=' * 60}")
        print("Processing Complete!")
        print(f"Output file: {Config.OUTPUT_FILE}")
        print("=" * 60)

    except Exception as e:
        print(f"\nProcessing failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()