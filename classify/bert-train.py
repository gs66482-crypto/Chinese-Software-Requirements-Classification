import os
os.environ['TRANSFORMERS_NO_TF'] = '1'
os.environ['USE_TORCH'] = 'TRUE'
os.environ['TRANSFORMERS_NO_FLAX'] = '1'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import copy
import pandas as pd
import numpy as np
import torch
import json
import warnings
from datetime import datetime
import logging

from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    get_linear_schedule_with_warmup,
)

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix,
)
from sklearn.utils.class_weight import compute_class_weight

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Config:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

    DATA_DIR = os.path.join(PROJECT_ROOT, "data")
    DATA_FILE = "dataset.xlsx"
    DATA_PATH = os.path.join(DATA_DIR, DATA_FILE)

    OUTPUT_DIR = os.path.join(PROJECT_ROOT, "classifyOutcome", "BERT")

    MODEL_NAME = 'bert-base-chinese'
    MAX_LENGTH = 64
    BATCH_SIZE = 16
    LEARNING_RATE = 2e-5
    EPOCHS = 10

    TEST_SIZE = 0.2
    VAL_SIZE = 0.1
    RANDOM_STATE = 42

    WARMUP_RATIO = 0.1
    WEIGHT_DECAY = 0.01
    GRADIENT_CLIP = 1.0

    EARLY_STOPPING_PATIENCE = 3
    EARLY_STOPPING_THRESHOLD = 0.001

    USE_AUGMENTATION = True
    AUGMENTATION_RATIO = 0.3

    USE_CLASS_WEIGHTS = True

    CONFIDENCE_THRESHOLD = 0.7

    EXPECTED_NUM_LABELS = 5

    SAVE_TRAIN_VAL_PREDICTIONS = True


class DataAugmentor:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def synonym_replacement(self, text, n=1):
        words = list(text)
        if len(words) < 3:
            return text

        import random
        positions = random.sample(range(len(words)), min(n, len(words) // 2))

        for pos in positions:
            if random.random() > 0.5:
                words[pos] = ''
            else:
                words[pos] = words[pos] * 2

        return ''.join(words)

    def augment_dataset(self, df, ratio=0.3):
        augmented_dfs = [df]

        label_counts = df['label'].value_counts()
        minority_labels = label_counts[
            label_counts < label_counts.max() * 0.3
        ].index.tolist()

        for label in minority_labels:
            label_data = df[df['label'] == label]
            n_augment = int(len(label_data) * ratio)

            if n_augment > 0:
                sampled_data = label_data.sample(
                    n=min(n_augment, len(label_data)),
                    replace=True, random_state=42
                )
                augmented_rows = []

                for _, row in sampled_data.iterrows():
                    new_text = self.synonym_replacement(row['requirement'], n=1)
                    augmented_rows.append({
                        'requirement': new_text,
                        'label': int(label),
                        'sample_id': row.get('sample_id', -1),
                    })

                if augmented_rows:
                    augmented_dfs.append(pd.DataFrame(augmented_rows))

        return pd.concat(augmented_dfs, ignore_index=True)


class RequirementDataset(Dataset):
    def __init__(self, requirements, labels, tokenizer, max_length=64):
        self.requirements = requirements
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.requirements)

    def __getitem__(self, idx):
        requirement = str(self.requirements[idx])
        label = self.labels[idx]

        try:
            label_int = int(label)
        except (ValueError, TypeError):
            raise TypeError(
                f"Label at idx {idx} is not integer-convertible: "
                f"{label!r} (type={type(label).__name__}). "
                f"Check prepare_labels()."
            )

        encoding = self.tokenizer(
            requirement,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label_int, dtype=torch.long),
        }


class BertTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = None
        self.model = None
        self.label_mapping = None
        self.label_names = None
        self.num_labels = None

        logger.info(f"Using device: {self.device}")
        if torch.cuda.is_available():
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            logger.info(
                f"GPU Memory: "
                f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB"
            )

        logger.info(f"Project root: {config.PROJECT_ROOT}")
        logger.info(f"Data path:    {config.DATA_PATH}")
        logger.info(f"Output dir:   {config.OUTPUT_DIR}")

        os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    def load_data(self):
        logger.info("Loading data...")

        if not os.path.exists(self.config.DATA_PATH):
            logger.error(f"Data file not found: {self.config.DATA_PATH}")
            logger.info(f"Contents of data dir ({self.config.DATA_DIR}):")
            if os.path.isdir(self.config.DATA_DIR):
                for name in os.listdir(self.config.DATA_DIR):
                    logger.info(f"  {name}")
            raise FileNotFoundError(self.config.DATA_PATH)

        df = pd.read_excel(self.config.DATA_PATH)
        logger.info(f"Loaded {len(df)} samples")
        logger.info(f"Columns: {list(df.columns)}")

        if 'requirement' not in df.columns or 'label' not in df.columns:
            for col in df.columns:
                col_lower = col.lower()
                if 'requirement' in col_lower or '需求' in col_lower:
                    df = df.rename(columns={col: 'requirement'})
                if ('label' in col_lower or '标签' in col_lower
                        or '类别' in col_lower):
                    df = df.rename(columns={col: 'label'})

        df = df.dropna(subset=['requirement', 'label'])
        df = df.drop_duplicates(subset=['requirement']).reset_index(drop=True)

        id_candidates = [
            'sample_id', 'id', 'ID', 'requirement_id', '需求ID', '编号'
        ]
        id_col = next((c for c in id_candidates if c in df.columns), None)

        if id_col is not None:
            df = df.rename(columns={id_col: 'sample_id'})
            logger.info(f"Using existing column '{id_col}' as sample_id")
        else:
            df['sample_id'] = np.arange(len(df))
            logger.info("Generated sample_id from row index (0..N-1)")

        logger.info(f"After cleaning: {len(df)} samples")
        logger.info(
            f"sample_id range: "
            f"[{df['sample_id'].min()}, {df['sample_id'].max()}]"
        )
        return df

    def prepare_labels(self, df):
        raw_labels = df['label'].astype(str)
        unique_labels = sorted(raw_labels.unique())

        self.label_mapping = {
            label: idx for idx, label in enumerate(unique_labels)
        }
        df['label'] = raw_labels.map(self.label_mapping).astype(int)
        self.label_names = unique_labels

        self.num_labels = len(unique_labels)
        logger.info(f"Number of classes: {self.num_labels}")
        logger.info(f"Label names: {self.label_names}")
        logger.info(f"Label mapping: {self.label_mapping}")
        logger.info(f"Label dtype after mapping: {df['label'].dtype}")
        logger.info(
            f"Label value counts (top-10):\n"
            f"{df['label'].value_counts().head(10)}"
        )

        if self.config.EXPECTED_NUM_LABELS is not None:
            if self.num_labels != self.config.EXPECTED_NUM_LABELS:
                raise AssertionError(
                    f"Expected {self.config.EXPECTED_NUM_LABELS} categories "
                    f"per paper, got {self.num_labels}. "
                    f"Please reconcile paper text vs dataset: {self.label_names}"
                )

        return df

    def stratified_split(self, df):
        logger.info("Splitting dataset...")

        train_val_df, test_df = train_test_split(
            df,
            test_size=self.config.TEST_SIZE,
            random_state=self.config.RANDOM_STATE,
            stratify=df['label']
        )

        val_size = self.config.VAL_SIZE / (1 - self.config.TEST_SIZE)
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=val_size,
            random_state=self.config.RANDOM_STATE,
            stratify=train_val_df['label']
        )

        train_df = train_df.reset_index(drop=True)
        val_df = val_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)

        logger.info(f"train_df['label'].dtype = {train_df['label'].dtype}")
        logger.info(f"val_df['label'].dtype   = {val_df['label'].dtype}")
        logger.info(f"test_df['label'].dtype  = {test_df['label'].dtype}")

        logger.info(f"Training set:   {len(train_df)} samples")
        logger.info(f"Validation set: {len(val_df)} samples")
        logger.info(f"Test set:       {len(test_df)} samples")

        return train_df, val_df, test_df

    def create_dataloaders(self, train_df, val_df, test_df):
        logger.info("Creating dataloaders...")

        self.tokenizer = BertTokenizer.from_pretrained(self.config.MODEL_NAME)

        if self.config.USE_AUGMENTATION:
            augmentor = DataAugmentor(self.tokenizer)
            train_df = augmentor.augment_dataset(
                train_df, self.config.AUGMENTATION_RATIO
            )
            logger.info(f"After augmentation: {len(train_df)} samples")
            logger.info(
                f"After augmentation, train_df['label'].dtype = "
                f"{train_df['label'].dtype}"
            )

        train_df['label'] = train_df['label'].astype(int)
        val_df['label'] = val_df['label'].astype(int)
        test_df['label'] = test_df['label'].astype(int)

        train_dataset = RequirementDataset(
            train_df['requirement'].values,
            train_df['label'].values,
            self.tokenizer,
            self.config.MAX_LENGTH
        )
        val_dataset = RequirementDataset(
            val_df['requirement'].values,
            val_df['label'].values,
            self.tokenizer,
            self.config.MAX_LENGTH
        )
        test_dataset = RequirementDataset(
            test_df['requirement'].values,
            test_df['label'].values,
            self.tokenizer,
            self.config.MAX_LENGTH
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=True,
            num_workers=0
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=False,
            num_workers=0
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=False,
            num_workers=0
        )

        return train_loader, val_loader, test_loader, train_df

    def init_model(self):
        logger.info("Initializing model...")
        self.model = BertForSequenceClassification.from_pretrained(
            self.config.MODEL_NAME,
            num_labels=self.num_labels
        )
        self.model.to(self.device)
        return self.model

    def get_class_weights(self, train_df):
        if self.config.USE_CLASS_WEIGHTS:
            class_weights = compute_class_weight(
                'balanced',
                classes=np.unique(train_df['label']),
                y=train_df['label']
            )
            class_weights = torch.tensor(
                class_weights, dtype=torch.float
            ).to(self.device)
            logger.info(f"Class weights: {class_weights}")
            return class_weights
        return None

    def train_epoch(self, train_loader, optimizer, scheduler,
                    class_weights=None):
        self.model.train()
        total_loss = 0
        correct_predictions = 0

        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)

            optimizer.zero_grad()

            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss

            if class_weights is not None:
                loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights)
                loss = loss_fct(outputs.logits, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.GRADIENT_CLIP
            )
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            _, preds = torch.max(outputs.logits, dim=1)
            correct_predictions += torch.sum(preds == labels)

            if (batch_idx + 1) % 20 == 0:
                avg_loss = total_loss / (batch_idx + 1)
                accuracy = (
                    correct_predictions.double()
                    / ((batch_idx + 1) * self.config.BATCH_SIZE)
                )
                logger.info(
                    f"  Batch {batch_idx + 1}/{len(train_loader)}, "
                    f"Loss: {avg_loss:.4f}, Acc: {accuracy:.4f}"
                )

        avg_loss = total_loss / len(train_loader)
        accuracy = correct_predictions.double() / len(train_loader.dataset)
        return avg_loss, accuracy

    def evaluate(self, data_loader, return_details=False):
        self.model.eval()
        predictions, true_labels, confidences = [], [], []
        all_probabilities = []
        total_loss = 0

        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                total_loss += outputs.loss.item()

                probabilities = torch.softmax(outputs.logits, dim=1)
                conf, preds = torch.max(probabilities, dim=1)

                predictions.extend(preds.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())
                confidences.extend(conf.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())

        avg_loss = total_loss / len(data_loader)
        accuracy = accuracy_score(true_labels, predictions)

        if return_details:
            return (avg_loss, accuracy, predictions, true_labels,
                    confidences, all_probabilities)
        return avg_loss, accuracy

    def train(self, train_loader, val_loader, train_df):
        logger.info("Starting training...")

        class_weights = self.get_class_weights(train_df)

        optimizer = AdamW(
            self.model.parameters(),
            lr=self.config.LEARNING_RATE,
            weight_decay=self.config.WEIGHT_DECAY
        )

        total_steps = len(train_loader) * self.config.EPOCHS
        warmup_steps = int(total_steps * self.config.WARMUP_RATIO)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )

        best_val_accuracy = 0
        patience_counter = 0
        best_model_state = None

        history = {
            'train_loss': [], 'train_accuracy': [],
            'val_loss': [], 'val_accuracy': []
        }

        for epoch in range(self.config.EPOCHS):
            logger.info(f"\nEpoch {epoch + 1}/{self.config.EPOCHS}")
            logger.info("-" * 50)

            train_loss, train_acc = self.train_epoch(
                train_loader, optimizer, scheduler, class_weights
            )
            val_loss, val_acc = self.evaluate(val_loader)

            history['train_loss'].append(train_loss)
            history['train_accuracy'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_accuracy'].append(val_acc)

            logger.info(
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}"
            )
            logger.info(
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
            )

            if val_acc > best_val_accuracy + self.config.EARLY_STOPPING_THRESHOLD:
                best_val_accuracy = val_acc
                patience_counter = 0
                best_model_state = copy.deepcopy(self.model.state_dict())
                logger.info(f"New best model! Val Acc: {best_val_accuracy:.4f}")
            else:
                patience_counter += 1
                logger.info(
                    f"Early stopping patience: "
                    f"{patience_counter}/{self.config.EARLY_STOPPING_PATIENCE}"
                )
                if patience_counter >= self.config.EARLY_STOPPING_PATIENCE:
                    logger.info("Early stopping triggered!")
                    break

        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            logger.info(
                f"Restored best model with val accuracy: "
                f"{best_val_accuracy:.4f}"
            )

        self.plot_training_curves(history)
        return history, best_val_accuracy

    def evaluate_with_confidence(self, test_loader):
        logger.info("Evaluating on held-out test split...")

        (test_loss, test_acc, predictions, true_labels,
         confidences, all_probabilities) = self.evaluate(
            test_loader, return_details=True
        )

        logger.info(f"[HEADLINE] Test Accuracy: {test_acc:.4f}")

        confidence_metrics = self.calculate_confidence_metrics(
            true_labels, predictions, confidences
        )

        logger.info("\nClassification Report (held-out test):")
        logger.info(classification_report(
            true_labels, predictions,
            target_names=self.label_names, digits=4
        ))

        self.plot_confusion_matrix(true_labels, predictions)

        return {
            'test_loss': test_loss,
            'test_accuracy': test_acc,
            'predictions': predictions,
            'true_labels': true_labels,
            'confidences': confidences,
            'all_probabilities': all_probabilities,
            'confidence_metrics': confidence_metrics,
        }

    def calculate_confidence_metrics(self, true_labels, predictions,
                                     confidences, n_bins=10):
        true_labels = np.array(true_labels)
        predictions = np.array(predictions)
        confidences = np.array(confidences)

        correct = (true_labels == predictions).astype(int)

        pearson_corr, pearson_p = pearsonr(confidences, correct)

        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        for i in range(n_bins):
            in_bin = (
                (confidences >= bin_boundaries[i])
                & (confidences < bin_boundaries[i + 1])
            )
            n_in_bin = np.sum(in_bin)
            if n_in_bin > 0:
                acc_in_bin = np.mean(correct[in_bin])
                conf_in_bin = np.mean(confidences[in_bin])
                ece += np.abs(conf_in_bin - acc_in_bin) * n_in_bin
        ece = ece / len(confidences)

        high_conf_mask = confidences >= self.config.CONFIDENCE_THRESHOLD
        high_conf_acc = (
            np.mean(correct[high_conf_mask])
            if np.sum(high_conf_mask) > 0 else 0
        )

        logger.info("\nConfidence Metrics (held-out test):")
        logger.info(f"  Pearson Correlation: {pearson_corr:.4f}")
        logger.info(f"  ECE: {ece:.4f}")
        logger.info(
            f"  High Confidence (>={self.config.CONFIDENCE_THRESHOLD}) "
            f"Accuracy: {high_conf_acc:.4f}"
        )
        logger.info(
            f"  High Confidence Count: "
            f"{np.sum(high_conf_mask)}/{len(confidences)}"
        )

        return {
            'pearson_correlation': pearson_corr,
            'pearson_p_value': pearson_p,
            'ece': ece,
            'high_conf_accuracy': high_conf_acc,
            'high_conf_count': int(np.sum(high_conf_mask)),
            'confidences': confidences,
        }

    def plot_training_curves(self, history):
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
            epochs = range(1, len(history['train_loss']) + 1)

            ax1.plot(epochs, history['train_loss'], 'b-',
                     label='Training Loss', linewidth=2)
            ax1.plot(epochs, history['val_loss'], 'r-',
                     label='Validation Loss', linewidth=2)
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.set_title('Training and Validation Loss')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            ax2.plot(epochs,
                     [a * 100 for a in history['train_accuracy']],
                     'b-', label='Training Accuracy', linewidth=2)
            ax2.plot(epochs,
                     [a * 100 for a in history['val_accuracy']],
                     'r-', label='Validation Accuracy', linewidth=2)
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Accuracy (%)')
            ax2.set_title('Training and Validation Accuracy')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            out_path = os.path.join(
                self.config.OUTPUT_DIR, 'training_curves.png'
            )
            plt.savefig(out_path, dpi=300)
            plt.close()
            logger.info(f"Training curves saved to {out_path}")
        except Exception as e:
            logger.warning(f"Could not save training curves: {e}")

    def plot_confusion_matrix(self, true_labels, predictions):
        try:
            cm = confusion_matrix(true_labels, predictions)

            plt.figure(figsize=(12, 10))
            sns.heatmap(
                cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=self.label_names,
                yticklabels=self.label_names
            )
            plt.title('Confusion Matrix (held-out test split)',
                      fontsize=16, fontweight='bold')
            plt.ylabel('True Label', fontsize=14)
            plt.xlabel('Predicted Label', fontsize=14)
            plt.tight_layout()

            out_path = os.path.join(
                self.config.OUTPUT_DIR, 'confusion_matrix.png'
            )
            plt.savefig(out_path, dpi=300)
            plt.close()
            logger.info(f"Confusion matrix saved to {out_path}")
        except Exception as e:
            logger.warning(f"Could not save confusion matrix: {e}")

    def save_model(self):
        logger.info(f"Saving model to {self.config.OUTPUT_DIR}...")

        model_info = {
            'model_state_dict': self.model.state_dict(),
            'config': {
                'num_labels': self.num_labels,
                'label_names': self.label_names,
                'label_mapping': self.label_mapping,
                'max_length': self.config.MAX_LENGTH,
                'model_name': self.config.MODEL_NAME,
            }
        }

        model_path = os.path.join(self.config.OUTPUT_DIR, 'bert_model.pth')
        torch.save(model_info, model_path)

        config_path = os.path.join(self.config.OUTPUT_DIR, 'config.json')
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump({
                'num_labels': self.num_labels,
                'label_names': self.label_names,
                'label_mapping': self.label_mapping,
                'max_length': self.config.MAX_LENGTH,
                'model_name': self.config.MODEL_NAME,
            }, f, ensure_ascii=False, indent=2)

        logger.info(f"Model saved to {model_path}")
        logger.info(f"Config saved to {config_path}")

    def save_split_predictions(self, split_name, split_df, results):
        predictions, true_labels, confidences = results

        out = pd.DataFrame({
            'sample_id': split_df['sample_id'].values,
            'requirement': split_df['requirement'].values,
            'true_label_id': true_labels,
            'true_label': [self.label_names[i] for i in true_labels],
            'bert_pred_id': predictions,
            'bert_pred': [self.label_names[i] for i in predictions],
            'bert_confidence': confidences,
        })
        out['bert_correct'] = (
            out['true_label_id'] == out['bert_pred_id']
        ).astype(int)

        path = os.path.join(
            self.config.OUTPUT_DIR, f'bert_{split_name}_predictions.xlsx'
        )
        out.to_excel(path, index=False)
        logger.info(
            f"[{split_name}] Saved {len(out)} rows -> {path}, "
            f"acc={out['bert_correct'].mean():.4f}"
        )
        return out

    def predict_all_data(self, df):
        logger.info("\n" + "=" * 60)
        logger.info("DIAGNOSTIC: All-Data Performance (INCLUDES TRAINING DATA)")
        logger.info("DO NOT report this as a fair comparison vs. LLMs.")
        logger.info("=" * 60)

        all_dataset = RequirementDataset(
            df['requirement'].values,
            df['label'].values,
            self.tokenizer,
            self.config.MAX_LENGTH
        )
        all_loader = DataLoader(
            all_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=False,
            num_workers=0
        )

        _, _, predictions, true_labels, confidences, _ = self.evaluate(
            all_loader, return_details=True
        )

        accuracy = accuracy_score(true_labels, predictions)
        logger.info(f"All Data Accuracy: {accuracy:.4f} (n={len(df)})")

        return {
            'accuracy': accuracy,
            'predictions': predictions,
            'true_labels': true_labels,
            'confidences': confidences,
        }


def generate_final_report(trainer, test_results, all_results,
                          best_val_acc, config):
    report_path = os.path.join(config.OUTPUT_DIR, 'training_report.txt')

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("BERT TRAINING FINAL REPORT\n")
        f.write("=" * 70 + "\n\n")

        f.write(
            f"Training Date: "
            f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        )
        f.write(f"Project root: {config.PROJECT_ROOT}\n")
        f.write(f"Data path:    {config.DATA_PATH}\n")
        f.write(f"Output path:  {config.OUTPUT_DIR}\n\n")

        f.write("-" * 50 + "\n")
        f.write("HELD-OUT TEST SET (headline metrics)\n")
        f.write("-" * 50 + "\n")
        f.write(f"  n:              {len(test_results['true_labels'])}\n")
        f.write(f"  Accuracy:       {test_results['test_accuracy']:.4f}\n")
        f.write(f"  Loss:           {test_results['test_loss']:.4f}\n")
        f.write(
            f"  ECE:            "
            f"{test_results['confidence_metrics']['ece']:.4f}\n"
        )
        f.write(
            f"  Pearson r:      "
            f"{test_results['confidence_metrics']['pearson_correlation']:.4f}\n"
        )
        f.write(
            f"  High-Conf Acc:  "
            f"{test_results['confidence_metrics']['high_conf_accuracy']:.4f}\n\n"
        )

        f.write("-" * 50 + "\n")
        f.write("VALIDATION SET (model selection)\n")
        f.write("-" * 50 + "\n")
        f.write(f"  Best Val Accuracy: {best_val_acc:.4f}\n\n")

        f.write("-" * 50 + "\n")
        f.write("DIAGNOSTIC (INCLUDES TRAINING DATA - NOT for LLM comparison)\n")
        f.write("-" * 50 + "\n")
        f.write(f"  All-data accuracy: {all_results['accuracy']:.4f}\n")
        f.write(f"  n:                 {len(all_results['true_labels'])}\n\n")

        f.write("-" * 50 + "\n")
        f.write("PER-SAMPLE PREDICTIONS\n")
        f.write("-" * 50 + "\n")
        f.write(
            f"  test:  "
            f"{os.path.join(config.OUTPUT_DIR, 'bert_test_predictions.xlsx')}\n"
        )
        if config.SAVE_TRAIN_VAL_PREDICTIONS:
            f.write(
                f"  train: "
                f"{os.path.join(config.OUTPUT_DIR, 'bert_train_predictions.xlsx')}\n"
            )
            f.write(
                f"  val:   "
                f"{os.path.join(config.OUTPUT_DIR, 'bert_val_predictions.xlsx')}\n"
            )
        f.write("\n")
        f.write(
            "  These XLSX files contain the exact samples used for the "
            "headline metric.\n"
        )
        f.write("  Use them for statistical tests vs. LLM baselines.\n\n")

        f.write("-" * 50 + "\n")
        f.write("RECOMMENDATIONS\n")
        f.write("-" * 50 + "\n\n")

        if test_results['test_accuracy'] < 0.85:
            f.write("!! Test accuracy < 85%. Consider:\n")
            f.write("   - Increase training data\n")
            f.write("   - Use data augmentation\n")
            f.write("   - Try a different model (e.g., RoBERTa-wwm-ext)\n\n")

        if test_results['confidence_metrics']['ece'] > 0.1:
            f.write("!! ECE > 0.1. Model calibration issues:\n")
            f.write("   - Consider temperature scaling\n")
            f.write("   - Try label smoothing during training\n\n")

        if all_results['accuracy'] < test_results['test_accuracy'] * 0.9:
            f.write("!! Large train/test generalization gap detected.\n")
            f.write("   - All-data accuracy is diagnostic only\n")
            f.write("   - Report only held-out test accuracy for LLM comparison\n\n")

        f.write("=" * 70 + "\n")

    logger.info(f"Report saved to {report_path}")


def main():
    logger.info("=" * 60)
    logger.info("BERT Requirement Classification - Training")
    logger.info("=" * 60)

    config = Config()
    trainer = BertTrainer(config)

    df = trainer.load_data()
    df = trainer.prepare_labels(df)

    train_df, val_df, test_df = trainer.stratified_split(df)

    train_loader, val_loader, test_loader, augmented_train_df = \
        trainer.create_dataloaders(train_df, val_df, test_df)

    trainer.init_model()
    history, best_val_acc = trainer.train(
        train_loader, val_loader, augmented_train_df
    )

    test_results = trainer.evaluate_with_confidence(test_loader)

    trainer.save_split_predictions(
        'test', test_df,
        (test_results['predictions'],
         test_results['true_labels'],
         test_results['confidences'])
    )

    if config.SAVE_TRAIN_VAL_PREDICTIONS:
        train_eval_dataset = RequirementDataset(
            train_df['requirement'].values,
            train_df['label'].values,
            trainer.tokenizer,
            config.MAX_LENGTH
        )
        train_eval_loader = DataLoader(
            train_eval_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            num_workers=0
        )
        _, _, tr_pred, tr_true, tr_conf, _ = trainer.evaluate(
            train_eval_loader, return_details=True
        )
        trainer.save_split_predictions(
            'train', train_df, (tr_pred, tr_true, tr_conf)
        )

        _, _, va_pred, va_true, va_conf, _ = trainer.evaluate(
            val_loader, return_details=True
        )
        trainer.save_split_predictions(
            'val', val_df, (va_pred, va_true, va_conf)
        )

    all_results = trainer.predict_all_data(df)

    trainer.save_model()

    generate_final_report(
        trainer, test_results, all_results, best_val_acc, config
    )

    logger.info("\n" + "=" * 60)
    logger.info("Training completed successfully!")
    logger.info(f"All outputs saved to: {config.OUTPUT_DIR}")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()