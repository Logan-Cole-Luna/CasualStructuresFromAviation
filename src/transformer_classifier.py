"""
Transformer-based classifier for NTSB aviation accident narrative categorization.

Uses DistilBERT to classify narratives into top-level finding categories.
"""
import json
import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Guard transformer/torch imports so the rest of the project doesn't break
# if these heavy dependencies are not installed.
try:
    import torch
    from torch.utils.data import DataLoader
    from transformers import (
        DistilBertForSequenceClassification,
        DistilBertTokenizerFast,
        get_linear_schedule_with_warmup,
    )
    from torch.optim import AdamW
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from sklearn.metrics import classification_report
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


LABEL_COLS = [
    'Personnel issues',
    'Aircraft',
    'Environmental issues',
    'Not determined',
]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class NarrativeDataset:
    """
    PyTorch Dataset wrapping tokenized narratives and integer labels.
    Falls back gracefully if torch is not available (returns dicts of lists).
    """

    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 256):
        if not TORCH_AVAILABLE:
            raise ImportError("torch is required for NarrativeDataset.")

        self.labels = labels
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors='pt',
        )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        item = {
            'input_ids':      self.encodings['input_ids'][idx],
            'attention_mask': self.encodings['attention_mask'][idx],
            'labels':         torch.tensor(self.labels[idx], dtype=torch.long),
        }
        return item


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

class NTSBClassifier:
    """DistilBERT-based classifier for NTSB accident narrative categories."""

    def __init__(
        self,
        num_labels: int = 4,
        model_name: str = 'distilbert-base-uncased',
    ):
        if not TORCH_AVAILABLE:
            raise ImportError(
                "torch and transformers are required for NTSBClassifier. "
                "Install them with: pip install torch transformers"
            )

        self.num_labels = num_labels
        self.model_name = model_name
        self.label_map: dict = {}         # label_str -> int
        self.inv_label_map: dict = {}     # int -> label_str

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[Classifier] Using device: {self.device}")

        self.tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)
        self.model = DistilBertForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
        )
        self.model.to(self.device)

    # ------------------------------------------------------------------
    # Data preparation
    # ------------------------------------------------------------------

    def prepare_data(
        self,
        df: pd.DataFrame,
        text_col: str = 'narr_clean',
        label_col: str = 'top_category',
        test_size: float = 0.15,
        val_size: float = 0.15,
        max_samples: Optional[int] = None,
    ) -> Tuple['NarrativeDataset', 'NarrativeDataset', 'NarrativeDataset', dict]:
        """
        Filter to LABEL_COLS categories, optionally subsample, perform
        stratified train/val/test split, and return (train_ds, val_ds, test_ds, label_map).
        """
        # Filter to known categories
        df_filtered = df[df[label_col].isin(LABEL_COLS)].dropna(subset=[text_col, label_col]).copy()

        if max_samples is not None and len(df_filtered) > max_samples:
            df_filtered = df_filtered.sample(n=max_samples, random_state=42).reset_index(drop=True)

        # Build label mapping
        self.label_map = {label: idx for idx, label in enumerate(sorted(LABEL_COLS))}
        self.inv_label_map = {v: k for k, v in self.label_map.items()}

        texts = df_filtered[text_col].astype(str).tolist()
        labels = df_filtered[label_col].map(self.label_map).tolist()

        # Stratified split: first carve out test, then val from remainder
        train_texts, test_texts, train_labels, test_labels = train_test_split(
            texts, labels,
            test_size=test_size,
            random_state=42,
            stratify=labels,
        )

        # val_size is relative to the original dataset size
        val_relative = val_size / (1.0 - test_size)
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            train_texts, train_labels,
            test_size=val_relative,
            random_state=42,
            stratify=train_labels,
        )

        train_ds = NarrativeDataset(train_texts, train_labels, self.tokenizer)
        val_ds   = NarrativeDataset(val_texts,   val_labels,   self.tokenizer)
        test_ds  = NarrativeDataset(test_texts,  test_labels,  self.tokenizer)

        print(f"[Classifier] Data split — train: {len(train_ds)}, val: {len(val_ds)}, test: {len(test_ds)}")
        return train_ds, val_ds, test_ds, self.label_map

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        train_ds: 'NarrativeDataset',
        val_ds: 'NarrativeDataset',
        epochs: int = 3,
        batch_size: int = 16,
        lr: float = 2e-5,
        save_path: Optional[str] = None,
    ):
        """
        Fine-tune the classifier with AdamW + linear warmup scheduler.
        Prints epoch loss and validation accuracy. Saves model if save_path given.
        """
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

        optimizer = AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
        total_steps = len(train_loader) * epochs
        warmup_steps = max(1, int(0.1 * total_steps))

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        best_val_acc = 0.0

        for epoch in range(1, epochs + 1):
            # --- Training phase ---
            self.model.train()
            total_loss = 0.0

            for batch in train_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                optimizer.zero_grad()
                outputs = self.model(**batch)
                loss = outputs.loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)

            # --- Validation phase ---
            self.model.eval()
            correct = 0
            total = 0

            with torch.no_grad():
                for batch in val_loader:
                    batch = {k: v.to(self.device) for k, v in batch.items()}
                    outputs = self.model(**batch)
                    preds = outputs.logits.argmax(dim=-1)
                    correct += (preds == batch['labels']).sum().item()
                    total += batch['labels'].size(0)

            val_acc = correct / total if total > 0 else 0.0
            print(f"[Classifier] Epoch {epoch}/{epochs} — loss: {avg_loss:.4f}  val_acc: {val_acc:.4f}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                if save_path:
                    self.save(save_path)
                    print(f"[Classifier] Saved best model to {save_path}")

        print(f"[Classifier] Training complete. Best val accuracy: {best_val_acc:.4f}")

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self, dataset: 'NarrativeDataset') -> dict:
        """
        Evaluate on a dataset. Returns dict with accuracy and per-class metrics.
        """
        loader = DataLoader(dataset, batch_size=32, shuffle=False)
        self.model.eval()

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                preds = outputs.logits.argmax(dim=-1).cpu().numpy()
                labels = batch['labels'].cpu().numpy()
                all_preds.extend(preds.tolist())
                all_labels.extend(labels.tolist())

        accuracy = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)

        result = {'accuracy': round(accuracy, 4)}

        if SKLEARN_AVAILABLE and self.inv_label_map:
            target_names = [self.inv_label_map.get(i, str(i)) for i in sorted(self.inv_label_map.keys())]
            try:
                report = classification_report(
                    all_labels, all_preds,
                    target_names=target_names,
                    output_dict=True,
                    zero_division=0,
                )
                result['classification_report'] = report
            except Exception as e:
                result['classification_report_error'] = str(e)

        return result

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, text: str) -> Tuple[str, float]:
        """
        Predict top-level category for a single narrative text.
        Returns (predicted_label_str, confidence_float).
        """
        self.model.eval()
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=256,
            return_tensors='pt',
        )
        encoding = {k: v.to(self.device) for k, v in encoding.items()}

        with torch.no_grad():
            outputs = self.model(**encoding)
            probs = torch.softmax(outputs.logits, dim=-1)[0]
            pred_idx = probs.argmax().item()
            confidence = probs[pred_idx].item()

        label = self.inv_label_map.get(pred_idx, str(pred_idx))
        return label, round(confidence, 4)

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    def save(self, path: str):
        """Save model weights and label map (JSON sidecar) to directory."""
        save_dir = Path(path)
        save_dir.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(str(save_dir))
        self.tokenizer.save_pretrained(str(save_dir))

        label_map_path = save_dir / 'label_map.json'
        with open(label_map_path, 'w') as f:
            json.dump(self.label_map, f, indent=2)

        print(f"[Classifier] Model and label map saved to {save_dir}")

    def load(self, path: str):
        """Load model weights and label map from directory."""
        load_dir = Path(path)

        self.model = DistilBertForSequenceClassification.from_pretrained(str(load_dir))
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(str(load_dir))
        self.model.to(self.device)

        label_map_path = load_dir / 'label_map.json'
        if label_map_path.exists():
            with open(label_map_path) as f:
                self.label_map = json.load(f)
            self.inv_label_map = {v: k for k, v in self.label_map.items()}

        print(f"[Classifier] Model loaded from {load_dir}")


# ---------------------------------------------------------------------------
# Demo setup
# ---------------------------------------------------------------------------

def setup_demo(df: pd.DataFrame) -> 'NTSBClassifier':
    """
    Prepare a small demo run (200 samples) and initialize classifier.
    Prints data split sizes and label map.
    Returns the classifier instance WITHOUT training.
    """
    if not TORCH_AVAILABLE:
        print("[Classifier] torch/transformers not installed — classifier demo unavailable.")
        print("  Install with: pip install torch transformers")
        return None

    print("\n--- Transformer Classifier Setup Demo ---")
    print(f"  Model: distilbert-base-uncased")
    print(f"  Categories: {LABEL_COLS}")

    try:
        clf = NTSBClassifier(num_labels=len(LABEL_COLS))
        train_ds, val_ds, test_ds, label_map = clf.prepare_data(
            df,
            text_col='narr_clean',
            label_col='top_category',
            max_samples=200,
        )

        print(f"\n  Label map: {label_map}")
        print(f"  Train size: {len(train_ds)}")
        print(f"  Val size:   {len(val_ds)}")
        print(f"  Test size:  {len(test_ds)}")
        print(
            "\n  To run full training, call:\n"
            "    clf.train(train_ds, val_ds, epochs=3, batch_size=16, save_path='outputs/model')\n"
            "  Then evaluate with:\n"
            "    results = clf.evaluate(test_ds)"
        )
        return clf

    except Exception as e:
        print(f"[Classifier] Demo setup failed: {e}")
        return None
