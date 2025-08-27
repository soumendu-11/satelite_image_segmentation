# =========================
# utils.py
# =========================

import torch
import torch.nn as nn
import plotly.graph_objects as go
import plotly.express as px

from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from transformers import DataCollatorWithPadding, Trainer, TrainerCallback


# =========================
# Tokenization
# =========================
def tokenize_function(examples, tokenizer, max_length):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=max_length
    )


# =========================
# Compute class weights
# =========================
def get_class_weights(train_df):
    classes = train_df["label"].unique()
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=train_df["label"].values
    )
    return torch.tensor(class_weights, dtype=torch.float), classes


# =========================
# Metrics
# =========================
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary"
    )
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}


# =========================
# Custom Trainer with Weighted Loss
# =========================
class WeightedTrainer(Trainer):
    def __init__(self, class_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        loss_fct = nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))
        loss = loss_fct(
            logits.view(-1, self.model.config.num_labels),
            labels.view(-1)
        )
        return (loss, outputs) if return_outputs else loss


# =========================
# Callback to store metrics
# =========================
class MetricsCallback(TrainerCallback):
    def __init__(self):
        self.train_loss = []
        self.eval_loss = []
        self.eval_accuracy = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        if "loss" in logs:
            self.train_loss.append(logs["loss"])
        if "eval_loss" in logs:
            self.eval_loss.append(logs["eval_loss"])
        if "eval_accuracy" in logs:
            self.eval_accuracy.append(logs["eval_accuracy"])


# =========================
# Plot curves
# =========================
def plot_training_curves(callback):
    epochs = list(range(1, len(callback.eval_loss) + 1))

    fig_loss = go.Figure()
    fig_loss.add_trace(
        go.Scatter(x=list(range(1, len(callback.train_loss) + 1)), y=callback.train_loss, mode="lines+markers", name="Train Loss")
    )
    fig_loss.add_trace(
        go.Scatter(x=epochs, y=callback.eval_loss, mode="lines+markers", name="Validation Loss")
    )
    fig_loss.update_layout(title="Training vs Validation Loss", xaxis_title="Epochs", yaxis_title="Loss")
    fig_loss.show()

    fig_acc = go.Figure()
    fig_acc.add_trace(
        go.Scatter(x=epochs, y=callback.eval_accuracy, mode="lines+markers", name="Validation Accuracy")
    )
    fig_acc.update_layout(title="Validation Accuracy per Epoch", xaxis_title="Epochs", yaxis_title="Accuracy")
    fig_acc.show()


# =========================
# Confusion Matrix
# =========================
def plot_confusion_matrix(y_true, y_pred, classes, precision, recall, f1):
    cm = confusion_matrix(y_true, y_pred)
    fig_cm = px.imshow(
        cm,
        text_auto=True,
        color_continuous_scale="Blues",
        labels=dict(x="Predicted", y="True", color="Count"),
        x=list(classes),
        y=list(classes),
    )
    fig_cm.update_layout(title=f"Confusion Matrix<br>Precision={precision:.2f}, Recall={recall:.2f}, F1={f1:.2f}")
    fig_cm.show()
