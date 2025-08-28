# =========================
# utils.py
# =========================

import torch
import torch.nn as nn
import plotly.graph_objects as go
import plotly.express as px
import numpy as np

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

# def plot_confusion_matrix(y_true, y_pred, precision=None, recall=None, f1=None, class_names=None):
#     """
#     Plots a confusion matrix using Plotly with:
#       - Counts
#       - Percentages
#       - True Positives (TP) and False Positives (FP) for each class
#     """
#     if class_names is None:
#         class_names = sorted(list(set(y_true + y_pred)))

#     cm = confusion_matrix(y_true, y_pred, labels=class_names)
#     cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

#     # Compute TP and FP for each class
#     tp_list = np.diag(cm)
#     fp_list = cm.sum(axis=0) - tp_list  # sum of column minus TP

#     # Create text annotations with count, %, TP, FP
#     text = []
#     for i in range(len(class_names)):
#         row = []
#         for j in range(len(class_names)):
#             cell_text = f"{cm[i,j]}<br>{cm_percent[i,j]:.1f}%"
#             if i == j:  # diagonal = TP
#                 cell_text += f"<br>TP: {tp_list[i]}"
#             if i != j:  # off-diagonal = FP for predicted class j
#                 cell_text += f"<br>FP: {fp_list[j]}"
#             row.append(cell_text)
#         text.append(row)

#     fig = go.Figure(
#         data=go.Heatmap(
#             z=cm,
#             x=class_names,
#             y=class_names,
#             text=text,
#             hoverinfo="text",
#             colorscale="Blues",
#             showscale=True
#         )
#     )

#     fig.update_layout(
#         title=f"Confusion Matrix" + (f" | Precision: {precision:.2f}, Recall: {recall:.2f}, F1: {f1:.2f}" if precision is not None else ""),
#         xaxis_title="Predicted label",
#         yaxis_title="True label",
#         yaxis=dict(autorange="reversed")  # reverse y-axis to match sklearn
#     )

#     fig.show()




def plot_confusion_matrix(y_true, y_pred, precision=None, recall=None, f1=None, class_names=None):
    """
    Plots a confusion matrix (binary only) using Plotly with:
      - Counts
      - Percentages
      - TP, FP, FN, TN explicitly written inside each of the 4 boxes
    """
    if class_names is None:
        class_names = sorted(list(set(y_true + y_pred)))

    cm = confusion_matrix(y_true, y_pred, labels=class_names)

    if cm.shape != (2, 2):
        raise ValueError("This visualization is designed for binary classification only (2x2 matrix).")

    # Extract values
    tn, fp, fn, tp = cm.ravel()

    # Create text for each box
    text = [
        [f"TN: {tn}", f"FP: {fp}"],
        [f"FN: {fn}", f"TP: {tp}"]
    ]

    fig = go.Figure(
        data=go.Heatmap(
            z=cm,
            x=["Predicted: " + str(cls) for cls in class_names],
            y=["Actual: " + str(cls) for cls in class_names],
            text=text,
            texttemplate="%{text}",   # <--- this forces numbers to be drawn inside the box
            colorscale="Blues",
            showscale=True
        )
    )

    fig.update_layout(
        title=f"Confusion Matrix"
              + (f" | Precision: {precision:.2f}, Recall: {recall:.2f}, F1: {f1:.2f}" if precision is not None else ""),
        xaxis_title="Predicted label",
        yaxis_title="True label",
        yaxis=dict(autorange="reversed")  # match sklearn
    )

    fig.show()
