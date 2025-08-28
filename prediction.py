# =========================
# prediction.py
# =========================

import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
from prep_utils import clean_text
import time
import json

# =========================
# Config
# =========================
TEST_CSV = "test.csv"
BASE_MODEL = "distilbert-base-uncased"
ADAPTER_DIR = "classification_lora-adapter"  # your LoRA folder
MAX_SEQ_LENGTH = 512
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# Load label mapping
# =========================
with open("label_mapping.json") as f:
    label_mapping = json.load(f)

label2id = {k: int(v) for k, v in label_mapping["label2id"].items()}
id2label = {int(k): v for k, v in label_mapping["id2label"].items()}

# Detect label type (int or str)
sample_label = list(id2label.values())[0]
label_is_int = isinstance(sample_label, int)

print(f"Detected label type: {'int' if label_is_int else 'str'}")

# =========================
# Load test data
# =========================
test_df = pd.read_csv(TEST_CSV)
test_df['Cleaned_Text'] = test_df['text'].apply(clean_text)

# =========================
# Load tokenizer and model
# =========================
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
base_model = AutoModelForSequenceClassification.from_pretrained(
    BASE_MODEL,
    num_labels=len(id2label)
)
model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
model.to(DEVICE)
model.eval()

# =========================
# Prediction function
# =========================
def predict_text(text):
    inputs = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=MAX_SEQ_LENGTH,
        return_tensors="pt"
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = F.softmax(logits, dim=1)
        pred_class_id = torch.argmax(probs, dim=1).item()
        confidence = probs[0, pred_class_id].item()
    
    # Convert prediction to string or int depending on mapping
    pred_label = id2label[pred_class_id]
    if label_is_int:  # ensure numeric output if mapping is int
        pred_label = int(pred_label)

    return pred_label, confidence

# =========================
# Predict all rows
# =========================
pred_labels = []
confidences = []
start_time = time.time()

for text in test_df['Cleaned_Text']:
    label, conf = predict_text(text)
    pred_labels.append(label)
    confidences.append(conf)

end_time = time.time()
print(f"Inference done for {len(test_df)} rows in {end_time - start_time:.2f} sec")

# =========================
# Save results
# =========================
test_df['predicted_label'] = pred_labels
test_df['confidence'] = confidences
test_df.to_csv("test_predictions_with_labels.csv", index=False)
print("Predictions saved to 'test_predictions_with_labels.csv'")
