# =========================
# prediction_lora_with_time.py
# =========================

import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
from prep_utils import clean_text
import time

# =========================
# Config
# =========================
TEST_CSV = "test.csv"
BASE_MODEL = "distilbert-base-uncased"
ADAPTER_DIR = "classification_lora-adapter"  # your local LoRA folder
MAX_SEQ_LENGTH = 512
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# Load test data
# =========================
test_df = pd.read_csv(TEST_CSV)
test_df['Cleaned_Text'] = test_df['text'].apply(clean_text)

# =========================
# Load tokenizer and base model
# =========================
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
base_model = AutoModelForSequenceClassification.from_pretrained(
    BASE_MODEL,
    num_labels=2  # set correct number of labels
)

# Load LoRA adapter
model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
model.to(DEVICE)
model.eval()

# =========================
# Prediction function with timing
# =========================
def predict_text(text):
    start_time = time.time()  # start timer

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
        pred_class = torch.argmax(probs, dim=1).item()
        confidence = probs[0, pred_class].item()

    end_time = time.time()  # end timer
    response_time = end_time - start_time
    return pred_class, confidence, response_time

# =========================
# Predict all rows
# =========================
predictions = []
confidences = []
response_times = []

for text in test_df['Cleaned_Text']:
    pred, conf, resp_time = predict_text(text)
    predictions.append(pred)
    confidences.append(conf)
    response_times.append(resp_time)

test_df['predicted_label'] = predictions
test_df['confidence'] = confidences
test_df['response_time_sec'] = response_times

# =========================
# Save predictions
# =========================
test_df.to_csv("test_predictions_with_time.csv", index=False)
print("\nPredictions with response time saved to 'test_predictions_with_time.csv'")
