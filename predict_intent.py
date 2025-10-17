import argparse
import torch
import torch.nn.functional as F
import joblib
from transformers import BertTokenizer, BertForSequenceClassification

MODEL_DIR = "./bert_intent_model"
LABEL_ENCODE_PATH = "label_encoder.pkl"


tokenizer = BertTokenizer.from_pretrained(MODEL_DIR)
model = BertForSequenceClassification.from_pretrained(MODEL_DIR)
label_encoder = joblib.load(LABEL_ENCODE_PATH)


model.eval()

def predict_intent(text: str):
    "predicts intent for a given text using fine-tuned BERT."
    inputs = tokenizer(
        text,
        return_tensors ="pt",
        truncation=True,
        padding=True,
        max_length=64
    )
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = F.softmax(logits, dim = 1)

    pred_idx = torch.argmax(probs, dim=1).item()
    pred_intent = label_encoder.inverse_transform([pred_idx])[0]
    confidence = probs[0][pred_idx].item()
    return pred_intent, confidence  

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BERT Intent Classification CLI")
    parser.add_argument("--text", type=str, required=True, help="Input text for intent prediction")
    args = parser.parse_args()

    intent, confidence = predict_intent(args.text)
    print(f"Predicted intent: {intent}")
    print(f"confidence: {confidence}")