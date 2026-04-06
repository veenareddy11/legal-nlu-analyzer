import gradio as gr
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import json

# 1. Load the model and tokenizer from the current folder
model_path = "./" 
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# 2. Load your label map
with open("label_map.json", "r") as f:
    label_map = json.load(f)
# Ensure keys are integers
label_map = {int(k): v for k, v in label_map.items()}

def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    confidences = {label_map[i]: float(probs[0][i]) for i in range(len(label_map))}
    return confidences

demo = gr.Interface(
    fn=predict, 
    inputs=gr.Textbox(lines=5, label="Input Legal Text"),
    outputs=gr.Label(num_top_classes=3),
    title="Legal-NLU: AI Contract Analyzer",
    examples=["This Agreement is governed by the laws of New York."]
)

demo.launch()