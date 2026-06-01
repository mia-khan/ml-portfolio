import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import json

# Load test data
with open('data/processed/test_qa_pairs.json', 'r') as f:
    test_data = json.load(f)

# Load your already trained model
checkpoint = torch.load('models/best_cooking_bert_qa.pt', weights_only=False)
tokenizer = checkpoint['tokenizer']
model = AutoModelForQuestionAnswering.from_pretrained('distilbert-base-uncased-distilled-squad')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print("✅ Loaded trained model")
print(f"📊 Test samples: {len(test_data)}")

# Quick evaluation
correct = 0
total = min(20, len(test_data))  # Test on 20 examples

for qa in test_data[:total]:
    inputs = tokenizer(qa['question'], qa['context'], return_tensors='pt', truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
        start = torch.argmax(outputs.start_logits)
        end = torch.argmax(outputs.end_logits)
        pred = tokenizer.decode(inputs['input_ids'][0][start:end + 1], skip_special_tokens=True)

    if qa['answer'].lower() in pred.lower() or pred.lower() in qa['answer'].lower():
        correct += 1

accuracy = correct / total
print(f"🎯 Final Test Accuracy: {accuracy:.3f} ({accuracy * 100:.1f}%)")
print("🎉 Your BERT fine-tuning is complete!")