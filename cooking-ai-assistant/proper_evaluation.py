"""
Proper BERT QA Evaluation with Academic Metrics
CS 6120 Project - Real Performance Evaluation
"""

import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import json
import re
from collections import Counter
import string


def normalize_answer(s):
    """Normalize answer for evaluation."""

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    """Calculate F1 score between prediction and ground truth."""
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()

    if len(prediction_tokens) == 0 or len(ground_truth_tokens) == 0:
        return int(prediction_tokens == ground_truth_tokens)

    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0

    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)

    return f1


def exact_match_score(prediction, ground_truth):
    """Calculate exact match score."""
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def evaluate_model_properly():
    """Proper evaluation with F1 and EM scores."""
    print("🎯 Running Proper BERT QA Evaluation")
    print("=" * 50)

    # Load test data
    with open('data/processed/test_qa_pairs.json', 'r') as f:
        test_data = json.load(f)

    # Load trained model
    checkpoint = torch.load('models/best_cooking_bert_qa.pt', weights_only=False)
    tokenizer = checkpoint['tokenizer']
    model = AutoModelForQuestionAnswering.from_pretrained('distilbert-base-uncased-distilled-squad')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"✅ Loaded model and {len(test_data)} test samples")

    # Evaluation metrics
    f1_scores = []
    em_scores = []
    predictions = []
    references = []

    # Evaluate on more samples for better statistics
    num_samples = min(50, len(test_data))
    print(f"📊 Evaluating on {num_samples} samples...")

    for i, qa in enumerate(test_data[:num_samples]):
        question = qa['question']
        context = qa['context']
        true_answer = qa['answer']

        # Tokenize with proper settings
        inputs = tokenizer(
            question,
            context,
            add_special_tokens=True,
            return_tensors='pt',
            max_length=512,  # Increased context length
            truncation=True,
            padding=True,
            return_offsets_mapping=True if hasattr(tokenizer, 'return_offsets_mapping') else False
        )

        # Get model prediction
        with torch.no_grad():
            outputs = model(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask']
            )

            start_logits = outputs.start_logits
            end_logits = outputs.end_logits

            # Get best start and end positions
            start_idx = torch.argmax(start_logits, dim=1).item()
            end_idx = torch.argmax(end_logits, dim=1).item()

            # Ensure end >= start
            if end_idx < start_idx:
                end_idx = start_idx

            # Extract prediction
            if end_idx < len(inputs['input_ids'][0]):
                predicted_tokens = inputs['input_ids'][0][start_idx:end_idx + 1]
                predicted_answer = tokenizer.decode(predicted_tokens, skip_special_tokens=True)
            else:
                predicted_answer = ""

        # Handle empty predictions
        if not predicted_answer.strip():
            predicted_answer = "[No Answer Found]"

        # Calculate metrics
        f1 = f1_score(predicted_answer, true_answer)
        em = exact_match_score(predicted_answer, true_answer)

        f1_scores.append(f1)
        em_scores.append(em)
        predictions.append(predicted_answer)
        references.append(true_answer)

        # Show progress for first few examples
        if i < 5:
            print(f"\n📝 Example {i + 1}:")
            print(f"   Q: {question}")
            print(f"   True: {true_answer}")
            print(f"   Pred: {predicted_answer}")
            print(f"   F1: {f1:.3f}, EM: {em}")

    # Calculate final metrics
    avg_f1 = sum(f1_scores) / len(f1_scores)
    avg_em = sum(em_scores) / len(em_scores)

    print(f"\n🎯 FINAL RESULTS:")
    print(f"   📊 F1 Score: {avg_f1:.3f} ({avg_f1 * 100:.1f}%)")
    print(f"   📊 Exact Match: {avg_em:.3f} ({avg_em * 100:.1f}%)")
    print(f"   📊 Samples Evaluated: {num_samples}")

    # Analyze prediction quality
    non_empty_preds = sum(1 for p in predictions if p.strip() and p != "[No Answer Found]")
    print(f"   📊 Non-empty Predictions: {non_empty_preds}/{num_samples} ({non_empty_preds / num_samples * 100:.1f}%)")

    # Performance interpretation
    print(f"\n📈 Performance Analysis:")
    if avg_f1 >= 0.6:
        print(f"   🎉 EXCELLENT F1 Score! Your model performs very well!")
    elif avg_f1 >= 0.4:
        print(f"   ✅ GOOD F1 Score! Your model learned the cooking domain!")
    elif avg_f1 >= 0.2:
        print(f"   📈 DECENT F1 Score! Model shows learning, room for improvement!")
    else:
        print(f"   📚 Model needs more training or data adjustment!")

    # Show best and worst predictions
    f1_with_idx = [(f1_scores[i], i) for i in range(len(f1_scores))]
    f1_with_idx.sort(reverse=True)

    print(f"\n🏆 Best Prediction (F1: {f1_with_idx[0][0]:.3f}):")
    best_idx = f1_with_idx[0][1]
    print(f"   Q: {test_data[best_idx]['question']}")
    print(f"   A: {predictions[best_idx]}")

    print(f"\n🔧 Challenging Example (F1: {f1_with_idx[-1][0]:.3f}):")
    worst_idx = f1_with_idx[-1][1]
    print(f"   Q: {test_data[worst_idx]['question']}")
    print(f"   Expected: {references[worst_idx]}")
    print(f"   Predicted: {predictions[worst_idx]}")

    # Save detailed results
    results = {
        'f1_score': avg_f1,
        'exact_match': avg_em,
        'num_samples': num_samples,
        'predictions': predictions[:10],  # Save first 10 for inspection
        'references': references[:10],
        'individual_f1_scores': f1_scores[:10]
    }

    with open('results/detailed_evaluation.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n💾 Detailed results saved to: results/detailed_evaluation.json")

    return avg_f1, avg_em


def compare_with_baseline():
    """Compare with a simple baseline."""
    print(f"\n🔄 Comparing with Random Baseline...")

    # Load test data for baseline
    with open('data/processed/test_qa_pairs.json', 'r') as f:
        test_data = json.load(f)

    # Simple baseline: return first few words of context
    baseline_f1s = []
    for qa in test_data[:20]:
        context_words = qa['context'].split()[:5]  # First 5 words
        baseline_answer = ' '.join(context_words)
        f1 = f1_score(baseline_answer, qa['answer'])
        baseline_f1s.append(f1)

    baseline_avg = sum(baseline_f1s) / len(baseline_f1s)
    print(f"   📊 Random Baseline F1: {baseline_avg:.3f}")

    return baseline_avg


if __name__ == "__main__":
    # Run proper evaluation
    f1, em = evaluate_model_properly()

    # Compare with baseline
    baseline = compare_with_baseline()

    # Final summary
    improvement = f1 - baseline
    print(f"\n🎊 FINAL SUMMARY:")
    print(f"   🤖 Your BERT Model F1: {f1:.3f}")
    print(f"   📊 Baseline F1: {baseline:.3f}")
    print(f"   🚀 Improvement: +{improvement:.3f} ({improvement / baseline * 100:+.1f}%)")

    if improvement > 0.1:
        print(f"   🎉 Significant improvement! Your BERT fine-tuning worked!")
    elif improvement > 0:
        print(f"   ✅ Positive improvement! Model learned from fine-tuning!")
    else:
        print(f"   📚 Model needs more training or data adjustment!")