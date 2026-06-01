# Cooking AI Assistant — Fine-Tuned BERT Question Answering System

Fine-tuned DistilBERT on a cooking domain QA dataset, achieving a 1,887% improvement
over the baseline through real gradient descent on 795 curated question-answer pairs.
Includes a rigorous evaluation pipeline (5-fold cross-validation, 13-configuration
ablation study, bias-variance analysis) and a live Streamlit demo with meal planning
and grocery list generation.

## Results

| Metric | Score |
|--------|-------|
| F1 Score (best checkpoint) | 58.4% |
| F1 Score (5-fold CV mean) | 58.6% ± 2.2% |
| Baseline F1 (keyword matching) | ~2.9% |
| Improvement over baseline | **1,887%** |

## Why This Project

Off-the-shelf DistilBERT has no cooking-domain knowledge — it was pre-trained on
Wikipedia and BookCorpus. The goal is to demonstrate that domain-specific fine-tuning
on a small, carefully curated dataset can produce a genuinely useful QA system, and
to rigorously characterize *how well* it works through systematic evaluation rather
than reporting a single headline number.

## Architecture

DistilBERT (distilbert-base-uncased) with a span-prediction head fine-tuned for
extractive QA. Given a question and a recipe passage, the model predicts start and
end token positions of the answer span within the passage.

```
Input:  [CLS] What temperature should I bake salmon? [SEP] Bake at 400°F for 12-15 minutes. [SEP]
Output: span → "400°F for 12-15 minutes"
```

Model size: 66.4M parameters  
Task format: SQuAD-style extractive QA  
Training data: RecipeNLG dataset, filtered and converted to QA pairs

## Evaluation Methodology

### 5-Fold Cross-Validation
The dataset is split into 5 folds. Each fold is held out once as a test set while
the model trains on the remaining 4. This gives a stable estimate of generalization
performance (58.6% ± 2.2% F1) rather than a single lucky/unlucky split.

### 13-Configuration Ablation Study
Systematic sweep over:
- Learning rate: {1e-5, 2e-5, 3e-5}
- Batch size: {8, 16, 32}
- Number of training epochs: {2, 3, 5}

Results surface which hyperparameter choices matter most and which are robust —
epoch count had the largest impact; batch size had the least.

### Bias-Variance Analysis
Training vs. validation F1 curves across epochs reveal the classic bias-variance
tradeoff: early epochs underfit (high bias), later epochs overfit (high variance).
The best checkpoint is selected at the inflection point before validation F1 degrades.

## Training Setup

```python
model = DistilBertForQuestionAnswering.from_pretrained('distilbert-base-uncased')
optimizer = AdamW(model.parameters(), lr=2e-5)
loss = CrossEntropyLoss on (start_logits, end_logits)
```

Training runs for 3 epochs on 795 QA pairs. The best checkpoint (by validation F1)
is saved and used for all downstream evaluation.

## Demo

The Streamlit app lets you ask any cooking question and get an extracted answer from
the relevant recipe passage. Additional features:
- Meal plan generation: suggest a weekly menu based on dietary preferences
- Grocery list generation: extract ingredients from a set of recipes

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
streamlit run demo_app.py
```

## Loading the Model

```python
import torch
checkpoint = torch.load('best_cooking_bert_qa.pt', weights_only=False)
model = checkpoint['model']
tokenizer = checkpoint['tokenizer']
```

Pre-trained model download: [best_cooking_bert_qa.pt (256MB)](https://drive.google.com/file/d/15heU5tsyAs6E_qJntB0do-iN1pw4oEEi/view?usp=sharing)

## Files

| File | Description |
|------|-------------|
| `train_bert_qa.py` | Fine-tuning script with early stopping and checkpoint saving |
| `run_training.py` | Training orchestrator with config management |
| `proper_evaluation.py` | 5-fold CV + ablation study |
| `enhanced_evaluation.py` | Bias-variance analysis and calibration plots |
| `demo_app.py` | Streamlit demo interface |
| `generate_qa_pairs.py` | Converts RecipeNLG recipes into SQuAD-format QA pairs |
| `evaluation_report.md` | Full evaluation results and analysis |

## Technologies

Python · PyTorch · HuggingFace Transformers · DistilBERT · Streamlit · spaCy · NLTK
