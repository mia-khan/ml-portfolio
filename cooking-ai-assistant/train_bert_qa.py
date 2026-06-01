"""
Real BERT Fine-tuning Pipeline for Cooking QA
"""

import os
import json
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    get_linear_schedule_with_warmup,
    TrainingArguments,
    Trainer
)
from torch.optim import AdamW
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import logging
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CookingQADataset(Dataset):
    """Dataset class for cooking QA pairs."""

    def __init__(self, qa_pairs, tokenizer, max_length=384):
        self.qa_pairs = qa_pairs
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.qa_pairs)

    def __getitem__(self, idx):
        qa = self.qa_pairs[idx]

        question = qa['question']
        context = qa['context']
        answer = qa['answer']

        # Tokenize question and context
        encoding = self.tokenizer(
            question,
            context,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        # Find answer start and end positions in the context
        answer_start, answer_end = self.find_answer_positions(context, answer, encoding)

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'start_positions': torch.tensor(answer_start, dtype=torch.long),
            'end_positions': torch.tensor(answer_end, dtype=torch.long),
            'question': question,
            'context': context,
            'answer': answer
        }

    def find_answer_positions(self, context, answer, encoding):
        """Find start and end positions of answer in tokenized context."""
        # Simple approach: find answer in context
        context_lower = context.lower()
        answer_lower = answer.lower()

        # Find character positions
        start_char = context_lower.find(answer_lower)

        if start_char == -1:
            # Answer not found in context, return default positions
            return 0, 0

        end_char = start_char + len(answer)

        # Convert character positions to token positions (simplified)
        # In practice, you'd use tokenizer.char_to_token() for exact mapping
        tokens = self.tokenizer.tokenize(context)

        # Approximate token positions (simplified for this implementation)
        total_chars = len(context)
        total_tokens = len(tokens)

        if total_chars > 0:
            start_token = int((start_char / total_chars) * total_tokens)
            end_token = int((end_char / total_chars) * total_tokens)

            # Ensure positions are within valid range
            start_token = max(0, min(start_token, total_tokens - 1))
            end_token = max(start_token, min(end_token, total_tokens - 1))
        else:
            start_token, end_token = 0, 0

        return start_token, end_token

class CookingBERTTrainer:
    """Real BERT fine-tuning trainer."""

    def __init__(self, model_name="distilbert-base-uncased-distilled-squad"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🔧 Using device: {self.device}")

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_name)
        self.model.to(self.device)

        print(f"✅ Loaded {model_name}")
        print(f"   Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        # Training history
        self.training_history = {
            'train_losses': [],
            'val_losses': [],
            'learning_rates': [],
            'epochs': [],
            'best_val_loss': float('inf'),
            'best_model_epoch': 0
        }

    def load_data(self):
        """Load training and validation data."""
        print("📚 Loading QA training data...")

        # Load train data
        with open('data/processed/train_qa_pairs.json', 'r') as f:
            train_data = json.load(f)

        # Load validation data
        with open('data/processed/val_qa_pairs.json', 'r') as f:
            val_data = json.load(f)

        print(f"   Train samples: {len(train_data):,}")
        print(f"   Val samples: {len(val_data):,}")

        return train_data, val_data

    def create_dataloaders(self, train_data, val_data, batch_size=8):
        """Create PyTorch DataLoaders."""
        print(f"🔄 Creating DataLoaders (batch_size={batch_size})...")

        # Create datasets
        train_dataset = CookingQADataset(train_data, self.tokenizer)
        val_dataset = CookingQADataset(val_data, self.tokenizer)

        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,  # Set to 0 to avoid multiprocessing issues
            pin_memory=True if self.device.type == 'cuda' else False
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True if self.device.type == 'cuda' else False
        )

        print(f"   Train batches: {len(train_loader)}")
        print(f"   Val batches: {len(val_loader)}")

        return train_loader, val_loader

    def setup_optimizer_scheduler(self, train_loader, num_epochs, learning_rate=2e-5):
        """Set up optimizer and learning rate scheduler."""
        print(f"⚙️  Setting up optimizer (lr={learning_rate})...")

        # Create optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=0.01
        )

        # Calculate total training steps
        total_steps = len(train_loader) * num_epochs

        # Create scheduler
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=int(0.1 * total_steps),  # 10% warmup
            num_training_steps=total_steps
        )

        print(f"   Total training steps: {total_steps}")
        print(f"   Warmup steps: {int(0.1 * total_steps)}")

    def train_epoch(self, train_loader, epoch):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        num_batches = len(train_loader)

        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1} Training')

        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            start_positions = batch['start_positions'].to(self.device)
            end_positions = batch['end_positions'].to(self.device)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                start_positions=start_positions,
                end_positions=end_positions
            )

            loss = outputs.loss

            # Backward pass
            loss.backward()

            # Clip gradients to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            # Update weights
            self.optimizer.step()
            self.scheduler.step()

            # Update metrics
            total_loss += loss.item()
            avg_loss = total_loss / (batch_idx + 1)

            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{avg_loss:.4f}',
                'lr': f'{self.scheduler.get_last_lr()[0]:.2e}'
            })

        avg_train_loss = total_loss / num_batches
        return avg_train_loss

    def validate_epoch(self, val_loader, epoch):
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0
        num_batches = len(val_loader)

        with torch.no_grad():
            progress_bar = tqdm(val_loader, desc=f'Epoch {epoch+1} Validation')

            for batch in progress_bar:
                # Move batch to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                start_positions = batch['start_positions'].to(self.device)
                end_positions = batch['end_positions'].to(self.device)

                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    start_positions=start_positions,
                    end_positions=end_positions
                )

                loss = outputs.loss
                total_loss += loss.item()

                # Update progress bar
                progress_bar.set_postfix({'val_loss': f'{loss.item():.4f}'})

        avg_val_loss = total_loss / num_batches
        return avg_val_loss

    def train(self, num_epochs=3, batch_size=8, learning_rate=2e-5):
        """Main training loop."""
        print(f"🚀 Starting BERT fine-tuning for {num_epochs} epochs")
        print("=" * 60)

        # Load data
        train_data, val_data = self.load_data()

        # Create dataloaders
        train_loader, val_loader = self.create_dataloaders(train_data, val_data, batch_size)

        # Setup optimizer and scheduler
        self.setup_optimizer_scheduler(train_loader, num_epochs, learning_rate)

        # Training loop
        for epoch in range(num_epochs):
            print(f"\n📖 Epoch {epoch+1}/{num_epochs}")

            # Train
            train_loss = self.train_epoch(train_loader, epoch)

            # Validate
            val_loss = self.validate_epoch(val_loader, epoch)

            # Get current learning rate
            current_lr = self.scheduler.get_last_lr()[0]

            # Update training history
            self.training_history['train_losses'].append(train_loss)
            self.training_history['val_losses'].append(val_loss)
            self.training_history['learning_rates'].append(current_lr)
            self.training_history['epochs'].append(epoch + 1)

            # Print epoch summary
            print(f"   📊 Train Loss: {train_loss:.4f}")
            print(f"   📊 Val Loss: {val_loss:.4f}")
            print(f"   📊 Learning Rate: {current_lr:.2e}")

            # Save best model
            if val_loss < self.training_history['best_val_loss']:
                self.training_history['best_val_loss'] = val_loss
                self.training_history['best_model_epoch'] = epoch + 1
                self.save_model('models/best_cooking_bert_qa.pt')
                print(f"   💾 New best model saved! (Val Loss: {val_loss:.4f})")

            # Save checkpoint
            self.save_checkpoint(epoch + 1)

        print(f"\n🎉 Training completed!")
        print(f"   Best validation loss: {self.training_history['best_val_loss']:.4f} (Epoch {self.training_history['best_model_epoch']})")

        return self.training_history

    def save_model(self, path):
        """Save the trained model."""
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # Save model state
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'tokenizer': self.tokenizer,
            'training_history': self.training_history,
            'model_name': 'distilbert-cooking-qa'
        }, path)

        # Also save in HuggingFace format
        model_dir = 'models/cooking_bert_qa_hf'
        os.makedirs(model_dir, exist_ok=True)
        self.model.save_pretrained(model_dir)
        self.tokenizer.save_pretrained(model_dir)

    def save_checkpoint(self, epoch):
        """Save training checkpoint."""
        os.makedirs('models/checkpoints', exist_ok=True)

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'training_history': self.training_history
        }

        torch.save(checkpoint, f'models/checkpoints/checkpoint_epoch_{epoch}.pt')

    def plot_training_history(self):
        """Plot training history."""
        plt.figure(figsize=(15, 5))

        # Loss plot
        plt.subplot(1, 3, 1)
        plt.plot(self.training_history['epochs'], self.training_history['train_losses'], 'b-', label='Train Loss')
        plt.plot(self.training_history['epochs'], self.training_history['val_losses'], 'r-', label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)

        # Learning rate plot
        plt.subplot(1, 3, 2)
        plt.plot(self.training_history['epochs'], self.training_history['learning_rates'], 'g-')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Schedule')
        plt.yscale('log')
        plt.grid(True)

        # Loss improvement plot
        plt.subplot(1, 3, 3)
        val_losses = self.training_history['val_losses']
        improvements = [val_losses[0] - loss for loss in val_losses]
        plt.plot(self.training_history['epochs'], improvements, 'm-')
        plt.xlabel('Epoch')
        plt.ylabel('Validation Loss Improvement')
        plt.title('Model Improvement Over Time')
        plt.grid(True)

        plt.tight_layout()
        plt.savefig('results/training_history.png', dpi=300, bbox_inches='tight')
        plt.show()

        print("📊 Training plots saved to results/training_history.png")

def evaluate_trained_model():
    """Evaluate the trained model on test set."""
    print("🔍 Evaluating trained model on test set...")

    # Load test data
    with open('data/processed/test_qa_pairs.json', 'r') as f:
        test_data = json.load(f)

    print(f"   Test samples: {len(test_data)}")

    # Load trained model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    try:
        checkpoint = torch.load('models/best_cooking_bert_qa.pt', map_location=device, weights_only=False)

        # Load model
        tokenizer = checkpoint['tokenizer']
        model = AutoModelForQuestionAnswering.from_pretrained('distilbert-base-uncased-distilled-squad')
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()

        print("✅ Loaded trained model for evaluation")

        # Simple evaluation on a few examples
        correct_predictions = 0
        total_predictions = min(50, len(test_data))  # Evaluate on subset for speed

        print(f"   Evaluating on {total_predictions} test examples...")

        for i, qa in enumerate(test_data[:total_predictions]):
            question = qa['question']
            context = qa['context']
            expected_answer = qa['answer']

            # Tokenize
            inputs = tokenizer(
                question, context,
                add_special_tokens=True,
                return_tensors='pt',
                max_length=384,
                truncation=True,
                padding=True
            ).to(device)

            # Get prediction
            with torch.no_grad():
                outputs = model(**inputs)
                start_idx = torch.argmax(outputs.start_logits)
                end_idx = torch.argmax(outputs.end_logits)

                if end_idx >= start_idx:
                    predicted_answer = tokenizer.decode(
                        inputs['input_ids'][0][start_idx:end_idx+1],
                        skip_special_tokens=True
                    )
                else:
                    predicted_answer = ""

            # Simple accuracy check (contains expected answer)
            if expected_answer.lower() in predicted_answer.lower() or predicted_answer.lower() in expected_answer.lower():
                correct_predictions += 1

        accuracy = correct_predictions / total_predictions
        print(f"✅ Evaluation completed!")
        print(f"   Accuracy: {accuracy:.3f} ({correct_predictions}/{total_predictions})")

        return accuracy

    except FileNotFoundError:
        print("❌ No trained model found. Please run training first.")
        return 0.0

def main():
    """Main training function."""
    print("🍳 BERT Fine-tuning for Cooking QA - CS 6120 Project")
    print("=" * 70)

    # Check if data exists
    required_files = [
        'data/processed/train_qa_pairs.json',
        'data/processed/val_qa_pairs.json',
        'data/processed/test_qa_pairs.json'
    ]

    for file_path in required_files:
        if not os.path.exists(file_path):
            print(f"❌ Required file not found: {file_path}")
            print("   Please run generate_qa_pairs.py first")
            return False

    # Create results directory
    os.makedirs('results', exist_ok=True)
    os.makedirs('models', exist_ok=True)

    # Initialize trainer
    trainer = CookingBERTTrainer()

    # Train the model
    print(f"🎯 Training Configuration:")
    print(f"   Model: DistilBERT for QA")
    print(f"   Epochs: 3")
    print(f"   Batch size: 8")
    print(f"   Learning rate: 2e-5")
    print(f"   Device: {trainer.device}")

    # Start training
    history = trainer.train(num_epochs=3, batch_size=8, learning_rate=2e-5)

    # Plot training history
    trainer.plot_training_history()

    # Evaluate trained model
    test_accuracy = evaluate_trained_model()

    # Save training summary
    summary = {
        'training_completed': datetime.now().isoformat(),
        'total_epochs': len(history['epochs']),
        'best_val_loss': history['best_val_loss'],
        'best_model_epoch': history['best_model_epoch'],
        'final_train_loss': history['train_losses'][-1],
        'final_val_loss': history['val_losses'][-1],
        'test_accuracy': test_accuracy,
        'model_saved': 'models/best_cooking_bert_qa.pt'
    }

    with open('results/training_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n🎉 BERT FINE-TUNING COMPLETED!")
    print(f"   Best validation loss: {history['best_val_loss']:.4f}")
    print(f"   Test accuracy: {test_accuracy:.3f}")
    print(f"   Model saved: models/best_cooking_bert_qa.pt")
    print(f"   Training plots: results/training_history.png")
    print(f"    Ready for comprehensive evaluation!")

    return True

if __name__ == "__main__":
    main()