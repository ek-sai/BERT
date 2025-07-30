import torch
import numpy as np
from transformers import (
    BertForMaskedLM, BertTokenizer,
    Trainer, TrainingArguments,
    TrainerCallback
)
from torch.utils.data import Dataset, random_split
import os
import json
from datetime import datetime
import matplotlib.pyplot as plt
import random

# CORRECTED Dataset for Next Word Prediction
class NextWordPredictionDataset(Dataset):
    def __init__(self, tokenizer, file_path, block_size=64):  # Smaller block size
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.examples = []

        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # Process each line to create next word prediction examples
        for line in lines:
            line = line.strip()
            # Better filtering
            if (len(line) > 20 and len(line) < 300 and
                len(line.split()) >= 6 and len(line.split()) <= 25 and
                line.replace(' ', '').isascii() and
                not line.startswith(('http', 'www', '@', '#')) and
                any(c.isalpha() for c in line)):

                words = line.split()
                # Create multiple training examples from each sentence
                for i in range(3, len(words)):  # Start from position 3 to have context
                    context = " ".join(words[:i])
                    target_word = words[i]

                    # Only use words that are actual vocabulary
                    if (target_word.lower().isalpha() and
                        len(target_word) > 1 and
                        target_word.lower() not in ['the', 'a', 'an', 'and', 'or', 'but']):  # Skip very common words for now

                        self.examples.append({
                            'context': context,
                            'target': target_word.lower()
                        })

        print(f"Created {len(self.examples)} next-word prediction examples from {len(lines)} lines")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]
        context = example['context']
        target = example['target']

        # Create input with [MASK] at the end for next word prediction
        input_text = context + " " + self.tokenizer.mask_token

        # Tokenize
        encoding = self.tokenizer(
            input_text,
            truncation=True,
            max_length=self.block_size,
            padding='max_length',
            return_tensors='pt'
        )

        # Get target token ID
        target_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(target))
        if target_id:
            target_id = target_id[0]  # Take first token if target is multi-token
        else:
            target_id = self.tokenizer.unk_token_id

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(target_id, dtype=torch.long), # Renamed from target_id
            'context': context,
            'target_word': target
        }

# SIMPLIFIED and CORRECTED Data Collator for Next Word Prediction
class NextWordPredictionCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.mask_token_id = tokenizer.mask_token_id
        self.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        self.ignore_index = -100

    def __call__(self, examples):
        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []

        for ex in examples:
            input_ids = ex['input_ids']
            attention_mask = ex['attention_mask']
            # The 'labels' key now holds the target_id
            target_id = ex['labels'] 

            # Find the position of [MASK] token
            mask_positions = (input_ids == self.mask_token_id).nonzero(as_tuple=True)[0]

            if len(mask_positions) > 0:
                # Create labels - only predict the masked position
                labels = torch.full_like(input_ids, self.ignore_index)
                mask_pos = mask_positions[0]  # First (should be only) mask position
                labels[mask_pos] = target_id

                batch_input_ids.append(input_ids)
                batch_attention_mask.append(attention_mask)
                batch_labels.append(labels)

        if not batch_input_ids:
            # Fallback if no valid examples
            dummy_input = examples[0]['input_ids']
            return {
                'input_ids': torch.stack([ex['input_ids'] for ex in examples]),
                'attention_mask': torch.stack([ex['attention_mask'] for ex in examples]),
                'labels': torch.full((len(examples), dummy_input.size(0)), self.ignore_index)
            }

        return {
            'input_ids': torch.stack(batch_input_ids),
            'attention_mask': torch.stack(batch_attention_mask),
            'labels': torch.stack(batch_labels)
        }

# Training history callback
class HistoryCallback(TrainerCallback):
    def __init__(self, trainer_instance):
        self.trainer_instance = trainer_instance

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            if 'loss' in logs:
                self.trainer_instance.training_history['train_loss'].append(logs['loss'])
            if 'eval_loss' in logs:
                self.trainer_instance.training_history['eval_loss'].append(logs['eval_loss'])
            if 'learning_rate' in logs:
                self.trainer_instance.training_history['learning_rate'].append(logs['learning_rate'])

class NextWordBERTTrainer:
    def __init__(self, model_name="bert-base-uncased"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.model = BertForMaskedLM.from_pretrained(model_name)
        self.tokenizer = BertTokenizer.from_pretrained(model_name)

        # Ensure pad token exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            self.model.resize_token_embeddings(len(self.tokenizer))

        self.model.to(self.device)
        self.training_history = {
            'train_loss': [],
            'eval_loss': [],
            'learning_rate': []
        }

    def prepare_datasets(self, train_file, validation_split=0.2, block_size=64):
        print(f"Loading dataset from {train_file}...")
        dataset = NextWordPredictionDataset(
            tokenizer=self.tokenizer,
            file_path=train_file,
            block_size=block_size
        )

        if len(dataset) == 0:
            raise ValueError("No valid examples found in dataset!")

        train_size = int((1 - validation_split) * len(dataset))
        val_size = len(dataset) - train_size

        print(f"Splitting dataset: {len(dataset)} total examples")
        train_dataset, val_dataset = random_split(
            dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )

        print(f"Train samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")

        return train_dataset, val_dataset

    def setup_training_args(self, output_dir, num_epochs=4, batch_size=32):
        return TrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=True,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=2,
            learning_rate=3e-5,  # Good learning rate for fine-tuning
            weight_decay=0.01,
            warmup_ratio=0.1,
            lr_scheduler_type="linear",
            logging_steps=50,
            eval_steps=150,
            save_steps=300,
            eval_strategy="steps",  # FIXED: Changed from evaluation_strategy to eval_strategy
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            fp16=True,
            dataloader_num_workers=2,
            dataloader_pin_memory=True,
            report_to=None,  # Disable wandb
            logging_dir=f"{output_dir}/logs",
            save_total_limit=2,
        )

    def train_model(self, train_file, output_dir="./bert-nextword-v3", **kwargs):
        os.makedirs(output_dir, exist_ok=True)

        train_dataset, val_dataset = self.prepare_datasets(
            train_file=train_file,
            validation_split=kwargs.get('validation_split', 0.2),
            block_size=kwargs.get('block_size', 64)
        )

        data_collator = NextWordPredictionCollator(tokenizer=self.tokenizer)

        training_args = self.setup_training_args(
            output_dir=output_dir,
            num_epochs=kwargs.get('num_epochs', 4),
            batch_size=kwargs.get('batch_size', 32)
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            callbacks=[HistoryCallback(self)]
        )

        print("Starting Next Word Prediction Training with BERT...")
        print("="*60)

        trainer.train()

        # Save model and tokenizer
        final_path = f"{output_dir}/final_model"
        trainer.save_model(final_path)
        self.tokenizer.save_pretrained(final_path)
        print(f"Model and tokenizer saved to: {final_path}")

        # Save training history
        with open(f"{output_dir}/training_history.json", 'w') as f:
            json.dump(self.training_history, f, indent=2)

        self.plot_training_curves(output_dir)
        return trainer

    def plot_training_curves(self, output_dir):
        if not self.training_history['train_loss']:
            return

        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        plt.plot(self.training_history['train_loss'], label='Training Loss', color='blue')
        if self.training_history['eval_loss']:
            plt.plot(self.training_history['eval_loss'], label='Validation Loss', color='red')
        plt.xlabel('Steps')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 2, 2)
        if self.training_history['learning_rate']:
            plt.plot(self.training_history['learning_rate'], label='Learning Rate', color='green')
        plt.xlabel('Steps')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Schedule')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"{output_dir}/training_curves.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Training curves saved: {output_dir}/training_curves.png")

# CORRECTED Evaluator for Next Word Prediction
class NextWordEvaluator:
    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = BertForMaskedLM.from_pretrained(model_path)
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()

    def predict_next_word(self, context, top_k=10):
        """
        Predict the next word given a context
        """
        # Clean context
        context = context.strip()

        # Create input with [MASK] for next word prediction
        input_text = context + " " + self.tokenizer.mask_token

        # Tokenize
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=64,
            padding=True
        )

        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits

        # Find mask position
        mask_token_id = self.tokenizer.mask_token_id
        mask_positions = (inputs['input_ids'] == mask_token_id).nonzero(as_tuple=True)

        if len(mask_positions[1]) == 0:
            return []

        # Get predictions for the mask position
        mask_pos = mask_positions[1][-1]  # Last mask position
        mask_logits = logits[0, mask_pos, :]

        # Get top k predictions
        top_token_ids = torch.topk(mask_logits, top_k * 2).indices

        # Decode and filter predictions
        predictions = []
        for token_id in top_token_ids:
            token = self.tokenizer.decode([token_id]).strip()

            # Filter for valid words
            if (token and
                not token.startswith('[') and
                not token.startswith('#') and
                token.lower().isalpha() and
                len(token) >= 2 and
                token.lower() not in ['the', 'and', 'or', 'but', 'for', 'with']):  # Skip very common words
                predictions.append(token.lower())

                if len(predictions) >= top_k:
                    break

        return predictions

    def evaluate_next_word(self, test_sentences, top_k=5):
        """
        Evaluate next word prediction accuracy
        """
        correct = 0
        total = 0

        print(f"=== Next Word Prediction Evaluation (Top-{top_k}) ===")
        print("Sample predictions:")
        print("-" * 70)

        for i, sentence in enumerate(test_sentences):
            words = sentence.strip().split()
            if len(words) < 4:  # Need at least 4 words
                continue

            # Target word (last word, cleaned)
            target_word = words[-1].lower().strip('.,!?;:"()[]{}')

            # Context (all words except the last one)
            context = " ".join(words[:-1])

            # Skip if target is a very common word (harder to predict meaningfully)
            if target_word in ['the', 'and', 'or', 'but', 'for', 'with', 'on', 'in', 'at', 'to']:
                continue

            # Get predictions
            predictions = self.predict_next_word(context, top_k=top_k)

            # Check if target is in predictions
            is_correct = target_word in predictions
            if is_correct:
                correct += 1
            total += 1

            # Show sample results
            if i < 7 or is_correct:
                print(f"Context: '{context}'")
                print(f"Target: '{target_word}'")
                print(f"Predictions: {predictions}")
                print(f"✅ Correct: {is_correct}")
                print("-" * 50)

        accuracy = correct / total if total > 0 else 0
        print(f"\n🎯 Next Word Prediction Top-{top_k} Accuracy: {accuracy:.1%} ({correct}/{total})")
        return accuracy

    def interactive_demo(self):
        """Interactive demo for testing"""
        print("\n=== Interactive Next Word Prediction Demo ===")
        print("Enter a sentence fragment and I'll predict the next word!")
        print("Type 'quit' to exit\n")

        while True:
            try:
                context = input("Context: ").strip()
                if context.lower() == 'quit':
                    break

                predictions = self.predict_next_word(context, top_k=8)
                print(f"Next word predictions: {predictions}\n")

            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")

def test_model_performance():
    """Test the trained model"""
    print("=== Testing Trained Model ===")

    try:
        evaluator = NextWordEvaluator("./bert-nextword-v3/final_model")

        # Quick test examples
        test_contexts = [
            "The cat is sitting",
            "I want to eat",
            "She went to",
            "The weather looks",
            "He decided to",
            "The movie was very",
            "I need to buy"
        ]

        print("Sample next word predictions:")
        print("-" * 50)
        for context in test_contexts:
            predictions = evaluator.predict_next_word(context, top_k=6)
            print(f"'{context}' → {predictions}")

        return evaluator

    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def run_next_word_training():
    """
    Main function to train and evaluate next word prediction
    """
    print("🚀 Starting BERT Next Word Prediction Training")
    print("=" * 60)

    # Initialize trainer
    trainer = NextWordBERTTrainer()

    # Train the model
    trainer.train_model(
        train_file="next.txt",
        output_dir="./bert-nextword-v3",
        num_epochs=4,
        batch_size=32,
        validation_split=0.2,
        block_size=64
    )

    print("\n" + "="*60)
    print("🧪 Testing the trained model...")

    # Test the model
    evaluator = test_model_performance()

    if evaluator:
        # Evaluation sentences
        test_sentences = [
            "The quick brown fox jumps over the lazy dog",
            "She went to the store to buy some groceries",
            "I really like to eat pizza",
            "The weather today is very hot",
            "He opened the door and saw his friend",
            "Machine learning is becoming more important",
            "The sun rises in the morning",
            "She decided to read a book",
            "The children are playing in the park",
            "We need to complete our homework",
            "The car was parked outside the house",
            "I enjoy listening to classical music"
        ]

        # Run evaluation
        accuracy = evaluator.evaluate_next_word(test_sentences, top_k=5)

        print(f"\n🎉 Training Complete! Final Accuracy: {accuracy:.1%}")

        # Uncomment for interactive demo
        # evaluator.interactive_demo()

if __name__ == "__main__":
    run_next_word_training()
