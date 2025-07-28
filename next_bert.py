import torch
import numpy as np
from transformers import (
    BertForMaskedLM, BertTokenizer,
    Trainer, TrainingArguments,
    LineByLineTextDataset,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
    TrainerCallback,
    get_linear_schedule_with_warmup
)
from torch.utils.data import Dataset, DataLoader, random_split
import os
import json
from datetime import datetime
import matplotlib.pyplot as plt

class EnhancedTextDataset(Dataset):
    """Enhanced dataset with better preprocessing and augmentation"""

    def __init__(self, tokenizer, file_path, block_size=256, augment=True):
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.augment = augment

        # Read and preprocess text
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # Clean and filter lines
        self.lines = []
        for line in lines:
            line = line.strip()
            if len(line) > 10 and len(line.split()) >= 3:  # Minimum quality check
                self.lines.append(line)

        print(f"Loaded {len(self.lines)} valid lines from {file_path}")

        # Data augmentation
        if augment:
            self.lines = self.augment_data(self.lines)
            print(f"After augmentation: {len(self.lines)} lines")

    def augment_data(self, lines):
        """Simple data augmentation techniques"""
        augmented = lines.copy()

        for line in lines[:len(lines)//2]:  # Augment 50% of data
            words = line.split()
            if len(words) > 5:
                # Random word shuffle (preserve first/last)
                middle = words[1:-1]
                np.random.shuffle(middle)
                shuffled = [words[0]] + middle + [words[-1]]
                augmented.append(' '.join(shuffled))

        return augmented

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        line = self.lines[idx]
        # Tokenize and truncate/pad to block_size
        encoding = self.tokenizer(
            line,
            truncation=True,
            padding='max_length',
            max_length=self.block_size,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0)
        }

class ImprovedBERTTrainer:
    """Enhanced BERT trainer with monitoring and optimization"""

    def __init__(self, model_name="bert-base-uncased", use_large=False):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Choose model size
        if use_large:
            model_name = "bert-large-uncased"
            print("Using BERT-Large for better performance")

        # Load model and tokenizer
        self.model = BertForMaskedLM.from_pretrained(model_name)
        self.tokenizer = BertTokenizer.from_pretrained(model_name)

        # Move to GPU
        self.model.to(self.device)

        # Training history
        self.training_history = {
            'train_loss': [],
            'eval_loss': [],
            'learning_rate': [],
            'epoch': []
        }

    def prepare_datasets(self, train_file, validation_split=0.1, block_size=256):
        """Prepare training and validation datasets"""
        print("Preparing datasets...")

        # Create enhanced dataset
        full_dataset = EnhancedTextDataset(
            tokenizer=self.tokenizer,
            file_path="next.txt",
            block_size=block_size,
            augment=True
        )

        # Split into train/validation
        train_size = int((1 - validation_split) * len(full_dataset))
        val_size = len(full_dataset) - train_size

        train_dataset, val_dataset = random_split(
            full_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )

        print(f"Training samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")

        return train_dataset, val_dataset

    def create_data_collator(self, mlm_probability=0.15):
        """Create enhanced data collator"""
        return DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=True,
            mlm_probability=mlm_probability
        )

    def setup_training_args(self, output_dir, num_epochs=5, batch_size=32):
        """Setup optimized training arguments"""
        return TrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=True,

            # Training schedule
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=2,

            # Optimization
            learning_rate=2e-5,
            weight_decay=0.01,
            warmup_ratio=0.1,
            lr_scheduler_type="linear",

            # Monitoring and saving
            logging_steps=50,
            eval_steps=200,
            save_steps=400,
            eval_strategy="steps",
            save_strategy="steps",

            # Best model selection
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,

            # Performance optimizations
            fp16=True,
            dataloader_num_workers=4,
            dataloader_pin_memory=True,

            # Disable external logging
            report_to=None,

            # Additional monitoring
            logging_dir=f"{output_dir}/logs",
            run_name=f"bert_improved_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )

    def train_model(self, train_file, output_dir="./bert-improved", **kwargs):
        """Main training function with all improvements"""

        # Setup
        os.makedirs(output_dir, exist_ok=True)

        # Prepare data
        train_dataset, val_dataset = self.prepare_datasets(
            train_file=train_file,
            validation_split=kwargs.get('validation_split', 0.1),
            block_size=kwargs.get('block_size', 256)
        )

        # Data collator
        data_collator = self.create_data_collator(
            mlm_probability=kwargs.get('mlm_probability', 0.15)
        )

        # Training arguments
        training_args = self.setup_training_args(
            output_dir=output_dir,
            num_epochs=kwargs.get('num_epochs', 5),
            batch_size=kwargs.get('batch_size', 32)
        )

        # Custom callback for tracking
        class HistoryCallback(TrainerCallback):
            def __init__(self, trainer_instance):
                self.trainer_instance = trainer_instance

            def on_log(self, args, state, control, logs=None, **kwargs):
                if logs:
                    if 'train_loss' in logs:
                        self.trainer_instance.training_history['train_loss'].append(logs['train_loss'])
                    if 'eval_loss' in logs:
                        self.trainer_instance.training_history['eval_loss'].append(logs['eval_loss'])
                    if 'learning_rate' in logs:
                        self.trainer_instance.training_history['learning_rate'].append(logs['learning_rate'])
                    self.trainer_instance.training_history['epoch'].append(state.epoch)

        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            callbacks=[
                EarlyStoppingCallback(early_stopping_patience=3),
                HistoryCallback(self)
            ]
        )

        print("Starting training...")
        print("=" * 50)

        # Train the model
        trainer.train()

        # Save final model
        final_path = f"{output_dir}/final_model"
        trainer.save_model(final_path)
        print(f"Final model saved to: {final_path}")

        # Save training history
        with open(f"{output_dir}/training_history.json", 'w') as f:
            json.dump(self.training_history, f, indent=2)

        # Plot training curves
        self.plot_training_curves(output_dir)

        return trainer

    def plot_training_curves(self, output_dir):
        """Plot training and validation curves"""
        if not self.training_history['train_loss']:
            return

        plt.figure(figsize=(12, 4))

        # Loss curves
        plt.subplot(1, 2, 1)
        if self.training_history['train_loss']:
            plt.plot(self.training_history['train_loss'], label='Training Loss')
        if self.training_history['eval_loss']:
            plt.plot(self.training_history['eval_loss'], label='Validation Loss')
        plt.xlabel('Steps')
        plt.ylabel('Loss')
        plt.title('Training Progress')
        plt.legend()
        plt.grid(True)

        # Learning rate
        plt.subplot(1, 2, 2)
        if self.training_history['learning_rate']:
            plt.plot(self.training_history['learning_rate'], label='Learning Rate')
        plt.xlabel('Steps')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Schedule')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(f"{output_dir}/training_curves.png", dpi=300, bbox_inches='tight')
        print(f"Training curves saved to: {output_dir}/training_curves.png")

# Enhanced Evaluation Class
class ComprehensiveEvaluator:
    """More robust evaluation with multiple test scenarios"""

    def __init__(self, model_path, tokenizer_name="bert-base-uncased"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = BertForMaskedLM.from_pretrained(model_path)
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
        self.model.to(self.device)
        self.model.eval()

    def create_stratified_test_data(self, test_file=None, num_samples=200):
        """Create test data with different difficulty levels"""

        if test_file and os.path.exists(test_file):
            with open(test_file, 'r', encoding='utf-8') as f:
                sentences = [line.strip() for line in f.readlines() if len(line.strip()) > 10]
        else:
            # Default test sentences (Sherlock Holmes themed)
            sentences = [
                "The detective examined the evidence with his magnifying glass.",
                "Holmes deduced the solution from the smallest clue.",
                "Watson followed his friend through the foggy London streets.",
                "The criminal left behind a mysterious calling card.",
                "Inspector Lestrade requested assistance with the difficult case.",
                "The victim was found in the locked study room.",
                "Sherlock observed every detail of the crime scene.",
                "The witness gave a detailed description of the suspect.",
                "The murder weapon was hidden in the garden shed.",
                "Holmes revealed the truth in his final explanation."
            ] * 20  # Repeat to get enough samples

        # Sample random sentences
        if len(sentences) > num_samples:
            sentences = np.random.choice(sentences, num_samples, replace=False).tolist()

        test_data = []
        for sentence in sentences:
            words = sentence.split()
            if len(words) < 4:
                continue

            # Create different masking strategies
            mask_strategies = [
                ([len(words)//2], 'middle'),  # Mask middle word
                ([1], 'beginning'),           # Mask second word
                ([len(words)-2], 'end'),      # Mask second-to-last word
                ([np.random.randint(1, len(words)-1)], 'random')  # Random word
            ]

            for mask_positions, strategy in mask_strategies:
                if all(0 <= pos < len(words) for pos in mask_positions):
                    original_words = words.copy()
                    masked_words = words.copy()

                    target_tokens = []
                    for pos in mask_positions:
                        target_tokens.append(original_words[pos])
                        masked_words[pos] = '[MASK]'

                    test_data.append({
                        'original': sentence,
                        'masked': ' '.join(masked_words),
                        'target_tokens': target_tokens,
                        'strategy': strategy,
                        'difficulty': self.assess_difficulty(target_tokens[0]) if target_tokens else 'medium'
                    })

        return test_data

    def assess_difficulty(self, word):
        """Assess word difficulty based on frequency and complexity"""
        common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'will', 'would', 'could', 'should', 'can', 'may', 'might', 'must'}

        if word.lower() in common_words:
            return 'easy'
        elif len(word) > 8 or word.istitle():
            return 'hard'
        else:
            return 'medium'

    def comprehensive_evaluation(self, test_file=None):
        """Run comprehensive evaluation"""
        print("Creating comprehensive test data...")
        test_data = self.create_stratified_test_data(test_file)

        # Group by difficulty
        easy_tests = [t for t in test_data if t['difficulty'] == 'easy']
        medium_tests = [t for t in test_data if t['difficulty'] == 'medium']
        hard_tests = [t for t in test_data if t['difficulty'] == 'hard']

        print(f"Test distribution - Easy: {len(easy_tests)}, Medium: {len(medium_tests)}, Hard: {len(hard_tests)}")

        # Evaluate each difficulty level
        results = {}
        for difficulty, tests in [('easy', easy_tests), ('medium', medium_tests), ('hard', hard_tests), ('overall', test_data)]:
            if not tests:
                continue

            print(f"\nEvaluating {difficulty} tests...")
            accuracy_results = self.evaluate_accuracy(tests, k_values=[1, 3, 5, 10])
            results[difficulty] = accuracy_results

        # Print results
        self.print_results(results)
        return results

    def evaluate_accuracy(self, test_data, k_values=[1, 3, 5, 10]):
        """Calculate top-k accuracy"""
        results = {}

        for k in k_values:
            correct = 0
            total = 0

            for item in test_data:
                predictions = self.predict_masked_tokens(item['masked'], top_k=k)

                for i, target in enumerate(item['target_tokens']):
                    if i < len(predictions):
                        if target.lower() in [p.lower() for p in predictions[i]]:
                            correct += 1
                    total += 1

            results[f'top_{k}'] = correct / total if total > 0 else 0

        return results

    def predict_masked_tokens(self, masked_sentence, top_k=5):
        """Predict top-k tokens for masked positions"""
        inputs = self.tokenizer(masked_sentence, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = outputs.logits[0]

        mask_positions = torch.where(inputs["input_ids"][0] == self.tokenizer.mask_token_id)[0]

        predicted_tokens = []
        for pos in mask_positions:
            top_predictions = torch.topk(predictions[pos], top_k)
            tokens = [self.tokenizer.decode(token_id).strip() for token_id in top_predictions.indices]
            predicted_tokens.append(tokens)

        return predicted_tokens

    def print_results(self, results):
        """Print formatted results"""
        print("\n" + "="*60)
        print("COMPREHENSIVE EVALUATION RESULTS")
        print("="*60)

        for difficulty, metrics in results.items():
            print(f"\n{difficulty.upper()} DIFFICULTY:")
            for metric, score in metrics.items():
                print(f"  {metric.replace('_', '-').upper()}: {score:.1%}")

# Usage Example
def run_improved_training():
    """Run the complete improved training pipeline"""

    # Initialize trainer
    trainer = ImprovedBERTTrainer(use_large=False)  # Set to True for BERT-large

    # Train with improved settings
    model_trainer = trainer.train_model(
        train_file="next.txt",  # Your training file
        output_dir="./bert-improved-v2",
        num_epochs=5,
        batch_size=32,
        validation_split=0.1,
        block_size=256,
        mlm_probability=0.15
    )

    print("\nTraining completed! Now evaluating...")

    # Comprehensive evaluation
    evaluator = ComprehensiveEvaluator("./bert-improved-v2/final_model")
    results = evaluator.comprehensive_evaluation()

    return model_trainer, results

if __name__ == "__main__":
    # Run the improved training
    trainer, results = run_improved_training()

    print("\n🎉 Training and evaluation completed!")
    print("Check the output directory for:")
    print("  • Final trained model")
    print("  • Training curves plot")
    print("  • Training history JSON")
    print("  • Comprehensive evaluation results")
