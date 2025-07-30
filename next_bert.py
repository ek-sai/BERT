import torch
import numpy as np
from transformers import (
    BertForMaskedLM, BertTokenizer,
    Trainer, TrainingArguments,
    TrainerCallback,
    get_linear_schedule_with_warmup
)
from torch.utils.data import Dataset, random_split
import os
import json
from datetime import datetime
import matplotlib.pyplot as plt

# Custom Dataset without augmentation or masking (masking done in collator)
class NextWordPredictionDataset(Dataset):
    def __init__(self, tokenizer, file_path, block_size=256):
        self.tokenizer = tokenizer
        self.block_size = block_size

        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # Simple filtering
        self.lines = [line.strip() for line in lines if len(line.strip()) > 10 and len(line.split()) >= 3]

        print(f"Loaded {len(self.lines)} lines from {file_path}")

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        line = self.lines[idx]

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

# Custom Data Collator that masks only the last token for next word prediction
class NextWordPredictionCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.mask_token_id = tokenizer.mask_token_id
        self.ignore_index = -100

    def __call__(self, examples):
        input_ids = torch.stack([ex['input_ids'] for ex in examples])
        attention_mask = torch.stack([ex['attention_mask'] for ex in examples])

        labels = input_ids.clone()

        # Mask all tokens except the last one - set labels to ignore_index so loss only on last token
        labels[:, :-1] = self.ignore_index

        # Replace the last token in input_ids with [MASK] token
        input_ids[:, -1] = self.mask_token_id

        batch = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }
        return batch

# Trainer callback to log losses and learning rates
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
            self.trainer_instance.training_history['epoch'].append(state.epoch)

class ImprovedBERTTrainerNextWord:
    def __init__(self, model_name="bert-base-uncased", use_large=False):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        if use_large:
            model_name = "bert-large-uncased"
            print("Using BERT-Large")

        self.model = BertForMaskedLM.from_pretrained(model_name)
        self.tokenizer = BertTokenizer.from_pretrained(model_name)

        self.model.to(self.device)

        self.training_history = {
            'train_loss': [],
            'eval_loss': [],
            'learning_rate': [],
            'epoch': []
        }

    def prepare_datasets(self, train_file, validation_split=0.1, block_size=256):
        dataset = NextWordPredictionDataset(
            tokenizer=self.tokenizer,
            file_path=train_file,
            block_size=block_size
        )

        train_size = int((1 - validation_split) * len(dataset))
        val_size = len(dataset) - train_size

        train_dataset, val_dataset = random_split(
            dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )

        print(f"Train samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")

        return train_dataset, val_dataset

    def create_data_collator(self):
        return NextWordPredictionCollator(tokenizer=self.tokenizer)

    def setup_training_args(self, output_dir, num_epochs=5, batch_size=32):
        return TrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=True,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=2,
            learning_rate=2e-5,
            weight_decay=0.01,
            warmup_ratio=0.1,
            lr_scheduler_type="linear",
            logging_steps=50,
            eval_steps=200,
            save_steps=400,
            eval_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            fp16=True,
            dataloader_num_workers=4,
            dataloader_pin_memory=True,
            report_to=None,
            logging_dir=f"{output_dir}/logs",
            run_name=f"bert_nextword_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )

    def train_model(self, train_file, output_dir="./bert-nextword", **kwargs):
        os.makedirs(output_dir, exist_ok=True)

        train_dataset, val_dataset = self.prepare_datasets(
            train_file=train_file,
            validation_split=kwargs.get('validation_split', 0.1),
            block_size=kwargs.get('block_size', 256)
        )

        data_collator = self.create_data_collator()

        training_args = self.setup_training_args(
            output_dir=output_dir,
            num_epochs=kwargs.get('num_epochs', 5),
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

        print("Starting training (Next Word Prediction with BERT)...")
        print("="*50)

        trainer.train()

        final_path = f"{output_dir}/final_model"
        trainer.save_model(final_path)
        print(f"Model saved to: {final_path}")

        with open(f"{output_dir}/training_history.json", 'w') as f:
            json.dump(self.training_history, f, indent=2)

        self.plot_training_curves(output_dir)

        return trainer

    def plot_training_curves(self, output_dir):
        if not self.training_history['train_loss']:
            return

        plt.figure(figsize=(12,4))

        plt.subplot(1,2,1)
        plt.plot(self.training_history['train_loss'], label='Training Loss')
        if self.training_history['eval_loss']:
            plt.plot(self.training_history['eval_loss'], label='Validation Loss')
        plt.xlabel('Steps')
        plt.ylabel('Loss')
        plt.title('Loss')
        plt.legend()
        plt.grid(True)

        plt.subplot(1,2,2)
        if self.training_history['learning_rate']:
            plt.plot(self.training_history['learning_rate'], label='Learning Rate')
        plt.xlabel('Steps')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(f"{output_dir}/training_curves.png", dpi=300, bbox_inches='tight')
        print(f"Training curves saved: {output_dir}/training_curves.png")

# Evaluation class for next word prediction (mask last token and predict)
class NextWordEvaluator:
    def __init__(self, model_path, tokenizer_name="bert-base-uncased"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = BertForMaskedLM.from_pretrained(model_path)
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
        self.model.to(self.device)
        self.model.eval()

    def predict_next_word(self, sentence, top_k=5):
        # Tokenize sentence normally
        tokens = self.tokenizer.tokenize(sentence)
        if len(tokens) == 0:
            return []

        # We mask last token
        # Remove last token, then add [MASK]
        tokens_masked = tokens[:-1] + [self.tokenizer.mask_token]

        input_ids = self.tokenizer.convert_tokens_to_ids(tokens_masked)
        input_ids = torch.tensor([input_ids]).to(self.device)

        attention_mask = torch.ones_like(input_ids).to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

        mask_token_index = (input_ids == self.tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0].item()
        mask_token_logits = logits[0, mask_token_index]

        top_tokens = torch.topk(mask_token_logits, top_k).indices.tolist()
        predicted_words = [self.tokenizer.decode([token]).strip() for token in top_tokens]

        return predicted_words

    def evaluate(self, sentences, top_k=5):
        correct = 0
        total = 0
        for sentence in sentences:
            tokens = self.tokenizer.tokenize(sentence)
            if len(tokens) < 2:
                continue
            target = tokens[-1]

            preds = self.predict_next_word(sentence, top_k=top_k)
            if target in preds:
                correct += 1
            total += 1

        accuracy = correct / total if total > 0 else 0
        print(f"Top-{top_k} accuracy: {accuracy:.2%}")
        return accuracy

# Usage example
def run_training_and_evaluation():
    trainer = ImprovedBERTTrainerNextWord(use_large=False)

    trainer.train_model(
        train_file="next.txt",  # Your training data file with sentences
        output_dir="./bert-nextword",
        num_epochs=5,
        batch_size=32,
        validation_split=0.1,
        block_size=256,
    )

    evaluator = NextWordEvaluator("./bert-nextword/final_model")
    # Example sentences for evaluation - replace with your own or read from file
    test_sentences = [
        "The quick brown fox jumps over the lazy",
        "She went to the store to buy some",
        "Artificial intelligence is changing the",
        "He opened the door and saw a",
        "This is an example of next word"
    ]
    evaluator.evaluate(test_sentences, top_k=5)

if __name__ == "__main__":
    run_training_and_evaluation()
