This project fine-tunes a pre-trained BERT model to perform next word prediction, treating it like a causal language modeling task. Although BERT is originally trained for masked language modeling (MLM), this script adapts it to predict the next token in a sentence using a shifted input-label strategy.

🧠 Project Overview
Next Word Prediction involves training a model to predict the next token given a sequence of preceding tokens. While models like GPT are designed for this task, this project creatively uses BERT (a bidirectional encoder model) for next-token prediction by aligning inputs and labels accordingly.

🏗️ How It Works
The script performs the following steps:

1. Data Preparation
Loads a plain text file (data.txt) containing training sentences.

Each line is treated as an independent sequence for training.

2. Tokenization
Uses BertTokenizer (bert-base-uncased) to tokenize text.

Converts text into input IDs.

Pads or truncates sequences to a fixed maximum length (default: 32 tokens).

3. Label Alignment
Labels are created by shifting the input tokens to the left by one.

This makes the model learn to predict token t[i+1] given t[i].

4. Model
Loads a pre-trained BertForMaskedLM from HuggingFace Transformers.

Although BertForMaskedLM is designed for masked token prediction, it's used here for causal-style next word prediction.

5. Training
Uses CrossEntropyLoss to compute loss between predicted and actual next tokens.

Optimizes the model over multiple epochs using AdamW optimizer.

Prints loss and accuracy after each epoch.
