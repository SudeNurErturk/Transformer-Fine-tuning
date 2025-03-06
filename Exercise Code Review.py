from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, EarlyStoppingCallback
from datasets import load_dataset
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Load IMDb dataset
# The IMDb dataset contains movie reviews labeled as positive or negative
# Useful for binary sentiment classification tasks
dataset = load_dataset("imdb")

# Initialize tokenizer and model
# Using the "bert-base-uncased" model for tokenizing and fine-tuning
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_function(examples):
    # Tokenize the text data with padding and truncation
    return tokenizer(examples["text"], padding="max_length", truncation=True)

# Tokenize and preprocess dataset
# Applying tokenization function on the entire dataset
encoded_dataset = dataset.map(tokenize_function, batched=True)

# Select subsets for training, validation, and testing
# Training on 20,000 samples, validation on 2,500, and testing on 5,000
train_dataset = encoded_dataset["train"].shuffle(seed=42).select(range(20000))
val_dataset = encoded_dataset["train"].shuffle(seed=42).select(range(20000, 22500))
test_dataset = encoded_dataset["test"].shuffle(seed=42).select(range(5000))

# Load pre-trained model
# Specifying 2 output labels for binary classification (positive/negative sentiment)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Training arguments
# Configuring the training parameters, including batch size, epochs, and logging
training_args = TrainingArguments(
    output_dir="./results",  
    evaluation_strategy="epoch",  
    save_strategy="epoch",  
    logging_dir="./logs",  
    per_device_train_batch_size=16,  
    per_device_eval_batch_size=16,  
    num_train_epochs=3,  
    weight_decay=0.01,  
    logging_steps=50,  
    load_best_model_at_end=True,  
    push_to_hub=False,  
)

def compute_metrics(pred):
    # Compute accuracy, precision, recall, and F1 score for evaluation
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)  # Convert logits to predictions
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

# Trainer setup
# Setting up the Trainer class for training and evaluation
trainer = Trainer(
    model=model,  
    args=training_args, 
    train_dataset=train_dataset,  
    eval_dataset=val_dataset,  
    tokenizer=tokenizer,  
    compute_metrics=compute_metrics,  
)

# Train the model
trainer.train()

# Evaluate the model
# Evaluate the model on the test dataset
results = trainer.evaluate(test_dataset)
print(results)

# Extract metrics from training logs
# Extracting training loss and steps for visualization
metrics = trainer.state.log_history
train_loss = [x["loss"] for x in metrics if "loss" in x]
steps = [x["step"] for x in metrics if "loss" in x]

# Plot training loss curve
plt.plot(steps, train_loss, label="Training Loss")
plt.xlabel("Steps")
plt.ylabel("Loss")
plt.title("Loss Curve")
plt.legend()
plt.show()

# Adjust training arguments with early stopping
# Adding early stopping to prevent overfitting and save training time
tuned_training_args = TrainingArguments(
    output_dir="./tuned_results", 
    evaluation_strategy="epoch",  
    save_strategy="epoch",  
    per_device_train_batch_size=32,  
    per_device_eval_batch_size=32,  
    num_train_epochs=3,  
    learning_rate=5e-5,  
    weight_decay=0.01,  
    logging_steps=50,  
    load_best_model_at_end=True,  
)

# Setup trainer with early stopping
# Adding an early stopping callback to halt training if no improvement
callbacks = [EarlyStoppingCallback(early_stopping_patience=2)]
tuned_trainer = Trainer(
    model=model,  # Model to be fine-tuned
    args=tuned_training_args,  # Updated training arguments
    train_dataset=train_dataset,  # Training dataset
    eval_dataset=val_dataset,  # Validation dataset
    tokenizer=tokenizer,  # Tokenizer for preprocessing
    compute_metrics=compute_metrics,  # Metrics function for evaluation
    callbacks=callbacks,  # Early stopping callback
)

# Train the model with early stopping
tuned_trainer.train()

# Evaluate the fine-tuned model
# Test the model on the test dataset after fine-tuning
test_results = tuned_trainer.evaluate(test_dataset)
print("Test Set Evaluation Metrics:", test_results)

# Extract metrics from training logs
# Extracting loss values for training and validation loss visualization
loss_history = tuned_trainer.state.log_history
train_loss = [x["loss"] for x in loss_history if "loss" in x]
eval_loss = [x["eval_loss"] for x in loss_history if "eval_loss" in x]
epochs = range(1, len(eval_loss) + 1)

# Plot training and validation loss curves
plt.plot(epochs, train_loss[:len(epochs)], label="Training Loss")
plt.plot(epochs, eval_loss, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss Curve")
plt.legend()
plt.show()
