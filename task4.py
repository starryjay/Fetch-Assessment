"""
Task 4:
If not already done, code the training loop for the Multi-Task Learning Expansion in Task 2.
Explain any assumptions or decisions made paying special attention to how training within a MTL framework operates. 
Please note you need not actually train the model.
Things to focus on:
• Handling of hypothetical data
• Forward pass
• Metrics
"""

import torch
import torch.nn as nn
from task1 import tokenizer, sample_sentences
from task2 import MultiTaskSentenceTransformer
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset
import numpy as np

class SampleSet(Dataset): # creating a custom dataset class extending PyTorch's Dataset class to ensure compatibility with DataLoader
    def __init__(self, sentences, sentence_labels, sentiment_labels):
        self.sentences = sentences
        self.sentence_labels = sentence_labels
        self.sentiment_labels = sentiment_labels

    def __len__(self):
        return len(self.sentences) # returning the length of the dataset

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        sentence_label = self.sentence_labels[idx]
        sentiment_label = self.sentiment_labels[idx]
        return sentence, sentence_label, sentiment_label
    
num_sentences = len(sample_sentences)
num_sentence_classes = 5 # arbitrary
num_sentiment_classes = 3 # arbitrary

np.random.seed(42)
sample_sentence_labels = np.random.randint(0, num_sentence_classes, size=num_sentences).tolist() # creating a random ground truth label set
sample_sentiment_labels = np.random.randint(0, num_sentiment_classes, size=num_sentences).tolist() # same as above

sample_set = SampleSet(sample_sentences, sample_sentence_labels, sample_sentiment_labels) # loading the sample sentences into a dataset
sample_dataloader = DataLoader(sample_set, batch_size=8, shuffle=True) # creating a dataloader to handle batching and shuffling of the dataset

model = MultiTaskSentenceTransformer()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5) # low learning rate for fine-tuning
criterion_sentence, criterion_sentiment = nn.CrossEntropyLoss(), nn.CrossEntropyLoss() # cross-entropy loss for multi-class classification tasks
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
def train_model(model, dataloader, optimizer, criterion_sentence, criterion_sentiment, device):
    model.train()
    for batch in dataloader:
        sentences, sentence_labels, sentiment_labels = batch
        sentences = tokenizer(sentences, return_tensors="pt", padding=True, truncation=True) # tokenizing the sentences and converting them to tensors
        sentences = {k: v.to(device) for k, v in sentences.items()} # moving the input tensors to the appropriate device (GPU/CPU)
        sentence_labels = sentence_labels.to(device)
        sentiment_labels = sentiment_labels.to(device)

        outputs = model(input_ids=sentences['input_ids'], attention_mask=sentences['attention_mask']) # performing a forward pass through the model
        sentence_classification_logits = outputs['sentence_classification_logits']
        sentiment_logits = outputs['sentiment_logits']

        loss_sentence = criterion_sentence(sentence_classification_logits, sentence_labels)
        loss_sentiment = criterion_sentiment(sentiment_logits, sentiment_labels)
        total_loss = loss_sentence + loss_sentiment # combining the losses from both tasks

        optimizer.zero_grad() # zeroing the gradients before backpropagation
        total_loss.backward() # performing backpropagation to compute gradients
        optimizer.step() # updating the model parameters
        print(f"Loss: {total_loss.item():.4f}")
        _, predicted_sentence_labels = torch.max(sentence_classification_logits, 1)
        _, predicted_sentiment_labels = torch.max(sentiment_logits, 1)
        sentence_accuracy = accuracy_score(sentence_labels.cpu(), predicted_sentence_labels.cpu())
        sentiment_accuracy = accuracy_score(sentiment_labels.cpu(), predicted_sentiment_labels.cpu())
        sentence_f1 = f1_score(sentence_labels.cpu(), predicted_sentence_labels.cpu(), average='weighted')
        sentiment_f1 = f1_score(sentiment_labels.cpu(), predicted_sentiment_labels.cpu(), average='weighted')
        print(f"Sentence Classification Accuracy: {sentence_accuracy:.4f}, F1 Score: {sentence_f1:.4f}")
        print(f"Sentiment Analysis Accuracy: {sentiment_accuracy:.4f}, F1 Score: {sentiment_f1:.4f}")

#train_model(model, sample_dataloader, optimizer, criterion_sentence, criterion_sentiment, device)

"""
Assumptions and Decisions:

1. Hypothetical Data:
    - We create a custom dataset class extending PyTorch's Dataset class to handle the hypothetical data.
    - Extending the Dataset class ensures compatibility with PyTorch's DataLoader, which handles batching and shuffling.
    - The dataset contains sentences, sentence labels, and sentiment labels.
    - We randomly generate labels for the sentences and sentiments.
    - We assume that the sentences are NOT already preprocessed and tokenized, and so we handle that in the training loop.
2. Forward Pass:
    - The forward pass is handled in the train_model function, where we pass the input sentences through the model.
    - The model outputs embeddings, sentence classification logits, and sentiment logits.
    - Logits allow us to compute the loss for each task.
    - We zero the gradients before backpropagation to avoid accumulating gradients from previous iterations.
3. Metrics:
    - We compute accuracy and F1 score for both tasks after each batch. 
    - F1 score helps us evaluate the model's performance on imbalanced datasets.
    - Accuracy helps us understand the model's overall performance.
    - We use the weighted average for F1 score to account for class imbalance.
    - We print the loss and metrics after each batch to monitor the training process.

"""