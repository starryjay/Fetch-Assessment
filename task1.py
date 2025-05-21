"""
Task 1: Sentence Transformer Implementation
Implement a sentence transformer model using any deep learning framework of your choice. 
This model should be able to encode input sentences into fixed-length embeddings. 
Test your implementation with a few sample sentences and showcase the obtained embeddings. 
Describe any choices you had to make regarding the model architecture outside of the transformer backbone.
"""
import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel

ds = load_dataset("agentlans/high-quality-english-sentences")

print(ds)

# using a sample dataset of 100 sentences for speed purposes
sample_sentences = ds['train']['text'][:100]

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased") # bert-base-uncased is good for context recognition
model = AutoModel.from_pretrained("bert-base-uncased")
inputs = tokenizer(sample_sentences, return_tensors="pt", padding=True, truncation=True)
with torch.no_grad():
    outputs = model(**inputs)
    mean_pooling = outputs.last_hidden_state.mean(dim=1) # mean pooling to get the sentence embeddings
    # we use mean pooling instead of CLS token since we are going to generalize the model to multi-task learning later,
    # and the CLS token is not as effective for that purpose since it does not generalize as well

class SentenceTransformer(nn.Module):
    def __init__(self, backbone_model='bert-base-uncased', embedding_dim=768): # BERT uses 768 dimensions for its embeddings
        super(SentenceTransformer, self).__init__()
        self.backbone = AutoModel.from_pretrained(backbone_model)
        self.pooling = "mean" # using mean pooling for the sentence embeddings
        self.embedding_dim = embedding_dim
        self.fc = nn.Linear(embedding_dim, embedding_dim) # fully connected layer to project the embedding to the same dimension
        self.norm = nn.LayerNorm(embedding_dim) # normalizing the layer to stabilize training

    def encode(self, input_ids, attention_mask): # function to encode the input sentences
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        if self.pooling == "mean":
            embeddings = outputs.last_hidden_state.mean(dim=1)
        elif self.pooling == "cls":
            embeddings = outputs.last_hidden_state[:, 0, :]
        else:
            raise ValueError("Pooling method not supported")
        embeddings = self.fc(embeddings)
        embeddings = self.norm(embeddings)
        return embeddings
        
model = SentenceTransformer()
inputs = tokenizer(sample_sentences, return_tensors="pt", padding=True, truncation=True)
with torch.no_grad(): # no gradient calculation needed for inference
    embeddings = model.encode(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
    print("Embeddings:")
    print(embeddings)
    print(embeddings.shape)


"""

The model architecture was chosen to be a simple transformer model with a fully connected layer and layer normalization.
We encode the input sentences using the transformer backbone, and then we apply mean pooling to get the sentence embeddings.
The mean pooling method was chosen over the CLS token since it is more effective for generalization to multi-task learning.
We project the embeddings to the same dimension as the transformer backbone output using a fully connected layer.
The layer normalization is used to stabilize the training process and improve convergence.
We then test the model with a few sample sentences and showcase the obtained embeddings.
The choice of the transformer backbone was made to leverage the pre-trained weights and the ability to handle context effectively.
The model is designed to be flexible and can be easily extended to handle multi-task learning by adding task-specific heads,
which we do in task2.py.

"""