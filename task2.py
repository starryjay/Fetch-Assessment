"""
Task 2: Multi-Task Learning Expansion
Expand the sentence transformer to handle a multi-task learning setting.
Task 2A: Sentence Classification: Classify sentences into predefined classes (you can make these up).
Task 2B: [Choose another relevant NLP task such as Named Entity Recognition, Sentiment Analysis, etc.] (you can make the labels up)
Describe the changes made to the architecture to support multi-task learning.
"""
import torch
import torch.nn as nn
from task1 import SentenceTransformer, tokenizer, sample_sentences

class MultiTaskSentenceTransformer(SentenceTransformer): # inheriting from the SentenceTransformer class to reuse the backbone model
    def __init__(self, backbone_model='bert-base-uncased', embedding_dim=768, num_classes=5, num_sentiments=3): # adding num_classes and num_sentiments for multi-task since we are doing sentence classification and sentiment analysis
        super(MultiTaskSentenceTransformer, self).__init__(backbone_model, embedding_dim)
        self.sentence_classifier = nn.Linear(in_features=embedding_dim, out_features=num_classes)
        self.sentiment_classifier = nn.Linear(in_features=embedding_dim, out_features=num_sentiments)
        
    def forward(self, input_ids, attention_mask): # overriding the forward method to include multi-task outputs, encoding the input sentences and passing them through the backbone model
        embeddings = self.encode(input_ids, attention_mask)
        sentence_classification_logits = self.sentence_classifier(embeddings)
        sentiment_logits = self.sentiment_classifier(embeddings)
        return {'embeddings': embeddings,
                'sentence_classification_logits': sentence_classification_logits,
                'sentiment_logits': sentiment_logits}
    
model = MultiTaskSentenceTransformer()
inputs = tokenizer(sample_sentences, return_tensors="pt", padding=True, truncation=True)
with torch.no_grad():
    outputs = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
    print("Embeddings:")
    print(outputs['embeddings'])
    print("Sentence Classification Logits:")
    print(outputs['sentence_classification_logits'])
    print("Sentiment Logits:")
    print(outputs['sentiment_logits'])


"""

The architecture was expanded to include two task-specific heads: one for sentence classification and one for sentiment analysis.
We ensure that MultiTaskSentenceTransformer inherits from the SentenceTransformer class to reuse the transformer backbone.
The sentence classification head takes the embeddings from the transformer backbone and passes them through a linear layer to 
produce logits for each class. 
The sentiment analysis head does the same, but with a different number of output classes. 
The model now outputs the embeddings, sentence classification logits, and sentiment logits, allowing for multi-task learning.
We reuse the encode method from the SentenceTransformer class to handle the encoding of the input sentences,
and we override the forward method to include the multi-task outputs.

"""