import torch
import torch.nn as nn
import torch.optim

class SimpleTextClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, num_classes=2):
        super(SimpleTextClassifier, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, input_ids):
        embeddings = self.embed(input_ids)
        averaged = embeddings.mean(dim=1)
        output = self.classifier(averaged)
        return output



class SimpleAttentionTextClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, num_classes=2):
        super(SimpleAttentionTextClassifier, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, input_ids):
        # Create embedding for each token
        embeddings = self.embed(input_ids)

        # For each token, create a vector of dot products with other vectors
        # Batch matrix multiplication: each entry is the dot product of embeddings
        # Transpose: (Batch, Tokens, Embedding) -> (Batch, Embedding, Tokens)
        attention_scores = torch.bmm(embeddings, embeddings.transpose(1, 2))

        # Apply softmax to attention scores for each token
        attention_soft_maxxing = torch.softmax(attention_scores, dim=2)

        # Computing contextual embeddings:
        # Each embedding (for each token) is the sum of all embeddings multiplied by their weights
        contextual_embeddings = torch.bmm(attention_soft_maxxing, embeddings)


        # For each token, compute attention score for every token in the sequence
        # For each of these attention scores, apply softmax (to all of them)
        # For each entry in the vector that is this token (the embedding), swap this entry for a weighted sum of all other vectors, with weights as scores
        # Proceed with mean
        # Add self-attention
        averaged = contextual_embeddings.mean(dim=1)
        output = self.classifier(averaged)
        return output