import torch.nn as nn

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