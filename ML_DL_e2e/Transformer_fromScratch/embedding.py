#class for embedding layer
from torch import nn
import math

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.embed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim, padding_idx=0)

    def forward(self, X):
        #nn.Embedding initializes weights from N(0, 1), so individual embedding values start with variance ~1 per dimension — 
        # but summed/compared against the sinusoidal PE (which is strictly bounded in [-1, 1]),
        #  the relative contribution of token identity vs. position depends on this scale. 
        # The √d_model factor makes the embedding's typical magnitude large enough that positional encoding 
        # meaningfully adds information without swamping it, per the original paper's stated reasoning.

        #So: the attention scaling factor addresses numerical stability inside softmax, several layers downstream. 
        # The embedding scaling factor addresses the relative balance between token and position signal, right at the input. 
        # Fixing one doesn't fix the other — they're independent design choices at different points in the forward pass.
        return self.embed(X) * math.sqrt(self.embed.embedding_dim)
        