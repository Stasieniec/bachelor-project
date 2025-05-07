import torch
import torch.nn as nn
import torch.optim
import torch.nn.functional as F


class FeedForward(nn.Module):
    def __init__(self, d_model, mult = 4):
        super().__init__()

        self.net = nn.Sequential(
            # Hidden layer: (B, T, d_model) -> (B, T, 4 * d_model)
            nn.Linear(d_model, mult*d_model),
            # Gaussian Error Linear Unit (instead of ReLU)
            nn.GELU(),
            # Ending with a tensor with original dimensions
            nn.Linear(mult*d_model, d_model)
        )

    def forward(self, x):
        return self.net(x)



class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, block_size=512):
        super().__init__()

        self.n_heads = n_heads
        self.head_dim = d_model//n_heads
        self.scale = self.head_dim ** -0.5

        # Query, Key, Value
        self.qkv = nn.Linear(d_model, 3 * d_model, bias = False)

        # Output (mix heads back together)
        self.proj = nn.Linear(d_model, d_model, bias = False)

        # Mask created out of a lower triangular matrix: future values are multiplied by 0 while past do not change
        mask = torch.tril(torch.ones(block_size, block_size))

        # Store the mask with the model but do not treat it as a learnable parameter
        self.register_buffer("mask", mask)

    def forward(self, x):
        # Batch, time, embedding dimensions
        B, T, _ = x.shape

        # Project by one and the same transformation, then split
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1) # (B, T, 3d) -> (B, T, d)

        # Split by heads and reshape
        # (Q, K, V) -> (B, h, T, d_h) for parallelism: pytorch can compute each B, h separately
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # Scaled dot product attention of queries and keys
        att = (q @ k.transpose(-2, -1) * self.scale) # (B, h, T, T)
        # Mask: block attention for future positions
        att = att.masked_fill(self.mask[:T, :T] == 0, -1e4)

        att = att.softmax(dim=-1)

        # Weighted average of value vectors
        out = att @ v

        # Merge heads
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        # view requires contiguous memory allocation (order in memory = the same order as if it was created like that)


        # Project output
        return self.proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()

        self.attention = MultiHeadAttention(d_model = d_model, n_heads = n_heads, )

        self.ff = FeedForward(d_model=d_model, mult=4)

        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x):
        # Attention
        a = self.ln1(x) # Normalize (B, T, D)
        a = self.attention(a)
        x = x + a # Residual, apparently needed (TODO: HOW DOES IT WORK??)

        # FeedForward
        f = self.ln2(x)
        f = self.ff(f)
        x = x + f

        return x



class ByteEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(512, d_model) # 512 is the max block size


    def forward(self, idx):
        B, T = idx.shape
        tok = self.token_emb
        pos = self.pos_emb(torch.arange(T, device=idx.device))

        return tok + pos



class SimpleGPT(nn.Module):
    def __init__(self,
                 vocab_size: int = 256,
                 d_model: int = 256,
                 n_heads: int = 4,
                 n_layers: int = 6,
                 block_size: int = 512,
                 dropout_p: float = 0.1):
        super().__init__()

        self.embed = ByteEmbedding(vocab_size, d_model)

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads)
            for _ in range(n_layers)
        ])

        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias = False)

        # Weight-tying trick
        #self.head.weight = self.embed.tok_emb.weight

        self.block_size = block_size

    def forward(self, idx, targets = None):
        B, T = idx.shape
        assert T <= self.block_size, "sequence longer than2 block size"

        x = self.embed(idx)                              # (B,T,d)

        for block in self.blocks:
            x = block(x, mask=None)                      # mask already in block

        x = self.ln_f(x)
        logits = self.head(x)                            # (B,T,256)

        if targets is None:
            return logits                                # inference

        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
        )
        return logits, loss

    @torch.no_grad()
    def generate(self,
                 prompt: torch.Tensor,  # (B, T₀) containing byte‑ids
                 max_new_tokens: int = 256,
                 temperature: float = 1.0,
                 top_k: int | None = 50):

        self.eval()
        B = prompt.size(0)
        out = prompt.clone()  # we will cat() onto this

        for _ in range(max_new_tokens):
            # clip to block_size
            xb = out[:, -self.block_size:]  # (B, ≤block,)

            logits, _ = self(xb)  # (B, T, vocab)
            logits = logits[:, -1, :] / temperature  # take last time step

            if top_k is not None:
                topk_vals, _ = torch.topk(logits, top_k)
                logits[logits < topk_vals[:, [-1]]] = -float("Inf")

            probs = torch.softmax(logits, dim=-1)  # (B, 256)
            next_id = torch.multinomial(probs, num_samples=1)  # (B, 1)

            out = torch.cat([out, next_id], dim=1)  # append

        return out

