import torch
import torch.nn as nn
from torch.nn import functional as F
import math

# First, let's create our own configuration class to hold the model's parameters.
class GPTConfig:
    def __init__(self, vocab_size=50257, n_positions=1024, n_embd=768, n_layer=12, n_head=12):
        self.vocab_size = vocab_size
        self.n_positions = n_positions  # Max sequence length
        self.n_embd = n_embd            # Embedding dimension
        self.n_layer = n_layer          # Number of transformer blocks
        self.n_head = n_head            # Number of attention heads

class MultiHeadSelfAttention(nn.Module):
    """
    The core component of the Transformer: Multi-Head Self-Attention.
    This layer allows the model to weigh the importance of different tokens in the sequence.
    """
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # A single linear layer projects the input to Query, Key, and Value for all heads at once.
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # Another linear layer to project the combined head outputs back to the embedding dimension.
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        # Store config
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # This is the causal mask to ensure the model can't "see" future tokens.
        self.register_buffer("bias", torch.tril(torch.ones(config.n_positions, config.n_positions))
                                     .view(1, 1, config.n_positions, config.n_positions))

    def forward(self, x):
        B, T, C = x.size() # Batch size, Sequence length, Embedding dimensionality

        # Get Query, Key, Value from the input and reshape for multi-head attention
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # The attention mechanism: (Q * K^T) / sqrt(d_k)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # Apply the causal mask
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        
        # Apply attention to the Values
        y = att @ v
        # Re-assemble the heads
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # Final projection
        y = self.c_proj(y)
        return y

class FeedForward(nn.Module):
    """ A simple two-layer MLP that follows the attention block. """
    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd)
        self.act = nn.GELU() # GELU activation function

    def forward(self, x):
        x = self.c_fc(x)
        x = self.act(x)
        x = self.c_proj(x)
        return x

class TransformerBlock(nn.Module):
    """ The complete Transformer block, combining Attention and FeedForward layers. """
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = MultiHeadSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.ffwd = FeedForward(config)

    def forward(self, x):
        # This is the architecture: LayerNorm -> Attention -> Residual -> LayerNorm -> FeedForward -> Residual
        x = x + self.attn(self.ln_1(x))
        x = x + self.ffwd(self.ln_2(x))
        return x

class GPTModel(nn.Module):
    """ The full GPT model. """
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),      # Token embeddings
            wpe = nn.Embedding(config.n_positions, config.n_embd),     # Positional embeddings
            h = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)]), # Stack of blocks
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # Tie the weights of the token embeddings and the final linear layer. This is a common optimization.
        self.transformer.wte.weight = self.lm_head.weight 

    def forward(self, idx):
        device = idx.device
        b, t = idx.size()
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)

        # Get token and position embeddings
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = tok_emb + pos_emb
        
        # Pass through all the transformer blocks
        for block in self.transformer.h:
            x = block(x)
        
        x = self.transformer.ln_f(x)
        # The final "language modeling head" that produces the output logits
        logits = self.lm_head(x)
        return logits

def create_model():
    """
    Creates the GPT model from scratch using our pure PyTorch implementation.
    """
    print("Creating a new GPT model from scratch using PyTorch...")
    config = GPTConfig() # Use default small model parameters
    model = GPTModel(config)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model created successfully.")
    print(f"Number of parameters: {num_params / 1_000_000:.2f} Million")

    return model
