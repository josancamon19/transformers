import math
import torch.nn as nn
import torch
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from transformers import GPT2Tokenizer


# We expect you to build these components from scratch. In particular, you may not
# use any definitions from torch.nn, torch.nn.functional, or torch.optim except for the following:
# • torch.nn.Parameter
# • Container classes in torch.nn (e.g., Module, ModuleList, Sequential, etc.)1
# • The torch.optim.Optimizer base class


class Embedding(nn.Module):
    def __init__(self, vocab_size: int, embed_dim):
        super().__init__()
        self.embeddings = nn.Parameter(torch.empty((vocab_size, embed_dim)))
        nn.init.trunc_normal_(self.embeddings, mean=0, std=1, a=-3, b=3)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.embeddings[token_ids]


class Linear(nn.Module):
    def __init__(self, inp: int, out: int, bias: bool = False, device: torch.device = None, dtype: torch.dtype = None):
        super().__init__()
        self.inp = inp
        self.out = out
        self.weights = nn.Parameter(torch.empty((out, inp), device=device, dtype=dtype))
        self.bias = None if not bias else nn.Parameter(torch.zeros((out,), device=device, dtype=dtype))

    def reset_parameters(self) -> None:
        std = math.sqrt(2 / (self.inp + self.out))
        nn.init.trunc_normal_(self.weights, mean=0, std=std, a=-3 * std, b=3 * std)

    def forward(self, x):
        if self.bias is not None:
            return x @ self.weights.T + self.bias
        return x @ self.weights.T


def get_positional_encodings(embed_dim, max_positions=3):
    positions = torch.arange(max_positions).unsqueeze(1)
    dim_indices = torch.arange(0, embed_dim, 2)  # TODO: not sure if this part is clear
    div_term = 10000 ** (dim_indices / embed_dim)

    pe = torch.empty((max_positions, embed_dim))
    pe[:, 0::2] = torch.sin(positions / div_term)
    pe[:, 1::2] = torch.cos(positions / div_term)
    return pe


class RMSNorm(nn.Module):
    def __init__(self, embedding_dim: int, eps: float = 1e-5):
        # TODO: understand reasoning, not only implement
        super().__init__()
        self.embedding_dim = embedding_dim
        self.eps = eps
        self.gain = Parameter(torch.ones(embedding_dim))

    def forward(self, x):
        # TODO: gpt recommends formula is not clear, review
        # --- identify variance/means comp and name properly
        x_dtype = x.dtype
        x = x.to(torch.float32)
        tsum = torch.sum(torch.pow(x, 2) + self.eps, dim=2)
        div_term = torch.sqrt((1 / self.embedding_dim) * tsum).unsqueeze(2)
        result = torch.divide(x, div_term) * self.gain
        return result.to(x_dtype)


def silu(w_out: torch.Tensor):
    return w_out * torch.sigmoid(w_out)
    pass


torch.manual_seed(42)
inp = torch.tensor([0, 1, 2, 3])
inp2 = torch.tensor([4, 5, 6, 7])
batched_inp = torch.stack([inp, inp2])
e = Embedding(10, 4)
e_out = e(batched_inp)
# l1 = Linear(2, 4, False)
# print("l1.data", l1.weights.data)
# l1_out = l1(e(batched_inp))
print(e_out)
silu(e_out)



# TODO: optimize computation MultiHeadSelfAttention batch,layer,seq,embeddings
# TODO: Custom LayerNorm
# TODO: RMSNorm + Understanding
# TODO: implement ReLU/SwiGLU + Understanding
# TODO: AdamW
# TODO: lr scheduler + gradient clipping
# TODO: checkpointing, wandb
# TODO: inference
# TODO: Read the actual document
# TODO: train, Tiny


class SelfAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads):
        super().__init__()
        head_size = embedding_dim // num_heads
        # print("SelfAttention.__init__:", shape)
        self.Q = Linear(embedding_dim, head_size)
        self.K = Linear(embedding_dim, head_size)
        self.V = Linear(embedding_dim, head_size)

    def forward(self, x, padding_mask):
        # batch, seq_length, embedding_dim
        q = x @ self.Q
        k = x @ self.K
        v = x @ self.V

        # print("SelfAttention.forward q.shape:", q.shape)
        attention_scores = q @ k.transpose(1, 2)
        causal_mask = torch.tril(torch.ones((x.shape[1], x.shape[1]))).to(self.Q.device)
        mask = causal_mask * padding_mask.unsqueeze(1) if padding_mask is not None else causal_mask
        attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        # print(attention_scores)
        # TODO: figoure out the output of 7,7, which i,j refer to which tokens

        # print("SelfAttention.forward attention_scores:", attention_scores.shape)
        attention_weights = torch.softmax(attention_scores / math.sqrt(k.shape[-1]), dim=-1)
        # print(attention_weights)
        # print("SelfAttention.forward attention_weights:", attention_weights.shape)
        output = attention_weights @ v
        # print("SelfAttention.forward output:", output.shape)
        return output


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads):
        super().__init__()
        self.attention = nn.Sequential(*[SelfAttention(embedding_dim, num_heads) for _ in range(num_heads)])
        self.W_O = Linear(embedding_dim, embedding_dim)

    def forward(self, x, padding_mask):
        # print(x.shape)
        output = [head(x, padding_mask) for head in self.attention]
        output = torch.cat(output, dim=-1)
        wo = output @ self.W_O
        # print("MultiHeadSelfAttention.forward wo.shape:", wo.shape)
        return wo


class TransformerBlock(nn.Module):
    def __init__(self, embedding_dim, num_heads):
        super().__init__()
        self.pre_mhsa_norm = nn.LayerNorm(embedding_dim)
        self.mhsa = MultiHeadSelfAttention(embedding_dim, num_heads)

        self.pre_mlp_norm = nn.LayerNorm(embedding_dim)

        self.mlp_in = Linear(embedding_dim, embedding_dim * 4, True)
        self.relu = F.relu
        self.mlp_out = Linear(4 * embedding_dim, embedding_dim, True)

    def forward(self, x, padding_mask):
        x = self.pre_mhsa_norm(x)
        attention = self.mhsa(x, padding_mask) + x

        attention_norm = self.pre_mlp_norm(attention)
        proj_in = attention_norm @ self.mlp_in + self.mlp_in_bias
        proj_in_activated = self.relu(proj_in)
        proj_out = proj_in_activated @ self.mlp_out + self.mlp_out_bias
        output = proj_out + attention
        return output


class Transformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        max_sequence_length: int,
        embedding_dim: int,
        num_layers: int,
        num_heads: int,
    ):
        super().__init__()
        self.embeddings = Embedding(vocab_size, embedding_dim)
        pe = get_positional_encodings(embedding_dim, max_sequence_length)
        self.register_buffer("pe", pe)

        self.blocks = nn.ModuleList(TransformerBlock(embedding_dim, num_heads) for _ in range(num_layers))
        self.pre_output_norm = nn.LayerNorm(embedding_dim)
        self.output = nn.Parameter(torch.empty(embedding_dim, vocab_size))
        nn.init.normal_(self.output, std=0.02)

    def forward(self, input_ids, padding_mask):
        # print(self.pe[: len(input_ids), :].shape)
        tokens = self.embeddings(input_ids) + self.pe[: input_ids.shape[-1], :]
        for block in self.blocks:
            tokens = block(tokens, padding_mask)

        tokens = self.pre_output_norm(tokens)
        output = tokens @ self.output
        # print("Transformer.forward output.shape:", output.shape)
        # return torch.softmax(output, dim=-1)
        return output  # output logits


# tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2")
# max_sequence_length = 100
# embedding_dim = 128
# num_layers = 1
# num_heads = 1

# tokenizer.pad_token = "[PAD]"
# tokenized = tokenizer(
#     ["Hi there, this is a test", "hey"],
#     return_tensors="pt",
#     padding=True,
#     truncation=True,
# )
# print(f"input:\n{tokenized['input_ids']}")
# print(tokenized["attention_mask"])
# tf = Transformer(tokenizer.vocab_size, max_sequence_length, embedding_dim, num_layers, num_heads)
# prob_dist = tf(tokenized["input_ids"], tokenized["attention_mask"])
