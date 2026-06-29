import torch
import torch.nn as nn

# ── Reproducibility ──────────────────────────────────────────────
SEED = 42
torch.manual_seed(SEED)

# ── Data Loading ─────────────────────────────────────────────────
text = open("../data/shakespeare.txt").read()

chars = sorted(set(text))
vocab_size = len(chars)
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for c, i in stoi.items()}

encode = lambda s: [stoi[c] for c in s]
decode = lambda ids: "".join([itos[i] for i in ids])

data = torch.tensor(encode(text), dtype=torch.long)

# ── Train / Validation Split (90% / 10%) ─────────────────────────
n = int(0.9 * len(data))
train_data = data[:n]
val_data   = data[n:]

print(f"Total tokens : {len(data):,}")
print(f"Train tokens : {len(train_data):,}")
print(f"Val tokens   : {len(val_data):,}")

# ── Hyperparameters ───────────────────────────────────────────────
BLOCK_SIZE = 8    # context length
BATCH_SIZE = 32
EMBED_DIM  = 32
LR         = 1e-2
EPOCHS     = 5000

# ── get_batch() ───────────────────────────────────────────────────
def get_batch(split):
    """
    Returns (x, y) tensors of shape (BATCH_SIZE, BLOCK_SIZE).
    x = input tokens, y = next-token targets (shifted by 1).
    """
    source = train_data if split == "train" else val_data
    ix = torch.randint(len(source) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack([source[i : i + BLOCK_SIZE]     for i in ix])
    y = torch.stack([source[i + 1 : i + BLOCK_SIZE + 1] for i in ix])
    return x, y

# Verify shapes
xb, yb = get_batch("train")
print(f"\nget_batch() shape check:")
print(f"  x : {xb.shape}  # (batch, block_size)")
print(f"  y : {yb.shape}  # (batch, block_size)")

# ── Model ─────────────────────────────────────────────────────────
class SimpleLanguageModel(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        # Token embedding: maps token id → vector
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        # Linear head: maps vector → logits over vocab
        self.lm_head = nn.Linear(embed_dim, vocab_size)

    def forward(self, idx, targets=None):
        # idx shape: (B, T)
        x = self.token_embedding(idx)          # (B, T, embed_dim)
        logits = self.lm_head(x)               # (B, T, vocab_size)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits_2d  = logits.view(B * T, C)    # (B*T, vocab_size)
            targets_1d = targets.view(B * T)       # (B*T,)
            loss = nn.functional.cross_entropy(logits_2d, targets_1d)

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, _ = self(idx)
            logits = logits[:, -1, :]              # last time step
            probs  = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_token], dim=1)
        return idx

model = SimpleLanguageModel(vocab_size, EMBED_DIM)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

# ── Training Loop ─────────────────────────────────────────────────
@torch.no_grad()
def estimate_loss(eval_iters=200):
    model.eval()
    results = {}
    for split in ["train", "val"]:
        losses = []
        for _ in range(eval_iters):
            xb, yb = get_batch(split)
            _, loss = model(xb, yb)
            losses.append(loss.item())
        results[split] = sum(losses) / len(losses)
    model.train()
    return results

print("\nStarting training...")
for step in range(EPOCHS):
    xb, yb = get_batch("train")
    logits, loss = model(xb, yb)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step == 0:
        print(f"Step {step:>5} | loss = {loss.item():.4f}  ← initial loss")

    if (step + 1) % 1000 == 0:
        stats = estimate_loss()
        print(f"Step {step+1:>5} | train = {stats['train']:.4f} | val = {stats['val']:.4f}")

print(f"\nFinal loss: {loss.item():.4f}")