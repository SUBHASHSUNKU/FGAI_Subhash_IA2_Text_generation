# %% [markdown]
# # Autoregressive Word-Level Text Generation: SimpleRNN vs BiRNN vs LSTM vs GRU
# **Course:** Foundations of Generative AI (ICT4403) - Assignment 2
# **Dataset:** Tiny Shakespeare (Karpathy, char-rnn) - https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
#
# Run on Google Colab with a GPU runtime (Runtime > Change runtime type > T4 GPU), then `Run all`.

# %%
import os, re, time, math, copy, random, json, urllib.request
from collections import Counter
import numpy as np
import pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
import matplotlib.pyplot as plt

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

# ---- Hyper-parameters (IDENTICAL for all four models) ----
SEQ_LEN    = 20      # input window (words)
MIN_FREQ   = 3       # words rarer than this (in train split) -> <unk>
EMB_DIM    = 128
HID_DIM    = 256
DROPOUT    = 0.3
BATCH      = 256
LR         = 2e-3
MAX_EPOCHS = 20
PATIENCE   = 3       # early stopping on validation loss
CLIP       = 1.0
TEMPERATURES = [0.5, 0.8, 1.0, 1.5]
GEN_WORDS  = 40
N_SAMPLES  = 10      # samples per (model, seed, temperature) for the metrics
SEEDS = [
    "first citizen : before we proceed any further , hear me speak .",
    "my lord , what news from the war ? the king",
    "i am the queen of this land and i will not",
]
MODELS = ["SimpleRNN", "BiRNN", "LSTM", "GRU"]
os.makedirs("results", exist_ok=True)
print("device:", DEVICE)

def set_seed(s=SEED):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)

# %% [markdown]
# ## 1. Dataset: characteristics and preprocessing

# %%
if not os.path.exists("input.txt"):
    urllib.request.urlretrieve(URL, "input.txt")
raw = open("input.txt", encoding="utf-8").read()
print(f"characters: {len(raw):,} | lines: {raw.count(chr(10)):,}")
print(raw[:300])

TOKEN_RE = re.compile(r"<nl>|[a-z]+(?:'[a-z]+)*|[^\sa-z]")
def tokenize(text):
    return TOKEN_RE.findall(text.lower().replace("\n", " <nl> "))

tokens = tokenize(raw)
print(f"total tokens: {len(tokens):,} | unique tokens: {len(set(tokens)):,}")

# contiguous 80/10/10 split (no leakage between windows of different splits)
n = len(tokens)
tr_tok, va_tok, te_tok = tokens[:int(.8*n)], tokens[int(.8*n):int(.9*n)], tokens[int(.9*n):]
cnt = Counter(tr_tok)
itos = ["<pad>", "<unk>"] + [w for w, c in cnt.most_common() if c >= MIN_FREQ]
stoi = {w: i for i, w in enumerate(itos)}
PAD, UNK, V = 0, 1, len(itos)
print(f"vocab size (freq>={MIN_FREQ}): {V:,}")
print("most common:", cnt.most_common(10))

def encode(text_or_tokens):
    toks = tokenize(text_or_tokens) if isinstance(text_or_tokens, str) else text_or_tokens
    return [stoi.get(t, UNK) for t in toks]

def detok(ids):
    s = " ".join(itos[i] for i in ids if i > 1)
    s = s.replace(" <nl> ", "\n").replace("<nl> ", "\n").replace(" <nl>", "\n")
    return re.sub(r" ([,.;:!?])", r"\1", s)

# %% [markdown]
# ## 2. Input-target sequences (sliding window)
# For every position *i*: input = words `[i, i+SEQ_LEN)`, target = word `i+SEQ_LEN` (next word).

# %%
def make_xy(tok_list):
    ids = np.array(encode(tok_list), dtype=np.int64)
    w = np.lib.stride_tricks.sliding_window_view(ids, SEQ_LEN + 1)
    X = torch.tensor(np.ascontiguousarray(w[:, :-1]), device=DEVICE)
    y = torch.tensor(np.ascontiguousarray(w[:, -1]), device=DEVICE)
    return X, y

Xtr, ytr = make_xy(tr_tok); Xva, yva = make_xy(va_tok); Xte, yte = make_xy(te_tok)
print("train/val/test sequences:", len(Xtr), len(Xva), len(Xte))
print("example input :", detok(Xtr[0].tolist()).replace("\n", " / "))
print("example target:", itos[ytr[0].item()])
TRAIN_BIGRAMS = set(zip(encode(tr_tok), encode(tr_tok)[1:]))

# %% [markdown]
# ## 2b. Dataset characteristics

# %%
WORD_RE = re.compile(r"[a-z]+(?:'[a-z]+)*")
lines = raw.split("\n"); non_empty = [l for l in lines if l.strip()]
word_toks = [t for t in tokens if WORD_RE.fullmatch(t)]
punct_toks = [t for t in tokens if t != "<nl>" and not WORD_RE.fullmatch(t)]
unk = lambda tl: np.mean([stoi.get(t, UNK) == UNK for t in tl])
line_len = [len(tokenize(l)) for l in non_empty]
speakers = Counter(re.findall(r"^([A-Z][A-Za-z ]+):$", raw, flags=re.M))
stats = [
    ("Characters", f"{len(raw):,}"), ("Lines (incl. blank)", f"{len(lines):,}"),
    ("Non-empty lines", f"{len(non_empty):,}"),
    ("Tokens (words + punctuation + <nl>)", f"{len(tokens):,}"),
    ("Word tokens", f"{len(word_toks):,}"), ("Punctuation tokens", f"{len(punct_toks):,}"),
    ("Unique tokens", f"{len(set(tokens)):,}"),
    (f"Vocabulary used (freq >= {MIN_FREQ}, + <pad>, <unk>)", f"{V:,}"),
    ("Avg tokens per non-empty line", f"{np.mean(line_len):.1f}"),
    ("Train / val / test tokens", f"{len(tr_tok):,} / {len(va_tok):,} / {len(te_tok):,}"),
    ("<unk> rate train / val / test", f"{unk(tr_tok)*100:.1f}% / {unk(va_tok)*100:.1f}% / {unk(te_tok)*100:.1f}%"),
    ("Distinct speaker labels", f"{len(speakers):,}"),
]
top_words = [(w, c) for w, c in Counter(word_toks).most_common(15)]
md = "## Dataset characteristics\n\n| Property | Value |\n|---|---|\n" + "\n".join(f"| {a} | {b} |" for a, b in stats)
md += "\n\nTop 15 words: " + ", ".join(f"{w} ({c:,})" for w, c in top_words) + "\n"
open("results/dataset_stats.md", "w").write(md); print(md)

freqs = sorted(Counter(tokens).values(), reverse=True)
fig, ax = plt.subplots(1, 3, figsize=(16, 3.8))
ax[0].bar([w for w, _ in top_words], [c for _, c in top_words]); ax[0].set_title("Top 15 words"); ax[0].tick_params(axis="x", rotation=60)
ax[1].loglog(range(1, len(freqs) + 1), freqs); ax[1].set_title("Rank-frequency (Zipf) plot"); ax[1].set_xlabel("rank"); ax[1].set_ylabel("frequency")
ax[2].hist(line_len, bins=range(0, 40, 2)); ax[2].set_title("Tokens per non-empty line"); ax[2].set_xlabel("tokens")
plt.tight_layout(); plt.savefig("results/dataset_characteristics.png", dpi=150); plt.show()

# %% [markdown]
# ## 3. Models (same embedding size, hidden size, dropout, optimiser for all)

# %%
class WordRNN(nn.Module):
    def __init__(self, kind):
        super().__init__()
        self.bi = kind == "BiRNN"
        self.lstm = kind == "LSTM"
        self.emb = nn.Embedding(V, EMB_DIM, padding_idx=PAD)
        cls = {"SimpleRNN": nn.RNN, "BiRNN": nn.RNN, "LSTM": nn.LSTM, "GRU": nn.GRU}[kind]
        self.rnn = cls(EMB_DIM, HID_DIM, batch_first=True, bidirectional=self.bi)
        self.drop = nn.Dropout(DROPOUT)
        self.fc = nn.Linear(HID_DIM * (2 if self.bi else 1), V)

    def forward(self, x):
        _, hn = self.rnn(self.drop(self.emb(x)))
        if self.lstm: hn = hn[0]
        h = torch.cat([hn[-2], hn[-1]], 1) if self.bi else hn[-1]
        return self.fc(self.drop(h))

# %% [markdown]
# ## 4. Training

# %%
@torch.no_grad()
def evaluate(model, X, y):
    model.eval(); tl = c1 = c5 = 0
    for i in range(0, len(X), 1024):
        xb, yb = X[i:i+1024], y[i:i+1024]
        lg = model(xb)
        tl += F.cross_entropy(lg, yb, reduction="sum").item()
        t5 = lg.topk(5, dim=1).indices
        c1 += (t5[:, 0] == yb).sum().item()
        c5 += (t5 == yb[:, None]).any(1).sum().item()
    n = len(X)
    return tl/n, math.exp(tl/n), c1/n, c5/n

def sync():
    if DEVICE == "cuda": torch.cuda.synchronize()

def train_model(kind):
    set_seed()
    model = WordRNN(kind).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    params = sum(p.numel() for p in model.parameters())
    hist = {"train_loss": [], "val_loss": [], "val_acc": []}
    best, best_state, bad = 1e9, None, 0
    sync(); t0 = time.time()
    for ep in range(1, MAX_EPOCHS + 1):
        model.train(); idx = torch.randperm(len(Xtr), device=DEVICE); tl = 0
        for i in range(0, len(idx), BATCH):
            b = idx[i:i+BATCH]
            loss = F.cross_entropy(model(Xtr[b]), ytr[b])
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), CLIP); opt.step()
            tl += loss.item() * len(b)
        vl, _, va, _ = evaluate(model, Xva, yva)
        hist["train_loss"].append(tl/len(idx)); hist["val_loss"].append(vl); hist["val_acc"].append(va)
        print(f"[{kind}] epoch {ep:2d} train {tl/len(idx):.3f} val {vl:.3f} acc {va:.3f}")
        if vl < best - 1e-4: best, best_state, bad = vl, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= PATIENCE: break
    sync(); secs = time.time() - t0
    model.load_state_dict(best_state)
    return model, hist, secs, params

trained, histories, rows = {}, {}, []
for kind in MODELS:
    m, h, secs, params = train_model(kind)
    trained[kind], histories[kind] = m, h
    te_loss, te_ppl, te_acc, te_top5 = evaluate(m, Xte, yte)
    ep_run = len(h["val_loss"])
    rows.append(dict(Model=kind, Parameters=params, Epochs_run=ep_run,
                     Best_epoch=int(np.argmin(h["val_loss"])) + 1,
                     Train_time_s=round(secs, 1), Time_per_epoch_s=round(secs/ep_run, 1),
                     Test_loss=round(te_loss, 4), Test_perplexity=round(te_ppl, 2),
                     Top1_acc=round(te_acc*100, 2), Top5_acc=round(te_top5*100, 2)))
comp = pd.DataFrame(rows)
comp.to_csv("results/model_comparison.csv", index=False)
json.dump(histories, open("results/training_history.json", "w"))
print(comp.to_string(index=False))

# %% [markdown]
# ## 5. Quantitative comparison plots

# %%
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
for k in MODELS:
    ax[0].plot(histories[k]["train_loss"], label=k)
    ax[1].plot(histories[k]["val_loss"], label=k)
ax[0].set_title("Training loss"); ax[1].set_title("Validation loss")
for a in ax: a.set_xlabel("epoch"); a.set_ylabel("cross-entropy"); a.legend()
plt.tight_layout(); plt.savefig("results/loss_curves.png", dpi=150); plt.show()

fig, ax = plt.subplots(1, 4, figsize=(16, 3.5))
for a, col, t in zip(ax, ["Test_perplexity", "Top1_acc", "Top5_acc", "Train_time_s"],
                     ["Test perplexity (lower better)", "Top-1 acc % ", "Top-5 acc %", "Training time (s)"]):
    a.bar(comp["Model"], comp[col], color=["#888", "#4c78a8", "#e45756", "#54a24b"]); a.set_title(t)
plt.tight_layout(); plt.savefig("results/metric_comparison.png", dpi=150); plt.show()

# %% [markdown]
# ## 6. Text generation with temperature sampling

# %%
@torch.no_grad()
def generate(model, seed_text, temperature, n_words=GEN_WORDS, n_samples=N_SAMPLES, gen_seed=0):
    model.eval()
    g = torch.Generator(device=DEVICE); g.manual_seed(gen_seed)
    ids = ([PAD] * SEQ_LEN + encode(seed_text))[-SEQ_LEN:]
    x = torch.tensor([ids] * n_samples, device=DEVICE); out = []
    for _ in range(n_words):
        lg = model(x) / temperature
        lg[:, PAD] = float("-inf"); lg[:, UNK] = float("-inf")
        nxt = torch.multinomial(F.softmax(lg, 1), 1, generator=g)
        out.append(nxt); x = torch.cat([x[:, 1:], nxt], 1)
    return torch.cat(out, 1).cpu().tolist()

def gen_metrics(seqs):
    d1, d2, rep, bv = [], [], [], []
    for s in seqs:
        bg = list(zip(s, s[1:])); tg = list(zip(s, s[1:], s[2:]))
        d1.append(len(set(s)) / len(s)); d2.append(len(set(bg)) / len(bg))
        rep.append(1 - len(set(tg)) / len(tg))
        bv.append(np.mean([b in TRAIN_BIGRAMS for b in bg]))
    return dict(Distinct1=np.mean(d1), Distinct2=np.mean(d2), Repetition=np.mean(rep), BigramValidity=np.mean(bv))

grows, samples_md = [], ["# Generated text samples (first sample of each cell, 40 words)\n"]
for si, seed in enumerate(SEEDS):
    samples_md.append(f"\n## Seed {si+1}: `{seed}`\n")
    for T in TEMPERATURES:
        samples_md.append(f"\n### Temperature {T}\n")
        for kind in MODELS:
            seqs = generate(trained[kind], seed, T, gen_seed=1000 + si)
            grows.append(dict(Model=kind, Seed=si+1, Temperature=T, **gen_metrics(seqs)))
            samples_md.append(f"**{kind}:** {detok(seqs[0]).replace(chr(10), ' / ')}\n")
gen = pd.DataFrame(grows)
gen.to_csv("results/generation_metrics_per_seed.csv", index=False)
gsum = gen.groupby(["Model", "Temperature"])[["Distinct1", "Distinct2", "Repetition", "BigramValidity"]].mean().round(3).reset_index()
gsum.to_csv("results/generation_metrics_by_temperature.csv", index=False)
open("results/generated_samples.md", "w", encoding="utf-8").write("\n".join(samples_md))
print(gsum.to_string(index=False))
print("\n".join(samples_md[:30]))

# %%
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
for a, col in zip(ax, ["Distinct2", "Repetition", "BigramValidity"]):
    for k in MODELS:
        d = gsum[gsum.Model == k]; a.plot(d.Temperature, d[col], marker="o", label=k)
    a.set_title(col); a.set_xlabel("temperature"); a.legend()
plt.tight_layout(); plt.savefig("results/temperature_analysis.png", dpi=150); plt.show()

# %% [markdown]
# ## 7. Auto-generated results summary (paste into README)

# %%
def md_table(df):
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(str(v) for v in r) + " |" for r in df.values.tolist()]
    return "\n".join(lines)

best = comp.sort_values("Test_perplexity").iloc[0]["Model"]
summary = ("## Model comparison (test set)\n\n" + md_table(comp) +
           f"\n\nLowest test perplexity: **{best}**\n\n## Generation metrics by temperature (mean over 3 seeds x {N_SAMPLES} samples)\n\n" +
           md_table(gsum))
open("results/summary.md", "w").write(summary)
print(summary)
