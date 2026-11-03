"""
EAN (Expert Arbitration Network) - end-to-end training + evaluation
===================================================================

Implements YOUR diagram literally:

  Latent Input  ->  n Interpretation Heads (H1..Hn)  ->  Candidate outputs P1..Pn
                                         |
  Decoupled Evidence Stream --(alpha modulation)--> Evidence-Guided Trust Gating
                                         |
                       Linear Arbitration Fusion:  Y = sum_i alpha_i * P_i

How each box is realised (nothing is hardcoded per head):

  Latent Input            : ONE shared frozen Mistral-7B (4-bit). All heads read the same
                            latent representation.
  Interpretation Heads    : n INDEPENDENT LoRA adapters (own A/B matrices, trained
                            end-to-end). They are NOT told "be the cautious one" or
                            "be the literal one". Specialisation EMERGES from the training
                            objective (mixture-of-experts likelihood, Jacobs et al. 1991).
  Candidate output P_i    : head i's next-token distribution p_i(y_t | y_<t, x).
  Evidence stream         : separate DeBERTa-v3 encoder (decoupled from the generator).
  Trust gating            : alpha_i,t = softmax_i( f(h_i,t , mean_j h_j,t , evidence, head_emb_i) )
                            - sees every head's state (so it can compare/arbitrate)
                            - sees the evidence vector
  Linear arbitration      : p(y_t) = sum_i alpha_i,t * p_i(y_t)   (training AND decoding)

Research ideas used (all published, well-known):
  * Mixture of experts / mixture likelihood training      (Jacobs et al. 1991)
  * Noisy gating + load balancing to prevent head collapse (Shazeer et al. 2017, Switch Transformer 2021)
  * Mixture-of-LoRA experts on a frozen LLM                (LoRAMoE / MixLoRA family)
  * Token-level ensembling in probability space at decode time
  * Oracle best-head upper bound to measure arbitration headroom

Honest scope:
  * Training signal = LawInstruct reference answers (supervised). No human preference
    labels here. Gate quality is therefore bounded by that supervision.
  * Evaluation = held-out LawInstruct (unseen-source AND same-source). NOT LegalBench.
    Run official LegalBench separately with its own scorers.
  * Written for a CUDA A100. Run with SMOKE=1 first to catch errors in minutes.

Usage:
  SMOKE=1 python ean_legal.py --stage all     # tiny sanity run
  python ean_legal.py --stage all             # full run
  python ean_legal.py --stage eval            # eval only (uses saved checkpoint)
"""

import os
import re
import gc
import sys
import json
import math
import random
import hashlib
import argparse
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModel,
    BitsAndBytesConfig,
)
from rouge_score import rouge_scorer

# ---------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------
SMOKE = os.environ.get("SMOKE", "0") == "1"

SEED = 42
DATASET_ID = "lawinstruct/lawinstruct"
DATASET_CONFIG = "all_english-10_english"
GENERATOR_ID = "mistralai/Mistral-7B-Instruct-v0.2"
EVIDENCE_ENCODER_ID = "microsoft/deberta-v3-base"

OUT = "./ean_legal_out"
CKPT = os.path.join(OUT, "ean_best.pt")
RESULTS = os.path.join(OUT, "results")
os.makedirs(RESULTS, exist_ok=True)

# heads
N_HEADS = 3
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
TARGETS = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}

# data
MAX_ROWS_SCAN = 3000 if SMOKE else 250_000
CAP_PER_SOURCE_TRAIN = 20 if SMOKE else 1500   # stops huge sources dominating
CAP_PER_SOURCE_EVAL = 10 if SMOKE else 60
MAX_TRAIN_ROWS = 64 if SMOKE else 24_000
MAX_PROMPT_TOKENS = 512 if SMOKE else 768
MAX_ANS_TOKENS = 64 if SMOKE else 192
EVID_TOKENS = 128 if SMOKE else 384

# training
BATCH = 1 if SMOKE else 2
ACCUM = 2 if SMOKE else 8
EPOCHS = 1
LR_LORA = 2e-4
LR_GATE = 3e-4
LR_EVID = 2e-5
WARMUP = 0.05
GATE_NOISE = 0.5          # noisy gating, annealed to 0
W_HEAD_NLL = 0.05         # keeps every head individually competent (no dead heads)
W_BALANCE = 0.02          # load balancing (no head collapse)
EVAL_EVERY = 5 if SMOKE else 150   # optimizer steps
VAL_ROWS = 6 if SMOKE else 160

# evaluation
EVAL_ROWS = 4 if SMOKE else 120
MAX_NEW = 24 if SMOKE else 128

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def seed_all(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


# ---------------------------------------------------------------
# TEXT UTILS + METRICS + JSON SAFEGUARD
# ---------------------------------------------------------------
def clean(t):
    return re.sub(r"\s+", " ", str(t or "")).strip()


def norm(t):
    t = clean(t).lower()
    t = re.sub(r"[^\w\s]", "", t)
    return re.sub(r"\s+", " ", t).strip()


def make_json_safe(obj):
    if isinstance(obj, dict):
        return {str(k): make_json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_safe(v) for v in obj]
    elif isinstance(obj, tuple):
        return [make_json_safe(v) for v in obj]
    elif isinstance(obj, (np.floating, float)):
        return float(obj)
    elif isinstance(obj, (np.integer, int)):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return make_json_safe(obj.tolist())
    elif torch.is_tensor(obj):
        return make_json_safe(obj.detach().cpu().tolist())
    return obj


_rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

def token_f1(pred, ref):
    p, g = norm(pred).split(), norm(ref).split()
    if not p or not g:
        return 0.0
    pc, gc_ = defaultdict(int), defaultdict(int)
    for w in p:
        pc[w] += 1
    for w in g:
        gc_[w] += 1
    ov = sum(min(c, gc_[w]) for w, c in pc.items())
    if ov == 0:
        return 0.0
    pr, rc = ov / len(p), ov / len(g)
    return 2 * pr * rc / (pr + rc)


def score(pred, ref):
    return {
        "rougeL": _rouge.score(ref, pred)["rougeL"].fmeasure if pred else 0.0,
        "f1": token_f1(pred, ref),
        "em": float(norm(pred) == norm(ref)),
    }


# ---------------------------------------------------------------
# DATA
# ---------------------------------------------------------------
def load_rows():
    print("Loading LawInstruct (this downloads a lot the first time)...")
    ds = load_dataset(DATASET_ID, DATASET_CONFIG, split="train", trust_remote_code=True)
    ds = ds.shuffle(seed=SEED)
    ds = ds.select(range(min(MAX_ROWS_SCAN, len(ds))))
    rows, seen = [], set()
    for r in ds:
        ans = clean(r.get("answer"))
        ins = clean(r.get("instruction"))
        pro = clean(r.get("prompt"))
        if not ans or not (ins or pro):
            continue
        fp = hashlib.sha256((norm(ins) + "||" + norm(pro) + "||" + norm(ans)).encode()).hexdigest()
        if fp in seen:
            continue
        seen.add(fp)
        rows.append({
            "instruction": ins,
            "prompt": pro,
            "answer": ans,
            "task_type": clean(r.get("task_type") or "legal_task"),
            "jurisdiction": clean(r.get("jurisdiction") or "unspecified"),
            "group": f"{r.get('dataset_name', 'unk')}::{r.get('subset_name', 'unk')}",
        })
    print("Usable rows:", len(rows))
    if len(rows) < 50:
        raise RuntimeError("Too few usable rows - check dataset schema/config.")
    return rows


def make_splits(rows):
    """Source-grouped OOD val/test + same-source iid val/test (so you see both)."""
    by = defaultdict(list)
    for r in rows:
        by[r["group"]].append(r)
    groups = sorted(by)
    rng = random.Random(SEED)
    rng.shuffle(groups)
    n = len(groups)
    n_test = max(1, int(0.15 * n))
    n_val = max(1, int(0.05 * n))
    test_g, val_g, train_g = groups[:n_test], groups[n_test:n_test + n_val], groups[n_test + n_val:]
    if not train_g:
        raise RuntimeError("Not enough source groups for a grouped split.")

    def take(gs, cap):
        out = []
        for g in gs:
            lst = by[g][:]
            rng.shuffle(lst)
            out += lst[:cap]
        rng.shuffle(out)
        return out

    train = take(train_g, CAP_PER_SOURCE_TRAIN)[:MAX_TRAIN_ROWS + 800]
    ood_val, ood_test = take(val_g, CAP_PER_SOURCE_EVAL), take(test_g, CAP_PER_SOURCE_EVAL)
    k = 8 if SMOKE else 200
    iid_val, iid_test = train[:k], train[k:k + (8 if SMOKE else 500)]
    train = train[k + len(iid_test):][:MAX_TRAIN_ROWS]
    print(f"sources train/val/test = {len(train_g)}/{len(val_g)}/{len(test_g)}")
    print(f"rows train={len(train)} iid_val={len(iid_val)} iid_test={len(iid_test)} "
          f"ood_val={len(ood_val)} ood_test={len(ood_test)}")
    return train, iid_val, iid_test, ood_val, ood_test


def build_prompt(r):
    return (
        "Legal task. Answer accurately and concisely. Do not invent statutes, cases or authorities.\n\n"
        f"Task type: {r['task_type']} | Jurisdiction: {r['jurisdiction']}\n\n"
        f"{r['instruction']}\n\n{r['prompt']}"
    ).strip()


tok = None      # generator tokenizer
etok = None     # evidence tokenizer


def encode_prompt(r):
    text = tok.apply_chat_template(
        [{"role": "user", "content": build_prompt(r)}],
        tokenize=False, add_generation_prompt=True,
    )
    ids = tok(text, add_special_tokens=False).input_ids
    if len(ids) > MAX_PROMPT_TOKENS:                 # keep the start + the question at the end
        ids = ids[:16] + ids[-(MAX_PROMPT_TOKENS - 16):]
    return ids


class LegalDS(Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        p = encode_prompt(r)
        a = tok(r["answer"], add_special_tokens=False, truncation=True,
                max_length=MAX_ANS_TOKENS - 1).input_ids + [tok.eos_token_id]
        ev = etok(build_prompt(r), truncation=True, max_length=EVID_TOKENS)["input_ids"]
        return {"ids": p + a, "labels": [-100] * len(p) + a, "ev": ev,
                "group": r["group"], "ttype": r["task_type"]}


def collate(items):
    B = len(items)
    L = max(len(x["ids"]) for x in items)
    LE = max(len(x["ev"]) for x in items)
    ids = torch.full((B, L), tok.pad_token_id, dtype=torch.long)
    mask = torch.zeros((B, L), dtype=torch.long)
    lab = torch.full((B, L), -100, dtype=torch.long)
    ev = torch.full((B, LE), etok.pad_token_id, dtype=torch.long)
    evm = torch.zeros((B, LE), dtype=torch.long)
    for i, x in enumerate(items):
        n, m = len(x["ids"]), len(x["ev"])
        ids[i, :n] = torch.tensor(x["ids"])
        mask[i, :n] = 1
        lab[i, :n] = torch.tensor(x["labels"])
        ev[i, :m] = torch.tensor(x["ev"])
        evm[i, :m] = 1
    return {"ids": ids, "mask": mask, "labels": lab, "ev": ev, "evm": evm,
            "groups": [x["group"] for x in items], "ttypes": [x["ttype"] for x in items]}


# ---------------------------------------------------------------
# MODEL: shared frozen LLM + n independent LoRA heads
# ---------------------------------------------------------------
class MultiLoRA(nn.Module):
    """
    Wraps a frozen linear layer with n INDEPENDENT LoRA adapters.
    The batch is laid out head-major: rows [0:B] go to head 0, [B:2B] to head 1, ...
    so all heads run in ONE forward pass (gradient-checkpoint safe, no global state).
    """

    def __init__(self, base, n, r, alpha, dropout):
        super().__init__()
        self.base = base
        self.n, self.scale = n, alpha / r
        dev = base.weight.device
        self.A = nn.Parameter(torch.empty(n, r, base.in_features, device=dev, dtype=torch.float32))
        self.B = nn.Parameter(torch.zeros(n, base.out_features, r, device=dev, dtype=torch.float32))
        with torch.no_grad():
            for i in range(n):                       # different random init per head
                nn.init.kaiming_uniform_(self.A[i], a=math.sqrt(5))
        self.drop = nn.Dropout(dropout)
        self.enabled = True

    def forward(self, x):
        y = self.base(x)
        if not self.enabled:
            return y
        nb, T, D = x.shape
        xs = self.drop(x).reshape(self.n, nb // self.n, T, D)
        h = torch.einsum("nbtd,nrd->nbtr", xs, self.A.to(x.dtype))
        d = torch.einsum("nbtr,nor->nbto", h, self.B.to(x.dtype))
        return y + (d * self.scale).reshape(nb, T, -1).to(y.dtype)


class EvidenceGate(nn.Module):
    """
    Evidence-guided trust allocation.
    alpha_{i,t} = softmax over heads of  f( h_{i,t}, mean_j h_{j,t}, evidence, head_embedding_i )
    One shared scorer -> works for any number of heads, and every head's score
    can depend on what the OTHER heads think (arbitration, not independent scoring).
    """

    def __init__(self, n, d_model, d_evid, d=512):
        super().__init__()
        self.h_proj = nn.Linear(d_model, d)
        self.ctx_proj = nn.Linear(d_model, d)
        self.e_proj = nn.Linear(d_evid, d)
        self.head_emb = nn.Parameter(torch.zeros(n, d))
        nn.init.normal_(self.head_emb, std=0.02)
        self.mlp = nn.Sequential(nn.GELU(), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))

    def forward(self, h, e, noise_std=0.0):
        # h: [n, N, D]   e: [N, De]  ->  log_alpha: [n, N]
        h = h.float()
        z = (self.h_proj(h)
             + self.ctx_proj(h.mean(0, keepdim=True))
             + self.e_proj(e.float()).unsqueeze(0)
             + self.head_emb[:, None, :])
        logits = self.mlp(z).squeeze(-1)
        if self.training and noise_std > 0:
            logits = logits + torch.randn_like(logits) * noise_std
        return F.log_softmax(logits, dim=0)


model = None
evid = None
gate = None
lora_mods = []


def build_system():
    global model, evid, gate, tok, etok, lora_mods
    tok = AutoTokenizer.from_pretrained(GENERATOR_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    etok = AutoTokenizer.from_pretrained(EVIDENCE_ENCODER_ID)

    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        GENERATOR_ID, quantization_config=bnb, device_map={"": 0},
        torch_dtype=torch.bfloat16, attn_implementation="sdpa",
    )
    for p in model.parameters():
        p.requires_grad_(False)

    # inject n LoRA heads
    for _, mod in list(model.named_modules()):
        for cn, child in list(mod.named_children()):
            if cn in TARGETS and isinstance(child, nn.Linear):
                setattr(mod, cn, MultiLoRA(child, N_HEADS, LORA_R, LORA_ALPHA, LORA_DROPOUT))
    lora_mods = [m for m in model.modules() if isinstance(m, MultiLoRA)]
    print(f"Injected {N_HEADS} heads into {len(lora_mods)} linear layers.")

    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model.config.use_cache = False

    evid = AutoModel.from_pretrained(EVIDENCE_ENCODER_ID).to(DEVICE)
    gate = EvidenceGate(N_HEADS, model.config.hidden_size, evid.config.hidden_size).to(DEVICE)

    n_lora = sum(m.A.numel() + m.B.numel() for m in lora_mods)
    print(f"Trainable: heads={n_lora/1e6:.1f}M gate={sum(p.numel() for p in gate.parameters())/1e6:.1f}M "
          f"evidence_enc={sum(p.numel() for p in evid.parameters())/1e6:.1f}M")


def set_lora(enabled):
    for m in lora_mods:
        m.enabled = enabled


def evid_vector(ev_ids, ev_mask):
    out = evid(input_ids=ev_ids, attention_mask=ev_mask).last_hidden_state
    m = ev_mask.unsqueeze(-1).to(out.dtype)
    return (out * m).sum(1) / m.sum(1).clamp_min(1)


def _ent(q):
    return -(q * q.clamp_min(1e-12).log()).sum(-1)


def core(batch, noise=0.0, diag=True):
    """One head-parallel forward. Returns per-head token log-probs, gate log-alphas and diagnostics."""
    n = N_HEADS
    ids, mask = batch["ids"].to(DEVICE), batch["mask"].to(DEVICE)
    labels = batch["labels"].to(DEVICE)
    B, T = ids.shape
    with torch.autocast("cuda", dtype=torch.bfloat16):
        h = model.model(input_ids=ids.repeat(n, 1), attention_mask=mask.repeat(n, 1),
                        use_cache=False).last_hidden_state                   # [nB,T,D]
        e = evid_vector(batch["ev"].to(DEVICE), batch["evm"].to(DEVICE))     # [B,De]
        tgt = labels[:, 1:]
        sel = tgt != -100                                                    # [B,T-1]
        D = h.shape[-1]
        hs = h[:, :-1].reshape(n, B, T - 1, D)[:, sel]                       # [n,N,D]
        tg = tgt[sel]                                                        # [N]
        e_tok = e[:, None, :].expand(B, T - 1, -1)[sel]                      # [N,De]
        logits = model.lm_head(hs)                                           # [n,N,V]
        N = tg.shape[0]
        logp = -F.cross_entropy(logits.float().reshape(n * N, -1), tg.repeat(n),
                                reduction="none").view(n, N)                 # [n,N]
        log_alpha = gate(hs.detach(), e_tok, noise)                          # [n,N]
    ex = {}
    if diag:
        with torch.no_grad():
            p = F.softmax(logits.detach().float(), dim=-1)                   # [n,N,V]
            ex["js"] = _ent(p.mean(0)) - _ent(p).mean(0)                     # head disagreement per token
            top = p.argmax(-1)
            ex["agree"] = (top == top[0:1]).all(0).float()                   # all heads pick same token
            ex["best"] = logp.detach().argmax(0)                             # which head was best per token
            ex["bidx"] = torch.arange(B, device=DEVICE)[:, None].expand(B, T - 1)[sel]
            del p
    return logp, log_alpha, ex


def ean_loss(logp, log_alpha):
    nll_mix = -torch.logsumexp(log_alpha + logp, dim=0).mean()   # -log sum_i alpha_i p_i(y)
    nll_heads = -logp.mean()
    mean_alpha = log_alpha.exp().mean(1)
    balance = (mean_alpha * (mean_alpha.clamp_min(1e-9).log() + math.log(N_HEADS))).sum()
    loss = nll_mix + W_HEAD_NLL * nll_heads + W_BALANCE * balance
    return loss, nll_mix.detach(), (-logp.mean(1)).detach(), mean_alpha.detach()


# ---------------------------------------------------------------
# TRAINING DIAGNOSTICS  (everything the plots need)
# ---------------------------------------------------------------
METRICS_DIR = os.path.join(OUT, "metrics")
os.makedirs(METRICS_DIR, exist_ok=True)
HISTORY = {"n_heads": N_HEADS, "smoke": SMOKE,
           "config": {"lora_r": LORA_R, "lr_lora": LR_LORA, "lr_gate": LR_GATE, "gate_noise": GATE_NOISE,
                      "w_head_nll": W_HEAD_NLL, "w_balance": W_BALANCE, "batch": BATCH, "accum": ACCUM},
           "steps": [], "val": []}


def save_history():
    with open(os.path.join(METRICS_DIR, "train_history.json"), "w", encoding="utf-8") as f:
        json.dump(make_json_safe(HISTORY), f)


@torch.no_grad()
def batch_stats(logp, la, ex):
    a = la.exp()
    return {
        "mix_nll": float((-torch.logsumexp(la + logp, 0)).mean().item()),
        "head_nll": [float(x) for x in (-logp.mean(1)).float().cpu().tolist()],
        "alpha": [float(x) for x in a.mean(1).float().cpu().tolist()],
        "alpha_entropy": float((-(a * la).sum(0)).mean().item()),
        "js": float(ex["js"].mean().item()),
        "agree": float(ex["agree"].mean().item()),
        "best_frac": [float(x) for x in F.one_hot(ex["best"], N_HEADS).float().mean(0).cpu().tolist()],
    }


def avg_stats(lst):
    out = {}
    for k in lst[0]:
        out[k] = np.mean([s[k] for s in lst], axis=0).tolist()
    return out


@torch.no_grad()
def lora_norms():
    s = torch.zeros(N_HEADS, device=DEVICE)
    for m in lora_mods:
        s += m.B.float().pow(2).sum((1, 2))
    return s.sqrt().cpu().tolist()


# ---------------------------------------------------------------
# TRAIN
# ---------------------------------------------------------------
def save_ckpt():
    torch.save({
        "lora": {i: {"A": m.A.detach().cpu(), "B": m.B.detach().cpu()} for i, m in enumerate(lora_mods)},
        "gate": gate.state_dict(),
        "evid": evid.state_dict(),
        "n_heads": N_HEADS,
    }, CKPT)


def load_ckpt():
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    assert ck["n_heads"] == N_HEADS
    for i, m in enumerate(lora_mods):
        m.A.data.copy_(ck["lora"][i]["A"])
        m.B.data.copy_(ck["lora"][i]["B"])
    gate.load_state_dict(ck["gate"])
    evid.load_state_dict(ck["evid"])
    print("Loaded", CKPT)


@torch.no_grad()
def validate(rows, limit, split):
    model.eval(); gate.eval(); evid.eval()
    set_lora(True)
    n = N_HEADS
    loader = DataLoader(LegalDS(rows[:limit]), batch_size=2, collate_fn=collate)
    tot = {"mix": 0.0, "tok": 0, "nll": np.zeros(n), "alpha": np.zeros(n),
           "js": 0.0, "agree": 0.0, "ent": 0.0}
    mk = lambda: {"tok": 0, "nll": np.zeros(n), "alpha": np.zeros(n), "mix": 0.0}
    grp, typ = defaultdict(mk), defaultdict(mk)
    for b in loader:
        logp, la, ex = core(b, 0.0)
        a = la.exp()
        nll = (-logp).cpu().numpy()                                  # [n,N]
        al = a.cpu().numpy()
        mixt = (-torch.logsumexp(la + logp, 0)).cpu().numpy()        # [N]
        bidx = ex["bidx"].cpu().numpy()
        tot["mix"] += mixt.sum()
        tot["nll"] += nll.sum(1)
        tot["alpha"] += al.sum(1)
        tot["js"] += float(ex["js"].sum())
        tot["agree"] += float(ex["agree"].sum())
        tot["ent"] += float((-(a * la).sum(0)).sum())
        tot["tok"] += logp.shape[1]
        for bi in np.unique(bidx):
            msk = bidx == bi
            for store, name in ((grp, b["groups"][bi]), (typ, b["ttypes"][bi])):
                s = store[name]
                s["tok"] += int(msk.sum())
                s["nll"] += nll[:, msk].sum(1)
                s["alpha"] += al[:, msk].sum(1)
                s["mix"] += float(mixt[msk].sum())
    t = max(tot["tok"], 1)

    def fin(store):
        return {
            k: {
                "tok": int(v["tok"]),
                "head_nll": [float(x) for x in (v["nll"] / max(v["tok"], 1)).round(5)],
                "alpha": [float(x) for x in (v["alpha"] / max(v["tok"], 1)).round(5)],
                "mix_nll": float(v["mix"] / max(v["tok"], 1))
            } 
            for k, v in store.items()
        }

    return {"split": split, "mix_nll": float(tot["mix"] / t), "head_nll": (tot["nll"] / t).round(5).tolist(),
            "alpha": (tot["alpha"] / t).round(5).tolist(), "js": float(tot["js"] / t),
            "agree": float(tot["agree"] / t), "alpha_entropy": float(tot["ent"] / t),
            "by_group": fin(grp), "by_ttype": fin(typ), "tokens": int(tot["tok"])}


def train(train_rows, iid_val, ood_val):
    seed_all(SEED)
    lora_params = [p for m in lora_mods for p in (m.A, m.B)]
    opt = torch.optim.AdamW([
        {"params": lora_params, "lr": LR_LORA},
        {"params": gate.parameters(), "lr": LR_GATE},
        {"params": evid.parameters(), "lr": LR_EVID},
    ], weight_decay=0.0)

    loader = DataLoader(LegalDS(train_rows), batch_size=BATCH, shuffle=True,
                        collate_fn=collate, drop_last=True)
    total = max(1, EPOCHS * len(loader) // ACCUM)
    warm = max(1, int(WARMUP * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    HISTORY["total_steps"] = total

    best = float("inf")
    step = micro = 0
    print(f"Training: {len(loader)} micro-batches/epoch, {total} optimizer steps")

    # step 0 = before any learning: all heads identical (B=0), the starting point of the emergence story
    for split, rows_ in (("iid", iid_val), ("ood", ood_val)):
        v = validate(rows_, VAL_ROWS, split)
        v["step"] = 0
        HISTORY["val"].append(v)
    save_history()
    model.train(); gate.train(); evid.train()
    set_lora(True)

    pending, losses = [], []
    for ep in range(EPOCHS):
        for b in loader:
            noise = GATE_NOISE * max(0.0, 1 - step / total)
            logp, la, ex = core(b, noise)
            loss, nmix, nheads, malpha = ean_loss(logp, la)
            (loss / ACCUM).backward()
            pending.append(batch_stats(logp, la, ex))
            losses.append(float(loss.detach()))
            micro += 1
            if micro % ACCUM:
                continue
            lr_now = opt.param_groups[0]["lr"]
            torch.nn.utils.clip_grad_norm_(
                lora_params + list(gate.parameters()) + list(evid.parameters()), 1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            step += 1
            rec = avg_stats(pending)
            rec.update({"step": step, "loss": float(np.mean(losses)), "lr": lr_now, "noise": noise,
                        "lora_norm": lora_norms()})
            HISTORY["steps"].append(rec)
            pending, losses = [], []
            if step % 10 == 0 or SMOKE:
                print(f"step {step}/{total} mix_nll={rec['mix_nll']:.4f} "
                      f"head_nll={np.round(rec['head_nll'], 3)} alpha={np.round(rec['alpha'], 3)} "
                      f"js={rec['js']:.4f} agree={rec['agree']:.3f}")
            if step % EVAL_EVERY == 0 or step == total:
                a = validate(iid_val, VAL_ROWS, "iid")
                o = validate(ood_val, VAL_ROWS, "ood")
                a["step"], o["step"] = step, step
                HISTORY["val"] += [a, o]
                save_history()
                print(f"  VAL iid mix={a['mix_nll']:.4f} heads={a['head_nll']} alpha={a['alpha']}")
                print(f"  VAL ood mix={o['mix_nll']:.4f} heads={o['head_nll']} alpha={o['alpha']}")
                crit = 0.5 * (a["mix_nll"] + o["mix_nll"])
                if crit < best:
                    best = crit
                    save_ckpt()
                    print("  saved best checkpoint")
                model.train(); gate.train(); evid.train()
    save_history()
    if not os.path.exists(CKPT):
        save_ckpt()


# ---------------------------------------------------------------
# DECODING:  Y = sum_i alpha_i * P_i   (token level, probability space)
# ---------------------------------------------------------------
def _tok_str(i):
    s = tok.convert_ids_to_tokens(int(i))
    return s.replace("▁", " ")


@torch.no_grad()
def decode(row, mode):
    """
    modes:
      base    - original Mistral (all heads switched off)
      solo    - every head decodes ITS OWN greedy sequence (returns n texts)
      uniform - fusion with alpha = 1/n
      gate    - EAN: evidence-guided gated fusion  (also records a per-token trace)
    """
    n = N_HEADS
    nb = 1 if mode == "base" else n
    set_lora(mode != "base")
    prompt = encode_prompt(row)
    ids = torch.tensor([prompt] * nb, device=DEVICE)
    ev = etok(build_prompt(row), truncation=True, max_length=EVID_TOKENS, return_tensors="pt").to(DEVICE)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        e = evid_vector(ev["input_ids"], ev["attention_mask"])               # [1,De]

    outs = [[] for _ in range(nb)]
    done = [False] * nb
    trace = {"tokens": [], "alpha": [], "head_p": [], "js": []}
    past, cur = None, ids
    for _ in range(MAX_NEW):
        rec = None
        with torch.autocast("cuda", dtype=torch.bfloat16):
            o = model.model(input_ids=cur, past_key_values=past, use_cache=True)
            past = o.past_key_values
            h = o.last_hidden_state[:, -1]                                    # [nb,D]
            probs = F.softmax(model.lm_head(h).float(), dim=-1)               # [nb,V]
            if mode in ("base", "solo"):
                nxt = probs.argmax(-1)
            else:
                if mode == "gate":
                    a = gate(h[:, None, :], e, 0.0).exp()                     # [n,1]
                else:
                    a = torch.full((n, 1), 1.0 / n, device=DEVICE)
                fused = (a * probs).sum(0)                                    # Y = sum alpha_i P_i
                chosen = fused.argmax()
                nxt = chosen.repeat(nb)
                if mode == "gate":
                    js = float(_ent(probs.mean(0)) - _ent(probs).mean())
                    rec = (int(chosen), a.squeeze(-1).float().cpu().numpy(),
                           probs[:, chosen].float().cpu().numpy(), js)
        if rec is not None and rec[0] != tok.eos_token_id:
            trace["tokens"].append(_tok_str(rec[0]))
            trace["alpha"].append([round(float(x), 4) for x in rec[1]])
            trace["head_p"].append([round(float(x), 4) for x in rec[2]])
            trace["js"].append(round(rec[3], 5))
        for i in range(nb):
            if not done[i]:
                t = int(nxt[i])
                if t == tok.eos_token_id:
                    done[i] = True
                else:
                    outs[i].append(t)
        if all(done):
            break
        cur = nxt[:, None]
    texts = [clean(tok.decode(x, skip_special_tokens=True)) for x in outs]
    info = {}
    if trace["alpha"]:
        A = np.array(trace["alpha"])                                          # [steps,n]
        info = {"mean_alpha": A.mean(0).round(4).tolist(),
                "alpha_entropy": float((-(A * np.log(A + 1e-9)).sum(1)).mean()),
                "mean_js": float(np.mean(trace["js"])),
                "trace": trace}
    return texts, info


def evaluate(rows, name):
    print(f"\nEvaluating on {name} ({min(EVAL_ROWS, len(rows))} rows)...")
    model.eval(); gate.eval(); evid.eval()
    methods = ["base"] + [f"head{i}" for i in range(N_HEADS)] + ["uniform_fusion", "EAN_gated", "oracle_best_head"]
    agg = {m: defaultdict(list) for m in methods}
    saved, alpha_stats = [], []
    for k, r in enumerate(rows[:EVAL_ROWS]):
        ref = r["answer"]
        base_t, _ = decode(r, "base")
        solo_t, _ = decode(r, "solo")
        uni_t, _ = decode(r, "uniform")
        gate_t, info = decode(r, "gate")
        preds = {"base": base_t[0], "uniform_fusion": uni_t[0], "EAN_gated": gate_t[0]}
        for i in range(N_HEADS):
            preds[f"head{i}"] = solo_t[i]
        head_scores = [score(solo_t[i], ref) for i in range(N_HEADS)]
        best_i = int(np.argmax([s["f1"] for s in head_scores]))
        preds["oracle_best_head"] = solo_t[best_i]
        sc = {m: score(p, ref) for m, p in preds.items()}
        for m in sc:
            for kk, v in sc[m].items():
                agg[m][kk].append(v)
        info = dict(info)
        info["head_f1"] = [s["f1"] for s in head_scores]
        info["head_f1_std"] = float(np.std(info["head_f1"]))
        info["solo_agree"] = float(len(set(solo_t)) == 1)
        if "mean_alpha" in info:
            alpha_stats.append(info)
        saved.append({"group": r["group"], "ttype": r["task_type"], "reference": ref,
                      "preds": preds, "scores": sc, "gate": info})
        if (k + 1) % 10 == 0:
            print(f"  {k + 1}/{min(EVAL_ROWS, len(rows))}")
    table = {m: {kk: float(np.mean(v)) for kk, v in d.items()} for m, d in agg.items()}
    if alpha_stats:
        table["_gate_usage"] = {
            "mean_alpha_per_head": np.mean([a["mean_alpha"] for a in alpha_stats], 0).round(3).tolist(),
            "mean_alpha_entropy(max=%.3f)" % math.log(N_HEADS): float(np.mean([a["alpha_entropy"] for a in alpha_stats])),
        }
    with open(os.path.join(RESULTS, f"{name}.json"), "w", encoding="utf-8") as f:
        json.dump(make_json_safe({"n_heads": N_HEADS, "metrics": table, "samples": saved}), f, ensure_ascii=False)
    print(f"\n=== {name} ===")
    print(f"{'method':20s} {'rougeL':>8s} {'f1':>8s} {'em':>8s}")
    for m in methods:
        t = table[m]
        print(f"{m:20s} {t['rougeL']:8.4f} {t['f1']:8.4f} {t['em']:8.4f}")
    print("gate usage:", table.get("_gate_usage"))
    return table


# ---------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["train", "eval", "all"], default="all")
    if "ipykernel" in sys.modules:
        args = ap.parse_args([])
    else:
        args = ap.parse_args()
        
    assert torch.cuda.is_available(), "CUDA GPU required (A100 recommended)."
    seed_all(SEED)
    print("GPU:", torch.cuda.get_device_name(0), "| SMOKE =", SMOKE)

    build_system()
    rows = load_rows()
    train_rows, iid_val, iid_test, ood_val, ood_test = make_splits(rows)
    del rows
    gc.collect()

    if args.stage in ("train", "all"):
        train(train_rows, iid_val, ood_val)
    if args.stage in ("eval", "all"):
        load_ckpt()
        evaluate(iid_test, "iid_test_same_sources")
        evaluate(ood_test, "ood_test_unseen_sources")
        print("\nNow run:  python make_plots.py --out ./ean_legal_out   (figures -> ean_legal_out/figures)")


if __name__ == "__main__":
    main()