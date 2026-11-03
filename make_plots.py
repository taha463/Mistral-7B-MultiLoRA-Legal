"""
make_plots.py - publication-grade diagnostics for EAN training and evaluation
==============================================================================
Reads what ean_legal.py writes:
    <out>/metrics/train_history.json
    <out>/results/iid_test_same_sources.json
    <out>/results/ood_test_unseen_sources.json
and writes PNG + PDF figures to <out>/figures.

    python make_plots.py --out ./ean_legal_out

Figures
  01_training_curves      loss, per-head NLL, validation gain of mixture over best head, schedules
  02_head_emergence       gate trust, best-head share, head disagreement, agreement, gate entropy, weight growth
  03_specialisation       what each head is good at, whether the gate follows it (sources + task types)
  04_token_gate_heatmap   per-token trust the gate gave each head, word by word
  05_method_comparison    base / heads / uniform / EAN / oracle with bootstrap CIs and paired EAN-vs-uniform test
  06_gate_vs_disagreement does arbitration help more when heads disagree?
  07_per_source           per-source dumbbell: base -> EAN, with every other method marked

Needs only numpy + matplotlib.
"""
import os
import json
import math
import argparse
import warnings

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from matplotlib.patches import Rectangle, Patch
from matplotlib.lines import Line2D

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------- style
INK, MUTED, FAINT, GRID = "#0F172A", "#64748B", "#CBD5E1", "#E8EDF3"
EAN_C = "#312E81"
HEAD_C = ["#E4572E", "#12A4A0", "#F2A900", "#C2185B", "#3B82F6", "#7CB518", "#8E44AD", "#795548"]
METHOD_C = {"base": "#94A3B8", "uniform_fusion": "#475569", "EAN_gated": EAN_C, "oracle_best_head": "#047857"}
METHOD_LABEL = {"base": "Base\nMistral", "uniform_fusion": "Uniform\nfusion", "EAN_gated": "EAN\n(gated)",
                "oracle_best_head": "Oracle\nbest head"}
GOOD, BAD, NEUTRAL = "#0F766E", "#C2410C", "#94A3B8"
DIV = LinearSegmentedColormap.from_list("div", ["#C2410C", "#F6D5C0", "#FFFFFF", "#BDE5E1", "#0F766E"])
SEQ = LinearSegmentedColormap.from_list("seq", ["#FFFFFF", "#C7C5F4", "#4F46E5", "#1E1B4B"])

plt.rcParams.update({
    "font.family": ["DejaVu Sans", "sans-serif"], "font.size": 9.5, "axes.edgecolor": FAINT,
    "axes.labelcolor": MUTED, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlecolor": INK,
    "axes.titlelocation": "left", "axes.titlepad": 10, "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": .8, "axes.axisbelow": True,
    "legend.frameon": False, "legend.fontsize": 8.5, "figure.facecolor": "white", "savefig.facecolor": "white",
    "pdf.fonttype": 42,
})


def hc(i):
    return HEAD_C[i % len(HEAD_C)]


def head_name(i):
    return f"Head {i}"


def header(fig, title, sub, y=.975):
    fig.text(.045, y, title, fontsize=17, fontweight="bold", color=INK, ha="left", va="top")
    fig.text(.045, y - .042, sub, fontsize=10.2, color=MUTED, ha="left", va="top")


def ttl(ax, title, sub=None, pad=24, subgap=8):
    ax.set_title(title, loc="left", pad=pad)
    if sub:
        ax.annotate(sub, xy=(0, 1), xycoords="axes fraction", xytext=(0, subgap), textcoords="offset points",
                    fontsize=8.3, color=MUTED, va="bottom", ha="left")


def badge(ax, letter, title, sub=None):
    ttl(ax, f"{letter}   {title}", sub)


def ema(x, k=None):
    x = np.asarray(x, float)
    if len(x) < 4:
        return x
    k = k or max(3, len(x) // 18)
    a = 2 / (k + 1)
    out, s, w = np.empty_like(x), 0.0, 0.0
    for i, v in enumerate(x):                          # bias-corrected EMA
        s = (1 - a) * s + a * v
        w = (1 - a) * w + a
        out[i] = s / w
    return out


def save(fig, outdir, name):
    os.makedirs(outdir, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(outdir, f"{name}.{ext}"), dpi=220 if ext == "png" else None, bbox_inches="tight")
    plt.close(fig)
    print("  wrote", name)


def short(s, m=34):
    s = str(s).replace("::", " / ")
    return s if len(s) <= m else s[:m - 1] + "…"


def boot_ci(x, n=2000, seed=0):
    x = np.asarray(x, float)
    if len(x) < 2:
        return float(x.mean()) if len(x) else 0.0, 0.0, 0.0
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), (n, len(x)))
    m = x[idx].mean(1)
    return float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def rank(x):
    return np.argsort(np.argsort(x)).astype(float)


def corr(a, b, spearman=False):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or a.std() == 0 or b.std() == 0:
        return float("nan")
    if spearman:
        a, b = rank(a), rank(b)
    return float(np.corrcoef(a, b)[0, 1])


def load(path):
    return json.load(open(path, encoding="utf-8")) if os.path.exists(path) else None


# ======================================================================
# 01  TRAINING CURVES
# ======================================================================
def fig_training(H, outdir):
    S, V = H["steps"], H["val"]
    if not S:
        return
    n = H["n_heads"]
    st = np.array([s["step"] for s in S])
    fig = plt.figure(figsize=(14.5, 9))
    gs = fig.add_gridspec(2, 2, hspace=.46, wspace=.2, left=.06, right=.97, top=.84, bottom=.07)

    # --- a: training mixture loss + validation
    ax = fig.add_subplot(gs[0, 0])
    mix = np.array([s["mix_nll"] for s in S])
    ax.plot(st, mix, color=EAN_C, alpha=.18, lw=1)
    ax.plot(st, ema(mix), color=EAN_C, lw=2.4, label="train (noisy gate)")
    for split, mk, ls in (("iid", "o", "-"), ("ood", "s", "--")):
        v = sorted([x for x in V if x["split"] == split], key=lambda z: z["step"])
        if v:
            ax.plot([x["step"] for x in v], [x["mix_nll"] for x in v], ls, color=INK, lw=1.4, marker=mk, ms=5,
                    mfc="white", mew=1.5, label=f"val · {'same sources' if split == 'iid' else 'unseen sources'}")
    badge(ax, "a", "Mixture loss", "-log Σ αᵢ pᵢ(y) per answer token · lower is better")
    ax.set_xlabel("optimizer step"); ax.set_ylabel("NLL / token"); ax.legend(loc="upper right")

    # --- b: per-head train NLL
    ax = fig.add_subplot(gs[0, 1])
    hn = np.array([s["head_nll"] for s in S])                   # [steps,n]
    sm = np.stack([ema(hn[:, i]) for i in range(n)], 1)
    ax.fill_between(st, sm.min(1), sm.max(1), color=FAINT, alpha=.45, lw=0, label="spread across heads")
    for i in range(n):
        ax.plot(st, hn[:, i], color=hc(i), alpha=.15, lw=.9)
        ax.plot(st, sm[:, i], color=hc(i), lw=2.1, label=head_name(i))
    ax.plot(st, ema(mix), color=INK, lw=2.4, ls=(0, (4, 2)), label="mixture (EAN)")
    badge(ax, "b", "Each head, individually", "heads are trained to be competent alone AND as a mixture")
    ax.set_xlabel("optimizer step"); ax.set_ylabel("NLL / token"); ax.legend(ncol=2, loc="upper right")

    # --- c: validation gain of the mixture over the best single head
    ax = fig.add_subplot(gs[1, 0])
    txt = []
    for split, ls, mk in (("iid", "-", "o"), ("ood", "--", "s")):
        v = sorted([x for x in V if x["split"] == split], key=lambda z: z["step"])
        if len(v) < 1:
            continue
        x = np.array([a["step"] for a in v])
        best = np.array([min(a["head_nll"]) for a in v])
        avg = np.array([np.mean(a["head_nll"]) for a in v])
        mx = np.array([a["mix_nll"] for a in v])
        ax.plot(x, best, ls, color=MUTED, lw=1.6, marker=mk, ms=4.5, mfc="white", mew=1.3)
        ax.plot(x, mx, ls, color=EAN_C, lw=2.2, marker=mk, ms=5, mfc="white", mew=1.5)
        ax.fill_between(x, mx, best, where=best >= mx, color=EAN_C, alpha=.10, lw=0)
        txt.append(f"{'same' if split == 'iid' else 'unseen'}: mixture beats best head by {best[-1] - mx[-1]:+.4f} nats")
    lo_, hi_ = ax.get_ylim(); ax.set_ylim(lo_, hi_ + .30 * (hi_ - lo_))
    ax.legend(handles=[Line2D([], [], color=EAN_C, lw=2.2, label="mixture (EAN)"),
                       Line2D([], [], color=MUTED, lw=1.6, label="best single head"),
                       Line2D([], [], color=INK, lw=1.2, ls="-", label="solid = same sources"),
                       Line2D([], [], color=INK, lw=1.2, ls="--", label="dashed = unseen sources")],
              loc="upper right", ncol=2)
    badge(ax, "c", "Does arbitration beat the best head?", " · ".join(txt) if txt else None)
    ax.set_xlabel("optimizer step"); ax.set_ylabel("validation NLL / token")

    # --- d: schedules
    ax = fig.add_subplot(gs[1, 1])
    ax.plot(st, [s["lr"] for s in S], color=EAN_C, lw=2.2)
    ax.set_ylabel("LoRA learning rate", color=EAN_C); ax.tick_params(axis="y", colors=EAN_C)
    ax2 = ax.twinx(); ax2.grid(False); ax2.spines["right"].set_visible(True); ax2.spines["right"].set_color(FAINT)
    ax2.plot(st, [s["noise"] for s in S], color=HEAD_C[0], lw=2.2, ls="--")
    ax2.set_ylabel("gate noise σ (annealed to 0)", color=HEAD_C[0]); ax2.tick_params(axis="y", colors=HEAD_C[0])
    vs = sorted({x["step"] for x in V})
    ax.plot(vs, [ax.get_ylim()[0]] * len(vs), "|", color=INK, ms=9, mew=1.6, clip_on=False)
    badge(ax, "d", "What the run was doing", "ticks at the bottom = validation passes · noise forces the gate to explore early")
    ax.set_xlabel("optimizer step")

    header(fig, "Training curves",
           f"{n} independent LoRA heads on one frozen Mistral-7B, mixed by an evidence-guided gate · "
           f"{len(st)} optimizer steps" + (" · SMOKE RUN" if H.get("smoke") else ""))
    save(fig, outdir, "01_training_curves")


# ======================================================================
# 02  HEAD EMERGENCE
# ======================================================================
def emergence_step(st, js):
    j = ema(js)
    if j.max() <= 0:
        return None
    k = np.argmax(j >= .25 * j.max())
    return int(st[k])


def fig_emergence(H, outdir):
    S = H["steps"]
    if not S:
        return
    n = H["n_heads"]
    st = np.array([s["step"] for s in S])
    fig = plt.figure(figsize=(14.5, 11.2))
    gs = fig.add_gridspec(3, 2, hspace=.58, wspace=.2, left=.06, right=.97, top=.87, bottom=.06)

    def stack(key, ax, ylabel):
        A = np.stack([ema([s[key][i] for s in S]) for i in range(n)])
        A = A / A.sum(0, keepdims=True)
        ax.stackplot(st, *A, colors=[hc(i) for i in range(n)], alpha=.92, edgecolor="white", linewidth=.7)
        for k in range(1, n):
            ax.axhline(k / n, color="white", lw=1.1, ls=(0, (2, 3)), alpha=.9)
        xr = st[-1] - st[0]
        ax.set_xlim(st[0], st[-1] + .13 * xr)
        base = np.concatenate([[0], np.cumsum(A[:, -1])])
        for i in range(n):
            ax.text(st[-1] + .015 * xr, (base[i] + base[i + 1]) / 2, f"{head_name(i)}  {A[i, -1] * 100:.0f}%",
                    color=hc(i), fontsize=8.8, fontweight="bold", va="center")
        ax.set_ylim(0, 1); ax.set_ylabel(ylabel); ax.set_xlabel("optimizer step"); ax.grid(False)

    ax = fig.add_subplot(gs[0, 0]); stack("alpha", ax, "share of gate trust α")
    badge(ax, "a", "Who the gate trusts", "white dashes = perfectly uniform split · bands drifting from them = real arbitration")
    ax = fig.add_subplot(gs[0, 1]); stack("best_frac", ax, "share of tokens where head was best")
    badge(ax, "b", "Who is actually best", "per-token winner among heads · compare with (a): does trust follow skill?")

    js = np.array([s["js"] for s in S])
    e_step = emergence_step(st, js)
    ax = fig.add_subplot(gs[1, 0])
    ax.fill_between(st, 0, ema(js), color=EAN_C, alpha=.14, lw=0)
    ax.plot(st, js, color=EAN_C, alpha=.18, lw=.9); ax.plot(st, ema(js), color=EAN_C, lw=2.4)
    ax.set_ylim(0, max(ema(js).max() * 1.35, 1e-3))
    if e_step is not None:
        ax.axvline(e_step, color=HEAD_C[0], lw=1.4, ls="--")
        ax.text(e_step, ax.get_ylim()[1] * .96, f"  heads start to differ\n  (step {e_step})", color=HEAD_C[0], fontsize=8.6, fontweight="bold", va="top")
    badge(ax, "c", "Head disagreement (Jensen-Shannon)", "0 = all heads output the same distribution · identical at init because LoRA B = 0")
    ax.set_xlabel("optimizer step"); ax.set_ylabel("JS divergence (nats)")

    ag = np.array([s["agree"] for s in S])
    ax = fig.add_subplot(gs[1, 1])
    ax.fill_between(st, ema(ag), 1, color=HEAD_C[0], alpha=.10, lw=0)
    ax.plot(st, ag, color=HEAD_C[0], alpha=.2, lw=.9); ax.plot(st, ema(ag), color=HEAD_C[0], lw=2.4)
    if e_step is not None:
        ax.axvline(e_step, color=HEAD_C[0], lw=1.2, ls="--")
    ax.set_ylim(min(.0, ema(ag).min() - .05) if ema(ag).min() < .5 else ema(ag).min() - .08, 1.02)
    badge(ax, "d", "Do the heads pick the same word?", "share of tokens where every head's top-1 choice is identical · shaded = disagreement")
    ax.set_xlabel("optimizer step"); ax.set_ylabel("top-1 agreement")

    ent = np.array([s["alpha_entropy"] for s in S]) / math.log(n)
    ax = fig.add_subplot(gs[2, 0])
    ax.axhspan(0, .25, color=HEAD_C[0], alpha=.10, lw=0); ax.axhspan(.9, 1.05, color=HEAD_C[2], alpha=.15, lw=0)
    ax.text(st[-1], .02, "gate collapsed onto a single head ", ha="right", color=HEAD_C[0], fontsize=8.5, fontweight="bold")
    ax.text(st[-1], .97, "gate is indifferent (≈ uniform) ", ha="right", color="#9A6B00", fontsize=8.5, fontweight="bold")
    ax.plot(st, ent, color=EAN_C, alpha=.2, lw=.9); ax.plot(st, ema(ent), color=EAN_C, lw=2.6)
    ax.set_ylim(0, 1.05)
    badge(ax, "e", "Is the gate actually deciding?", "normalised entropy of α per token · the healthy zone is in between")
    ax.set_xlabel("optimizer step"); ax.set_ylabel("H(α) / ln n")

    ax = fig.add_subplot(gs[2, 1])
    ln = np.array([s["lora_norm"] for s in S]) if "lora_norm" in S[0] else None
    if ln is not None:
        for i in range(n):
            ax.plot(st, ln[:, i], color=hc(i), lw=2.2, label=head_name(i))
        ax.legend(ncol=min(n, 4), loc="upper left")
    badge(ax, "f", "How far each head moved from the base model", "‖B‖ of the LoRA update, all layers · a head that stays small is a dead head")
    ax.set_xlabel("optimizer step"); ax.set_ylabel("‖ΔW‖ proxy")

    sub = (f"Heads start identical and are never told what to specialise in · "
           + (f"first measurable divergence at step {e_step}" if e_step is not None else "no divergence detected yet"))
    header(fig, "How the heads emerge and how the gate behaves", sub)
    save(fig, outdir, "02_head_emergence")


# ======================================================================
# 03  SPECIALISATION  (what makes heads different, does the gate follow)
# ======================================================================
def merged_final(V, key):
    last = {}
    for v in V:
        if v["split"] not in last or v["step"] > last[v["split"]]["step"]:
            last[v["split"]] = v
    acc = {}
    for v in last.values():
        for k, d in v[key].items():
            a = acc.setdefault(k, {"tok": 0, "nll": 0, "alpha": 0})
            a["nll"] = a["nll"] + np.array(d["head_nll"]) * d["tok"]
            a["alpha"] = a["alpha"] + np.array(d["alpha"]) * d["tok"]
            a["tok"] += d["tok"]
    return {k: (a["nll"] / a["tok"], a["alpha"] / a["tok"], a["tok"]) for k, a in acc.items() if a["tok"] > 0}


def fig_specialisation(H, outdir):
    V = H["val"]
    n = H["n_heads"]
    if not V:
        return
    tables = []
    for key, top, label in (("by_group", 14, "source"), ("by_ttype", 8, "task type")):
        d = merged_final(V, key)
        keys = sorted(d, key=lambda k: -d[k][2])[:top]
        if not keys:
            continue
        nll = np.stack([d[k][0] for k in keys]); al = np.stack([d[k][1] for k in keys])
        adv = nll.mean(1, keepdims=True) - nll                    # >0: better than the average head
        order = np.lexsort((-adv.max(1), adv.argmax(1)))
        tables.append((label, [keys[i] for i in order], adv[order], al[order], np.array([d[k][2] for k in keys])[order]))
    if not tables:
        return
    rows = [len(t[1]) for t in tables]
    fig = plt.figure(figsize=(15.5, 3.2 + .38 * sum(rows)))
    gs = fig.add_gridspec(len(tables), 3, width_ratios=[1, 1, 1.05], height_ratios=rows, hspace=.5, wspace=.24,
                          left=.20, right=.97, top=.84, bottom=.07)
    allx, ally, allc, alls = [], [], [], []
    vmax = max(np.abs(t[2]).max() for t in tables) or 1e-6
    for r, (label, keys, adv, al, tk) in enumerate(tables):
        for c, (M, cmap, norm, title_, fmt) in enumerate((
                (adv, DIV, (-vmax, vmax), f"Head skill by {label}", "{:+.3f}"),
                (al, SEQ, (0, max(.55, al.max())), f"Gate trust by {label}", "{:.0%}"))):
            ax = fig.add_subplot(gs[r, c]); ax.grid(False)
            im = ax.imshow(M, cmap=cmap, vmin=norm[0], vmax=norm[1], aspect="auto")
            ax.set_xticks(range(n)); ax.set_xticklabels([head_name(i) for i in range(n)], fontsize=8.8)
            for t, i in zip(ax.get_xticklabels(), range(n)):
                t.set_color(hc(i)); t.set_fontweight("bold")
            ax.xaxis.tick_top(); ax.tick_params(length=0)
            ax.set_yticks(range(len(keys)))
            ax.set_yticklabels([short(k) if c == 0 else "" for k in keys], fontsize=8.6)
            for s in ax.spines.values():
                s.set_visible(False)
            for i in range(M.shape[0]):
                for j in range(M.shape[1]):
                    v = M[i, j]
                    dark = (abs(v) > .6 * norm[1]) if c == 0 else (v > .5 * norm[1])
                    ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=7.8,
                            color="white" if dark else INK, fontweight="bold" if (c == 0 and v == M[i].max()) or (c == 1 and v == M[i].max()) else "normal")
            if r == 0:
                ttl(ax, title_, "NLL gap vs average head (nats, + = better)" if c == 0 else "mean α given to that head",
                    pad=40, subgap=24)
            else:
                ttl(ax, title_, None, pad=26)
            if c == 0:
                ax.set_ylabel(label + "s", fontsize=9, color=INK, fontweight="bold")
        allx += adv.ravel().tolist(); ally += (al - 1 / n).ravel().tolist()
        allc += [j for _ in range(adv.shape[0]) for j in range(n)]
        alls += [np.log10(max(t, 10)) for t in tk for _ in range(n)]

    # scatter: does the gate follow skill?
    ax = fig.add_subplot(gs[:, 2])
    allx, ally, allc, alls = map(np.array, (allx, ally, allc, alls))
    ax.axhline(0, color=FAINT, lw=1); ax.axvline(0, color=FAINT, lw=1)
    for j in range(n):
        m = allc == j
        ax.scatter(allx[m], ally[m], s=18 + 22 * (alls[m] - alls.min()), color=hc(j), alpha=.78, edgecolor="white", lw=.7, label=head_name(j))
    r = corr(allx, ally); rs = corr(allx, ally, True)
    if np.isfinite(r) and allx.std() > 0:
        k, b = np.polyfit(allx, ally, 1)
        xs = np.linspace(allx.min(), allx.max(), 50)
        ax.plot(xs, k * xs + b, color=EAN_C, lw=2.2)
        ax.text(.03, .97, f"Pearson r = {r:.2f}\nSpearman ρ = {rs:.2f}\nn = {len(allx)} (source/head cells)",
                transform=ax.transAxes, va="top", fontsize=9.4, color=EAN_C, fontweight="bold")
    ax.set_xlabel("head skill  (NLL gap vs average head, nats)"); ax.set_ylabel("gate trust  (α − 1/n)")
    ttl(ax, "Does the gate follow real skill?", "up-and-right = gate trusts heads that are actually better", pad=40, subgap=24)
    ax.yaxis.labelpad = 8
    ax.legend(loc="lower right", ncol=1, title="marker size ∝ tokens", title_fontsize=7.8)
    header(fig, "What each head learned, and whether the gate noticed",
           "Rows sorted by the head that wins them · specialisation is emergent: no head was told which domain or style to own",
           y=.985)
    fig.subplots_adjust(left=.19)
    save(fig, outdir, "03_specialisation")


# ======================================================================
# 04  PER-TOKEN GATE HEATMAP
# ======================================================================
def pick_samples(evals, k=3):
    cands = []
    for split, E in evals.items():
        for s in E["samples"]:
            tr = s.get("gate", {}).get("trace")
            if not tr or len(tr["tokens"]) < 8:
                continue
            A = np.array(tr["alpha"])
            dom = A.argmax(1)
            score = s["gate"].get("mean_js", 0) + .15 * len(set(dom.tolist()))
            cands.append((score, split, s))
    cands.sort(key=lambda c: -c[0])
    out, seen = [], set()
    for sc, sp, s in cands:
        if s["group"] in seen and len(cands) > k * 3:
            continue
        seen.add(s["group"]); out.append((sp, s))
        if len(out) == k:
            break
    return out


def disp_tok(t):
    if t.startswith("<0x") and t.endswith(">"):
        return "↵" if t == "<0x0A>" else "·"
    t = t.strip()
    t = t if t else "␣"
    return t if len(t) <= 9 else t[:8] + "…"


def blend(color, a):
    a = float(np.clip(a, 0, 1)) ** .8
    c = np.array(to_rgb(color))
    return tuple((1 - a) * np.ones(3) + a * c)


def fig_tokens(evals, outdir, n):
    picks = pick_samples(evals, 3)
    if not picks:
        return
    K, MAXT = 11, 55
    row_h = .62 + n * .30 + .34
    heights = []
    for _, s in picks:
        L = min(len(s["gate"]["trace"]["tokens"]), MAXT)
        heights.append(math.ceil(L / K) * row_h + .9)
    fig = plt.figure(figsize=(13.5, 2.2 + sum(heights) * .62))
    gs = fig.add_gridspec(len(picks), 1, height_ratios=heights, hspace=.28, left=.05, right=.97, top=.86, bottom=.04)
    for r, (split, s) in enumerate(picks):
        ax = fig.add_subplot(gs[r]); ax.set_axis_off()
        tr = s["gate"]["trace"]
        toks = tr["tokens"][:MAXT]; A = np.array(tr["alpha"][:MAXT]); J = np.array(tr["js"][:MAXT])
        jmax = max(J.max(), 1e-6)
        lines = math.ceil(len(toks) / K)
        ax.set_xlim(-.9, K); ax.set_ylim(lines * row_h, -.55)
        sc = s["scores"]
        best_head = max(sc[f"head{i}"]["f1"] for i in range(n))
        ax.set_title(f"{short(s['group'], 60)}   ·   F1:  EAN {sc['EAN_gated']['f1']:.2f}   uniform {sc['uniform_fusion']['f1']:.2f}   "
                     f"best head {best_head:.2f}   base {sc['base']['f1']:.2f}   ·   {'unseen' if split.startswith('ood') else 'seen'} sources",
                     fontsize=10, loc="left", pad=6)
        for L in range(lines):
            y0 = L * row_h
            for i in range(n):
                ax.text(-.15, y0 + .62 + i * .30 + .13, f"H{i}", ha="right", va="center", fontsize=8, color=hc(i), fontweight="bold")
            ax.text(-.15, y0 + .62 + n * .30 + .17, "JS", ha="right", va="center", fontsize=7.5, color=MUTED)
            for c in range(K):
                t = L * K + c
                if t >= len(toks):
                    break
                ax.text(c + .5, y0 + .30, disp_tok(toks[t]), ha="center", va="center", fontsize=9.2, color=INK, family="DejaVu Sans Mono")
                dom = int(A[t].argmax())
                for i in range(n):
                    a = A[t, i]
                    ax.add_patch(Rectangle((c + .04, y0 + .62 + i * .30), .92, .26, fc=blend(hc(i), a), ec="none"))
                    if i == dom:
                        ax.text(c + .5, y0 + .62 + i * .30 + .13, f"{a * 100:.0f}", ha="center", va="center",
                                fontsize=7.4, fontweight="bold", color="white" if a > .6 else INK)
                ax.add_patch(Rectangle((c + .04, y0 + .62 + n * .30 + .06), .92, .13, fc=blend("#475569", J[t] / jmax), ec="none"))
    handles = [Patch(fc=hc(i), label=f"{head_name(i)}") for i in range(n)] + [Patch(fc="#475569", label="head disagreement (JS)")]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(.045, .915), ncol=n + 1, fontsize=9)
    header(fig, "Word by word: which head did the gate trust?",
           "Cell colour = α for that head at that token (number = dominant head's %) · bottom strip = how much the heads disagreed there",
           y=.985)
    save(fig, outdir, "04_token_gate_heatmap")


# ======================================================================
# 05  METHOD COMPARISON
# ======================================================================
def method_list(n):
    return ["base"] + [f"head{i}" for i in range(n)] + ["uniform_fusion", "EAN_gated", "oracle_best_head"]


def mcolor(m):
    return hc(int(m[4:])) if m.startswith("head") else METHOD_C[m]


def mlabel(m):
    return f"Head {m[4:]}" if m.startswith("head") else METHOD_LABEL[m]


def vec(E, m, metric):
    return np.array([s["scores"][m][metric] for s in E["samples"]])


def paired_boot(a, b, n=4000, seed=1):
    d = np.asarray(a) - np.asarray(b)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), (n, len(d)))
    m = d[idx].mean(1)
    p = 2 * min((m <= 0).mean(), (m >= 0).mean())
    return float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5)), float(max(p, 1 / n))


def fig_methods(evals, outdir, n):
    ms = method_list(n)
    splits = [(k, v) for k, v in evals.items()]
    if not splits:
        return
    fig, axes = plt.subplots(2, len(splits), figsize=(7.4 * len(splits), 8.4), squeeze=False)
    fig.subplots_adjust(hspace=.5, wspace=.16, left=.07, right=.98, top=.84, bottom=.09)
    for c, (name, E) in enumerate(splits):
        for r, (metric, mname) in enumerate((("f1", "token F1"), ("rougeL", "ROUGE-L"))):
            ax = axes[r, c]
            stats = [boot_ci(vec(E, m, metric)) for m in ms]
            x = np.arange(len(ms))
            for i, m in enumerate(ms):
                mean, lo, hi = stats[i]
                ec = INK if m == "EAN_gated" else "white"
                ax.bar(i, mean, width=.72, color=mcolor(m), edgecolor=ec, lw=2 if m == "EAN_gated" else 0,
                       hatch="///" if m == "oracle_best_head" else None, alpha=1 if m != "oracle_best_head" else .85)
                ax.errorbar(i, mean, yerr=[[mean - lo], [hi - mean]], color=INK, capsize=3, lw=1.3, capthick=1.3)
                ax.text(i, hi + .004, f"{mean:.3f}", ha="center", va="bottom", fontsize=8.5, color=INK,
                        fontweight="bold" if m == "EAN_gated" else "normal")
            ax.set_xticks(x); ax.set_xticklabels([mlabel(m) for m in ms], fontsize=8.2)
            ax.grid(axis="x", visible=False)
            top = max(s[2] for s in stats)
            ax.set_ylim(0, top * 1.22)
            if "uniform_fusion" in ms:
                d, lo, hi, p = paired_boot(vec(E, "EAN_gated", metric), vec(E, "uniform_fusion", metric))
                i0, i1 = ms.index("uniform_fusion"), ms.index("EAN_gated")
                yb = top * 1.1
                ax.plot([i0, i0, i1, i1], [yb * .985, yb, yb, yb * .985], color=INK, lw=1.2)
                col = GOOD if lo > 0 else (BAD if hi < 0 else MUTED)
                ax.text((i0 + i1) / 2, yb * 1.01, f"EAN − uniform  {d:+.3f}  [{lo:+.3f}, {hi:+.3f}]  " + ("p<0.001" if p <= 2.5e-4 else f"p={p:.3f}"),
                        ha="center", va="bottom", fontsize=8.4, color=col, fontweight="bold")
            ax.set_ylabel(mname)
            if r == 0:
                ax.set_title(("Same sources, held-out rows" if name.startswith("iid") else "Unseen sources") + f"   (n={len(E['samples'])})", loc="left", pad=16)
    header(fig, "EAN against every alternative",
           "Bars = mean · whiskers = 95% bootstrap CI over test rows · paired bootstrap for EAN vs uniform fusion · oracle = best head picked with the answer in hand (upper bound)")
    save(fig, outdir, "05_method_comparison")


# ======================================================================
# 06  GATE vs DISAGREEMENT
# ======================================================================
def fig_disagreement(evals, outdir, n):
    S = [s for E in evals.values() for s in E["samples"] if s["gate"].get("trace")]
    if len(S) < 8:
        return
    js = np.array([s["gate"]["mean_js"] for s in S])
    ent = np.array([s["gate"]["alpha_entropy"] for s in S]) / math.log(n)
    f1 = {m: np.array([s["scores"][m]["f1"] for s in S]) for m in method_list(n)}
    gain = f1["EAN_gated"] - f1["uniform_fusion"]
    spread = np.array([s["gate"]["head_f1_std"] for s in S])

    fig = plt.figure(figsize=(15.5, 9.4))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.12, 1], hspace=.5, wspace=.17, left=.065, right=.97, top=.85, bottom=.07)

    # a scatter
    ax = fig.add_subplot(gs[0, 0])
    ax.axhline(0, color=FAINT, lw=1.2)
    jit = np.random.default_rng(0).normal(0, .004, len(gain))
    sc = ax.scatter(js, gain + jit, c=ent, cmap=LinearSegmentedColormap.from_list("e", ["#1E1B4B", "#6366F1", "#F2A900"]),
                    s=34, alpha=.85, edgecolor="white", lw=.6, vmin=0, vmax=1)
    qs = np.quantile(js, np.linspace(0, 1, 7))
    qs = np.unique(qs)
    cx, cm, clo, chi = [], [], [], []
    for a, b in zip(qs[:-1], qs[1:]):
        m = (js >= a) & (js <= b)
        if m.sum() >= 3:
            mm, lo, hi = boot_ci(gain[m])
            cx.append(js[m].mean()); cm.append(mm); clo.append(lo); chi.append(hi)
    if cx:
        ax.fill_between(cx, clo, chi, color=EAN_C, alpha=.14, lw=0)
        ax.plot(cx, cm, color=EAN_C, lw=2.6, marker="o", ms=5, mfc="white", mew=1.6)
    rho = corr(js, gain, True)
    ax.text(.02, .96, f"Spearman ρ(disagreement, gain) = {rho:.2f}" if np.isfinite(rho) else "", transform=ax.transAxes,
            va="top", fontsize=9.6, color=EAN_C, fontweight="bold")
    cb = fig.colorbar(sc, ax=ax, pad=.01, fraction=.035); cb.set_label("gate entropy H(α)/ln n", color=MUTED, fontsize=8.5)
    cb.outline.set_visible(False)
    badge(ax, "a", "Does arbitration pay off when heads disagree?", "each dot = one test answer · line = binned mean ± 95% CI")
    ax.set_xlabel("mean head disagreement while decoding (JS, nats)"); ax.set_ylabel("EAN F1 − uniform-fusion F1")

    # b terciles
    ax = fig.add_subplot(gs[0, 1])
    order = np.argsort(spread, kind="stable")
    groups = np.array_split(order, 3)
    names = ["heads agree", "heads partly differ", "heads disagree"]
    best_single = max(f"head{i}" for i in range(n))
    best_single = max([f"head{i}" for i in range(n)], key=lambda m: f1[m].mean())
    series = [("best_single", f"best single head ({best_single[-1] and 'Head ' + best_single[4:]})", "#94A3B8", f1[best_single]),
              ("uniform", "uniform fusion", METHOD_C["uniform_fusion"], f1["uniform_fusion"]),
              ("ean", "EAN (gated)", EAN_C, f1["EAN_gated"]),
              ("oracle", "oracle best head", METHOD_C["oracle_best_head"], f1["oracle_best_head"])]
    w = .2
    for k, (_, lab, col, v) in enumerate(series):
        for g, idx in enumerate(groups):
            mm, lo, hi = boot_ci(v[idx])
            x = g + (k - 1.5) * w
            ax.bar(x, mm, width=w * .92, color=col, hatch="///" if k == 3 else None, edgecolor="white", lw=0,
                   label=lab if g == 0 else None)
            ax.errorbar(x, mm, yerr=[[mm - lo], [hi - mm]], color=INK, capsize=2, lw=1)
    ax.set_xticks(range(3)); ax.set_xticklabels([f"{nm}\n(n={len(g)})" for nm, g in zip(names, groups)])
    ax.grid(axis="x", visible=False); ax.legend(ncol=2, loc="upper left")
    ax.set_ylim(0, ax.get_ylim()[1] * 1.12)
    badge(ax, "b", "Answers split by how much the heads disagree", "terciles of the spread of the heads' individual F1 · whiskers 95% CI")
    ax.set_ylabel("token F1")

    # c histogram of gate entropy by outcome
    ax = fig.add_subplot(gs[1, 0])
    bins = np.linspace(0, 1, 21)
    cats = [("EAN better", gain > .005, GOOD), ("tie", np.abs(gain) <= .005, NEUTRAL), ("EAN worse", gain < -.005, BAD)]
    bottom = np.zeros(len(bins) - 1)
    for lab, m, col in cats:
        h, _ = np.histogram(ent[m], bins=bins)
        ax.bar(bins[:-1], h, width=np.diff(bins), align="edge", bottom=bottom, color=col, alpha=.9, edgecolor="white", lw=.8,
               label=f"{lab}  (n={int(m.sum())})")
        bottom += h
    ax.legend(loc="upper left")
    badge(ax, "c", "Gate confidence at the moment of the outcome", "confident gate = low entropy · does EAN win more when the gate commits?")
    ax.set_xlabel("gate entropy H(α)/ln n (per answer)"); ax.set_ylabel("answers")

    # d dumbbell of win-rate by entropy band
    ax = fig.add_subplot(gs[1, 1])
    qb = np.array_split(np.argsort(ent, kind="stable"), 4)
    labs = ["most decisive\ngate", "decisive", "hesitant", "most hesitant\ngate"]
    wins = [(gain[i] > .005).mean() for i in qb]; loss = [(gain[i] < -.005).mean() for i in qb]
    for k, (w_, l_) in enumerate(zip(wins, loss)):
        ax.barh(k + .17, w_, height=.3, color=GOOD); ax.barh(k - .17, l_, height=.3, color=BAD)
        ax.text(w_ + .008, k + .17, f"{w_:.0%}", va="center", fontsize=8.8, color=GOOD, fontweight="bold")
        ax.text(l_ + .008, k - .17, f"{l_:.0%}", va="center", fontsize=8.8, color=BAD, fontweight="bold")
    ax.set_yticks(range(4)); ax.set_yticklabels([f"{l}\nmean gain {gain[i].mean():+.3f}" for l, i in zip(labs, qb)], fontsize=8.4)
    ax.grid(axis="y", visible=False); ax.invert_yaxis()
    ax.set_xlim(0, 1.12)
    ax.legend(handles=[Patch(fc=GOOD, label="EAN beats uniform"), Patch(fc=BAD, label="EAN loses to uniform")],
              loc="upper center", bbox_to_anchor=(.5, -.2), ncol=2)
    badge(ax, "d", "Win / loss rate by gate decisiveness", "answers grouped into quartiles of gate entropy")
    ax.set_xlabel("share of answers")

    header(fig, "Is the gate worth having?",
           f"{len(S)} test answers pooled across splits · the gate earns its keep only if gain rises where heads disagree and where it commits")
    save(fig, outdir, "06_gate_vs_disagreement")


# ======================================================================
# 07  PER-SOURCE DUMBBELL
# ======================================================================
def fig_sources(evals, outdir, n):
    panels = []
    for name, E in evals.items():
        by = {}
        for s in E["samples"]:
            by.setdefault(s["group"], []).append(s)
        rows = [(g, v) for g, v in by.items() if len(v) >= 3]
        if rows:
            panels.append((name, E, rows))
    if not panels:
        return
    heights = [min(len(r), 22) for _, _, r in panels]
    fig = plt.figure(figsize=(13.5, 2.6 + .36 * sum(heights)))
    gs = fig.add_gridspec(len(panels), 1, height_ratios=heights, hspace=.42, left=.30, right=.96, top=.80, bottom=.05)
    for pi, (name, E, rows) in enumerate(panels):
        ax = fig.add_subplot(gs[pi])
        bs = max([f"head{i}" for i in range(n)], key=lambda m: vec(E, m, "f1").mean())
        data = []
        for g, v in rows:
            f = lambda m: float(np.mean([s["scores"][m]["f1"] for s in v]))
            data.append((g, len(v), f("base"), f(bs), f("uniform_fusion"), f("EAN_gated"), f("oracle_best_head")))
        data.sort(key=lambda d: d[5] - d[2])
        data = data[-22:]
        y = np.arange(len(data))
        for i, (g, k, b, h, u, e, o) in enumerate(data):
            ax.plot([b, e], [i, i], color=GOOD if e >= b else BAD, lw=3, alpha=.35, solid_capstyle="round")
            ax.plot([e, o], [i, i], color=METHOD_C["oracle_best_head"], lw=1, alpha=.45, ls=":")
        ax.scatter([d[2] for d in data], y, s=62, facecolor="white", edgecolor=METHOD_C["base"], lw=2, zorder=3, label="base Mistral")
        ax.scatter([d[3] for d in data], y, s=40, marker="s", color=HEAD_C[1], zorder=3, label=f"best single head ({bs[4:]})")
        ax.scatter([d[4] for d in data], y, s=52, marker="D", color=METHOD_C["uniform_fusion"], zorder=4, label="uniform fusion")
        ax.scatter([d[5] for d in data], y, s=120, color=EAN_C, zorder=5, edgecolor="white", lw=1.4, label="EAN (gated)")
        ax.scatter([d[6] for d in data], y, s=60, marker="^", color=METHOD_C["oracle_best_head"], zorder=3, label="oracle best head")
        ax.set_yticks(y); ax.set_yticklabels([f"{short(d[0], 40)}  ·  n={d[1]}" for d in data], fontsize=8.6)
        for i, d in enumerate(data):
            gdiff = d[5] - d[2]
            ax.text(1.01, i, f"{gdiff:+.2f}", transform=ax.get_yaxis_transform(), fontsize=8.6, va="center",
                    color=GOOD if gdiff >= 0 else BAD, fontweight="bold")
        ax.set_ylim(-.7, len(data) + .35)
        ax.text(1.01, len(data) - .05, "EAN − base", transform=ax.get_yaxis_transform(), fontsize=8, color=MUTED, va="center")
        lo_x = max(0, min(min(d[2], d[3]) for d in data) - .06)
        ax.set_xlim(lo_x, min(1.02, max(d[6] for d in data) + .06))
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("mean token F1")
        ax.set_title(("Same sources, held-out rows" if name.startswith("iid") else "Unseen sources") + f"   ·   sorted by EAN gain over base", loc="left", pad=8)
        if pi == 0:
            ax.legend(ncol=5, loc="lower left", bbox_to_anchor=(0, 1.16), fontsize=8.4)
    header(fig, "Per-source breakdown",
           "Sources with ≥3 test rows · line from base to EAN: teal = EAN improved the source, orange = it hurt · small samples, read as direction not proof", y=.985)
    save(fig, outdir, "07_per_source")


# ======================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="./ean_legal_out")
    args = ap.parse_args()
    H = load(os.path.join(args.out, "metrics", "train_history.json"))
    evals = {}
    for key, fn in (("iid_test_same_sources", "iid_test_same_sources.json"), ("ood_test_unseen_sources", "ood_test_unseen_sources.json")):
        E = load(os.path.join(args.out, "results", fn))
        if E:
            evals[key] = E
    figdir = os.path.join(args.out, "figures")
    print("Writing figures to", figdir)
    n = (H or {}).get("n_heads") or next(iter(evals.values()))["n_heads"] if (H or evals) else 0
    if H:
        fig_training(H, figdir); fig_emergence(H, figdir); fig_specialisation(H, figdir)
    else:
        print("  no train_history.json - skipping figures 01-03")
    if evals:
        fig_tokens(evals, figdir, n); fig_methods(evals, figdir, n)
        fig_disagreement(evals, figdir, n); fig_sources(evals, figdir, n)
    else:
        print("  no eval results - skipping figures 04-07")


if __name__ == "__main__":
    main()
