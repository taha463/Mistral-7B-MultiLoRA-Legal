"""
draw_architecture.py - Generates publication-grade architectural diagrams for EAN
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

def create_architecture_diagram(out_dir="figures"):
    os.makedirs(out_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(15, 8.5), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    # Color Palette matching make_plots.py
    INK = "#0F172A"
    MUTED = "#64748B"
    FAINT = "#E2E8F0"
    BG_BOX = "#F8FAFC"
    FROZEN_C = "#1E293B"
    H0_C = "#E4572E"
    H1_C = "#12A4A0"
    H2_C = "#D97706"
    EVID_C = "#0284C7"
    GATE_C = "#4F46E5"
    FUSION_C = "#312E81"
    OUT_C = "#047857"

    # Helper function to draw rounded box
    def draw_box(x, y, w, h, bg_color, border_color, border_w=1.5, radius=1.5, zorder=2, alpha=1.0):
        box = FancyBboxPatch((x, y), w, h,
                             boxstyle=f"round,pad=0.2,rounding_size={radius}",
                             facecolor=bg_color, edgecolor=border_color,
                             linewidth=border_w, zorder=zorder, alpha=alpha)
        ax.add_patch(box)
        return box

    # Helper function to draw arrow
    def draw_arrow(x1, y1, x2, y2, color=MUTED, lw=1.8, style="-|>", rad=0.0):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle=style, color=color, lw=lw,
                                    shrinkA=4, shrinkB=4,
                                    connectionstyle=f"arc3,rad={rad}"))

    # Title Banner
    ax.text(50, 96, "Fine-Tuning Mistral-7B Parameter Model with Multi-LoRA",
            fontsize=16.5, fontweight="bold", color=INK, ha="center", va="top")
    ax.text(50, 92.2, "Evidence-Guided Multi-Head LoRA Fine-Tuning on 7B LLM with Token Probability Arbitration",
            fontsize=10, color=MUTED, ha="center", va="top")

    # =========================================================================
    # 1. INPUT STREAM (Left)
    # =========================================================================
    draw_box(3, 40, 14, 24, "#F1F5F9", FAINT, border_w=1.5)
    ax.text(10, 60, "Legal Input x", fontsize=11, fontweight="bold", color=INK, ha="center")
    ax.text(10, 56.5, "• Task Directive\n• Statutory Text\n• Case Facts\n• Jurisdiction",
            fontsize=8.5, color=MUTED, ha="center", va="top")

    # Split Arrows from Input
    draw_arrow(17, 56, 23, 68, color=FROZEN_C, lw=2.2) # to Mistral
    draw_arrow(17, 46, 23, 24, color=EVID_C, lw=2.2)   # to DeBERTa

    # =========================================================================
    # 2. GENERATIVE LATENT BACKBONE (Top Middle)
    # =========================================================================
    draw_box(23, 56, 19, 25, "#F8FAFC", FROZEN_C, border_w=2.0)
    ax.text(32.5, 77.5, "Shared 7B Latent Backbone", fontsize=10, fontweight="bold", color=FROZEN_C, ha="center")
    ax.text(32.5, 74.5, "Mistral-7B (7B Parameters)", fontsize=9.2, fontweight="bold", color=INK, ha="center")
    ax.text(32.5, 66, "• Frozen 4-bit NF4 Quant\n• Double Quantization\n• Flash Attention (SDPA)\n• 32 Transformer Blocks",
            fontsize=8, color=MUTED, ha="center")
    draw_box(25, 57.5, 15, 4.5, "#E2E8F0", FROZEN_C, border_w=1, radius=0.8)
    ax.text(32.5, 59.7, "[ FROZEN 7B BASE ]", fontsize=7.5, fontweight="bold", color=FROZEN_C, ha="center", va="center")

    # =========================================================================
    # 3. MULTI-LoRA INTERPRETATION HEADS (Middle-Right Top)
    # =========================================================================
    draw_box(46, 51, 23, 34, "#FAFAFA", FAINT, border_w=1.5)
    ax.text(57.5, 82, "Multi-LoRA Fine-Tuned Heads", fontsize=10, fontweight="bold", color=INK, ha="center")
    ax.text(57.5, 79.5, "3 Parallel LoRA Adapters on Mistral-7B (r=16, α=32)", fontsize=7.3, color=MUTED, ha="center")

    # Head 0
    draw_box(48, 70, 19, 7.5, "#FEF2F2", H0_C, border_w=1.6)
    ax.text(49.5, 75, "Head 0 (LoRA)", fontsize=8.5, fontweight="bold", color=H0_C, va="center")
    ax.text(49.5, 72.2, "Emergent Specialization 0", fontsize=7.5, color=MUTED, va="center")
    ax.text(65.5, 73.7, "h₀, P₀", fontsize=9, fontweight="bold", color=H0_C, ha="right", va="center")

    # Head 1
    draw_box(48, 61, 19, 7.5, "#F0FDFA", H1_C, border_w=1.6)
    ax.text(49.5, 66, "Head 1 (LoRA)", fontsize=8.5, fontweight="bold", color=H1_C, va="center")
    ax.text(49.5, 63.2, "Emergent Specialization 1", fontsize=7.5, color=MUTED, va="center")
    ax.text(65.5, 64.7, "h₁, P₁", fontsize=9, fontweight="bold", color=H1_C, ha="right", va="center")

    # Head 2
    draw_box(48, 52, 19, 7.5, "#FFFBEB", H2_C, border_w=1.6)
    ax.text(49.5, 57, "Head 2 (LoRA)", fontsize=8.5, fontweight="bold", color=H2_C, va="center")
    ax.text(49.5, 54.2, "Emergent Specialization 2", fontsize=7.5, color=MUTED, va="center")
    ax.text(65.5, 55.7, "h₂, P₂", fontsize=9, fontweight="bold", color=H2_C, ha="right", va="center")

    # Feed from Mistral to MultiLoRA
    draw_arrow(42, 73.5, 48, 73.5, color=H0_C, lw=1.8)
    draw_arrow(42, 68, 48, 64.5, color=H1_C, lw=1.8)
    draw_arrow(42, 62.5, 48, 55.5, color=H2_C, lw=1.8)

    # =========================================================================
    # 4. DECOUPLED EVIDENCE STREAM (Bottom Middle)
    # =========================================================================
    draw_box(23, 12, 19, 23, "#F0F9FF", EVID_C, border_w=1.8)
    ax.text(32.5, 31.5, "Decoupled Evidence Stream", fontsize=9.5, fontweight="bold", color=EVID_C, ha="center")
    ax.text(32.5, 28.5, "microsoft/deberta-v3-base", fontsize=9, fontweight="bold", color=INK, ha="center")
    ax.text(32.5, 21.5, "• 12-layer Bidirectional Enc\n• Disentangled Attention\n• Mean-pooled token representation",
            fontsize=7.8, color=MUTED, ha="center")
    draw_box(25, 14, 15, 4, "#E0F2FE", EVID_C, border_w=1, radius=0.8)
    ax.text(32.5, 16, "Evidence Vector  e ∈ ℝ⁷⁶⁸", fontsize=7.8, fontweight="bold", color=EVID_C, ha="center", va="center")

    # =========================================================================
    # 5. EVIDENCE-GUIDED TRUST GATING (Middle Bottom)
    # =========================================================================
    draw_box(46, 12, 23, 31, "#EEF2FF", GATE_C, border_w=2.0)
    ax.text(57.5, 39.5, "Evidence-Guided Trust Gating", fontsize=10, fontweight="bold", color=GATE_C, ha="center")
    ax.text(57.5, 36.5, "EvidenceGate Network (d = 512)", fontsize=8.5, fontweight="bold", color=INK, ha="center")
    ax.text(57.5, 29.5, "zᵢ = W_h hᵢ + W_ctx h̄ + W_e e + embᵢ\nαᵢ,ₜ = softmax( MLP(zᵢ) + ϵₜ )\nObserves all heads simultaneously",
            fontsize=7.8, color=MUTED, ha="center")

    draw_box(48, 14.5, 19, 6.5, "#E0E7FF", GATE_C, border_w=1.2, radius=1)
    ax.text(57.5, 18, "Dynamic Trust Allocation", fontsize=8, fontweight="bold", color=GATE_C, ha="center")
    ax.text(57.5, 15.8, "α = [α₀, α₁, α₂]  ·  Σ αᵢ = 1", fontsize=8.5, fontweight="bold", color=INK, ha="center")

    # Arrows into Gate
    draw_arrow(42, 23, 46, 23, color=EVID_C, lw=2.2) # e vector
    # h vectors from Heads down into Gate
    draw_arrow(57.5, 51, 57.5, 43, color=MUTED, lw=1.8, style="-|>")

    # =========================================================================
    # 6. LINEAR ARBITRATION FUSION (Far Right)
    # =========================================================================
    draw_box(74, 30, 23, 38, "#F5F3FF", FUSION_C, border_w=2.2)
    ax.text(85.5, 64, "Linear Arbitration Fusion", fontsize=11, fontweight="bold", color=FUSION_C, ha="center")
    ax.text(85.5, 60.5, "Token Probability Simplex", fontsize=8.5, color=MUTED, ha="center")

    draw_box(76, 44, 19, 13, "#FFFFFF", FUSION_C, border_w=1.4, radius=1)
    ax.text(85.5, 53.5, "Convex Combination", fontsize=8.5, fontweight="bold", color=INK, ha="center")
    ax.text(85.5, 48.5, r"p(yₜ) = $\sum_{i=1}^n$ αᵢ,ₜ · pᵢ(yₜ)", fontsize=10.5, fontweight="bold", color=FUSION_C, ha="center")

    ax.text(85.5, 38, "• Consistent in Training & Decoding\n• Mixture NLL: -log(Σ αᵢ pᵢ(y))\n• Greedy or Sampled Next Token",
            fontsize=7.8, color=MUTED, ha="center")

    draw_box(77, 32, 17, 4.5, "#DCFCE7", OUT_C, border_w=1.2, radius=0.8)
    ax.text(85.5, 34.2, "Next Token  yₜ ~ p(yₜ)", fontsize=8.5, fontweight="bold", color=OUT_C, ha="center", va="center")

    # Arrows into Fusion
    # Candidate distributions P_i from heads
    draw_arrow(67, 73.5, 75, 55, color=H0_C, lw=1.8)
    draw_arrow(67, 64.5, 75, 52, color=H1_C, lw=1.8)
    draw_arrow(67, 55.5, 75, 49, color=H2_C, lw=1.8)

    # Gating alpha from EvidenceGate
    draw_arrow(67, 18, 77, 44, color=GATE_C, lw=2.2, rad=0.1)

    # Arrow to Final Output (Bottom Right)
    draw_box(77, 9, 17, 14, "#ECFDF5", OUT_C, border_w=2.0)
    ax.text(85.5, 19.5, "Generated Answer Y", fontsize=9.5, fontweight="bold", color=OUT_C, ha="center")
    ax.text(85.5, 14.5, "Autoregressive Sequence\nLegal Opinion / Analysis", fontsize=8, color=MUTED, ha="center")
    draw_arrow(85.5, 30, 85.5, 23, color=OUT_C, lw=2.4)

    plt.tight_layout()
    png_path = os.path.join(out_dir, "00_architecture_diagram.png")
    pdf_path = os.path.join(out_dir, "00_architecture_diagram.pdf")
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {png_path} and {pdf_path}")

if __name__ == "__main__":
    create_architecture_diagram("figures")

