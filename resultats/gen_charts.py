import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NAVY = "#0B3D5C"
TEAL = "#3FA7A3"
CORAL = "#E85D4C"
GREY = "#5B6B76"
LIGHTGREY = "#E8EDF0"

plt.rcParams["axes.edgecolor"] = LIGHTGREY
plt.rcParams["axes.labelcolor"] = GREY
plt.rcParams["xtick.color"] = GREY
plt.rcParams["ytick.color"] = GREY

# ============================================================
# SFT - run final stable (float32 intégral)
# Source : logs Kaggle, Trainer HuggingFace, logging_steps=10
# ============================================================
steps_1 = [10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200,
           210,220,230,240,250,260,270,280,290,300,310,320,330,340,350,360,370,380,
           390,400,410,420,430]
loss_1 = [4.073417,4.036996,3.801363,3.726423,3.532856,3.442294,3.419453,3.419060,
          3.382062,3.390599,3.395438,3.166899,3.272858,3.220022,3.214875,3.166568,
          3.209978,3.136885,3.105227,3.026230,3.005172,3.061332,2.996235,3.056205,
          3.036353,3.013003,3.009826,2.976646,3.004190,3.021547,2.945940,2.993262,
          3.113196,2.898722,2.991497,2.905698,2.993559,2.917014,2.909764,2.995332,
          2.963688,2.943150,2.870471]

steps_2 = [330,340,350,360,370,380,390,400,410,420,430,440,450,460,470,480,490]
loss_2  = [3.161551,2.895494,2.985846,2.898259,2.983653,2.901941,2.895211,2.979223,
           2.946012,2.925157,2.851829,2.963108,2.887317,2.931733,2.883869,2.973122,
           2.916759]

fig, ax = plt.subplots(figsize=(9.5, 5.2), dpi=200)
ax.plot(steps_1, loss_1, color=NAVY, linewidth=2, marker="o", markersize=3.5,
        label="Run initial (steps 10-430)")
ax.plot(steps_2, loss_2, color=TEAL, linewidth=2, marker="o", markersize=3.5,
        label="Reprise depuis checkpoint (steps 330-490)")
ax.axhline(y=2.87, color=CORAL, linestyle="--", linewidth=1.2, alpha=0.7)
ax.text(495, 2.87, " plateau ~ 2.87-2.97", color=CORAL, fontsize=9, va="center")
ax.set_xlabel("Step d'entrainement", fontsize=11)
ax.set_ylabel("Training loss", fontsize=11)
ax.set_title("SFT - Courbe de loss (run final stable, float32 integral)", fontsize=13, color=NAVY, fontweight="bold", pad=14)
ax.legend(loc="upper right", fontsize=9.5, frameon=False)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="y", color=LIGHTGREY, linewidth=0.8)
ax.set_xlim(0, 520)
fig.tight_layout()
fig.savefig("resultats/sft_loss_curve.png", facecolor="white")
plt.close(fig)
print("SFT chart OK")

# ============================================================
# DPO - run final (float32 integral)
# ============================================================
steps_dpo = [50, 84]
train_loss_dpo = [0.609073, 0.542196]
val_loss_dpo = [0.527861, 0.522109]

fig, ax = plt.subplots(figsize=(9.5, 5.2), dpi=200)
ax.plot(steps_dpo, train_loss_dpo, color=NAVY, linewidth=2.5, marker="o", markersize=8, label="Training loss")
ax.plot(steps_dpo, val_loss_dpo, color=TEAL, linewidth=2.5, marker="s", markersize=8, label="Validation loss")
for x, y in zip(steps_dpo, train_loss_dpo):
    ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=9, color=NAVY, fontweight="bold")
for x, y in zip(steps_dpo, val_loss_dpo):
    ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(0, -16), ha="center", fontsize=9, color=TEAL, fontweight="bold")
ax.set_xlabel("Step d'entrainement", fontsize=11)
ax.set_ylabel("Loss", fontsize=11)
ax.set_title("DPO - Loss finale (train vs validation)", fontsize=13, color=NAVY, fontweight="bold", pad=14)
ax.legend(loc="upper right", fontsize=9.5, frameon=False)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="y", color=LIGHTGREY, linewidth=0.8)
ax.set_xlim(30, 104)
ax.set_ylim(0.45, 0.68)
fig.tight_layout()
fig.savefig("resultats/dpo_loss_curve.png", facecolor="white")
plt.close(fig)
print("DPO chart OK")

# ============================================================
# Latence endpoint /triage (5 requetes)
# ============================================================
cas = ["BPCO", "Benin", "Trauma", "Pediatrie", "Court"]
latences = [2.47, 2.41, 2.42, 2.34, 2.40]

fig, ax = plt.subplots(figsize=(9.5, 5.2), dpi=200)
bars = ax.bar(cas, latences, color=TEAL, width=0.55)
for b, v in zip(bars, latences):
    ax.annotate(f"{v:.2f}s", (b.get_x() + b.get_width()/2, v), textcoords="offset points",
                xytext=(0, 6), ha="center", fontsize=10, color=NAVY, fontweight="bold")
ax.set_ylim(2.0, 2.65)
ax.set_ylabel("Latence (secondes)", fontsize=11)
ax.set_title("Latence endpoint /triage - 5 requetes (GPU Tesla T4, Kaggle)", fontsize=13, color=NAVY, fontweight="bold", pad=14)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="y", color=LIGHTGREY, linewidth=0.8)
fig.tight_layout()
fig.savefig("resultats/latence_endpoint.png", facecolor="white")
plt.close(fig)
print("Latency chart OK")