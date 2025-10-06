import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

# === 0) Command-line argument ===
if len(sys.argv) < 2:
    print("Usage: python script.py <path_to_csv>")
    sys.exit(1)

csv_path = sys.argv[1]
print(f"Loading data from: {csv_path}")

# === 1) Load file robustly ===
df = pd.read_csv(
    csv_path,
    sep=r"[\s,]+",              # Match one or more spaces, tabs, or commas
    engine="python",
    header=None,
    names=["img1", "img2", "score"],
    usecols=[0, 1, 2],
    dtype={0: str, 1: str, 2: float},
    comment="#",
    skip_blank_lines=True,
)

# === 2) Clean and extract IDs ===
df["img1"] = df["img1"].astype(str).str.strip()
df["img2"] = df["img2"].astype(str).str.strip()
df["id1"] = df["img1"].str.partition("_")[0]
df["id2"] = df["img2"].str.partition("_")[0]

# === 3) Genuine / impostor labels ===
df["label"] = (df["id1"] == df["id2"]).astype(int)

# === 4) ROC curve (remember: lower distance = more genuine) ===
y_true = df["label"].to_numpy().astype(int)
y_score_similarity = -df["score"].to_numpy()  # flip so larger = more genuine

fpr, tpr, thresholds = roc_curve(y_true, y_score_similarity)
auc = roc_auc_score(y_true, y_score_similarity)

# === 5) EER ===
fnr = 1 - tpr
idx = np.nanargmin(np.abs(fnr - fpr))
eer = (fpr[idx] + fnr[idx]) / 2.0
eer_threshold_on_similarity = thresholds[idx]
eer_threshold_on_distance = -eer_threshold_on_similarity

print(f"AUC: {auc:.4f}")
print(f"EER: {eer:.4f}")
print(f"Threshold @ EER (distance space): {eer_threshold_on_distance:.6f}")

# === 6) Plot distributions ===
plt.figure()
plt.hist(df.loc[df["label"] == 1, "score"], bins=50, alpha=0.6, density=True, label="Genuine")
plt.hist(df.loc[df["label"] == 0, "score"], bins=50, alpha=0.6, density=True, label="Impostor")
plt.xlabel("Distance")
plt.ylabel("Density")
plt.title("Genuine vs Impostor Distributions")
plt.legend()
plt.tight_layout()
plt.show()

# === 7) ROC curve ===
plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.tight_layout()
plt.show()

# === 8) DET curve (FPR vs FNR) ===
plt.figure()
plt.plot(fpr, fnr, label="DET Curve")
plt.xlabel("False Positive Rate (FPR)")
plt.ylabel("False Negative Rate (FNR)")
plt.title("DET Curve")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
