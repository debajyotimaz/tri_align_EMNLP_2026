# ============================================================
# SAFE Entropy Analysis for CM Representations (ACL-READY)
# - PCA-based Gaussian entropy (numerically stable)
# - Float16 safe
# - Token-level safe
# - Adaptive PCA rank (capped)
# ============================================================

import os
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

# ============================================================
# PATHS
# ============================================================

REP_ROOT = "/reo_root/rep"
OUT_ROOT = "/output_root/entropy_results_pca_safe"
os.makedirs(OUT_ROOT, exist_ok=True)

MODEL_CONFIG = {
    "mbert": "encoders",
    "hing_mbert": "encoders",
    "hing_mbert_mixed": "encoders",
    "xlm_roberta_base": "encoders",
    "hing_roberta": "encoders",
    "hing_roberta_mixed": "encoders",
}

PCA_K = 64          # ACL-safe upper bound
RIDGE_ALPHA = 1.0
EPS = 1e-6

# ============================================================
# HELPERS
# ============================================================

def safe_load(path):
    if not os.path.exists(path):
        return None
    try:
        arr = np.load(path)

        if arr.size == 0:
            return None

        # 🔴 REQUIRED: float16 → float32
        if arr.dtype == np.float16:
            arr = arr.astype(np.float32)

        if not np.isfinite(arr).all():
            return None

        return arr
    except Exception:
        return None


def ensure_2d(X):
    """
    Converts token-level [N, T, D] → [N, D] via mean pooling.
    """
    if X.ndim == 3:
        return X.mean(axis=1)
    return X


def get_layers(rep_dir):
    return sorted(
        int(f.split("_")[-1].split(".")[0])
        for f in os.listdir(rep_dir)
        if f.startswith("layer_") and f.endswith(".npy")
    )


def pca_project(X, k=PCA_K):
    X = ensure_2d(X)
    X = X.astype(np.float32)

    X = X - X.mean(axis=0, keepdims=True)

    U, S, _ = np.linalg.svd(X, full_matrices=False)

    k_eff = min(k, S.shape[0])
    if k_eff < 2:
        raise ValueError("Effective PCA rank too small")

    return U[:, :k_eff] @ np.diag(S[:k_eff])


def entropy_gaussian(X):
    Sigma = np.cov(X, rowvar=False) + EPS * np.eye(X.shape[1])
    sign, logdet = np.linalg.slogdet(Sigma)

    if sign <= 0 or not np.isfinite(logdet):
        return np.nan

    d = X.shape[1]
    return 0.5 * (logdet + d * np.log(2 * np.pi * np.e))


def conditional_entropy_gaussian(X, Y):
    if X.shape[0] != Y.shape[0]:
        return np.nan

    reg = Ridge(alpha=RIDGE_ALPHA)
    reg.fit(Y, X)
    R = X - reg.predict(Y)

    Sigma = np.cov(R, rowvar=False) + EPS * np.eye(R.shape[1])
    sign, logdet = np.linalg.slogdet(Sigma)

    if sign <= 0 or not np.isfinite(logdet):
        return np.nan

    d = R.shape[1]
    return 0.5 * (logdet + d * np.log(2 * np.pi * np.e))

# ============================================================
# MAIN LOOP
# ============================================================

for model in MODEL_CONFIG:
    print(f"\n================ {model} =================")

    out_dir = os.path.join(OUT_ROOT, model)
    os.makedirs(out_dir, exist_ok=True)

    cm_dir = f"{REP_ROOT}/{model}/cm"
    if not os.path.exists(cm_dir):
        print("⚠️ CM representations missing — skipping model")
        continue

    layers = get_layers(cm_dir)
    print(f"▶ Found {len(layers)} candidate layers")

    rows = []

    for layer in layers:
        out_json = os.path.join(out_dir, f"H_CM_layer_{layer:02d}.json")

        if os.path.exists(out_json):
            print(f"_ Layer {layer} already exists — skipping")
            continue

        print(f"_ Computing Layer {layer}")

        R_CM = safe_load(f"{REP_ROOT}/{model}/cm/layer_{layer:02d}.npy")
        R_HI = safe_load(f"{REP_ROOT}/{model}/hi/layer_{layer:02d}.npy")
        R_EN = safe_load(f"{REP_ROOT}/{model}/en/layer_{layer:02d}.npy")

        if R_CM is None or R_HI is None or R_EN is None:
            print("   ⚠️ Missing / corrupted representations — skipping")
            continue

        if not (R_CM.shape[0] == R_HI.shape[0] == R_EN.shape[0]):
            print("   ⚠️ Sample count mismatch — skipping")
            continue

        # ---------- PCA ----------
        try:
            R_CM_p = pca_project(R_CM)
            R_HI_p = pca_project(R_HI)
            R_EN_p = pca_project(R_EN)
        except Exception as e:
            print(f"   ⚠️ PCA failed — skipping layer ({e})")
            continue

        # ---------- Entropies ----------
        h_cm = entropy_gaussian(R_CM_p)
        h_cm_hi = conditional_entropy_gaussian(R_CM_p, R_HI_p)
        h_cm_en = conditional_entropy_gaussian(R_CM_p, R_EN_p)
        h_cm_hi_en = conditional_entropy_gaussian(
            R_CM_p, np.hstack([R_HI_p, R_EN_p])
        )

        row = {
            "layer": layer,
            "H(CM)": h_cm,
            "H(CM | Hindi)": h_cm_hi,
            "H(CM | English)": h_cm_en,
            "H(CM | Hindi, English)": h_cm_hi_en,
            "pca_dim": R_CM_p.shape[1],
        }

        rows.append(row)

        with open(out_json, "w") as f:
            json.dump(row, f, indent=2)

    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(out_dir, "H_CM_ALL_entropies.csv"), index=False)

    print(f"✔ Saved entropy results for {model}")

print("\n✅ SAFE PCA-based entropy analysis completed.")
