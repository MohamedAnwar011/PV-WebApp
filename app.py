# --- PATHS: always resolve relative to this file so cloud/local are consistent ---
import os, json, joblib, glob, math
from pathlib import Path
import pandas as pd
import streamlit as st

BASE_DIR  = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models_cache"          # <— absolute path now

st.set_page_config(page_title="PV Annual Results Predictor", layout="wide")
st.title("🔆 PV Annual Results (kWh) — Model Comparison")

# ---------- Utilities ----------
def find_latest_bundle(model_dir: Path):
    """Return absolute path to most recent bundle_*.json (by meta.trained_at)."""
    if not model_dir.is_dir():
        return None, None
    bundles = sorted(model_dir.glob("bundle_*.json"))
    if not bundles:
        return None, None

    best = None
    best_meta = None
    for bp in bundles:
        try:
            with open(bp, "r", encoding="utf-8") as f:
                manifest = json.load(f)
            meta = manifest.get("meta", {})
            if best_meta is None or int(meta.get("trained_at", 0)) > int(best_meta.get("trained_at", 0)):
                best = bp
                best_meta = meta
        except Exception:
            continue
    return (best, best_meta)

def load_bundle(bundle_path: Path):
    """Load manifest, then load models with robust, cloud-safe path handling."""
    with open(bundle_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    feature_names = manifest["feature_names"]
    meta = manifest.get("meta", {})
    fingerprint = meta.get("fingerprint", "")

    # Show what's physically present in models_cache (debug)
    present_files = sorted(p.name for p in MODEL_DIR.glob("*"))
    st.caption("Files found in models_cache:")
    st.write(present_files)

    trained = {}
    for name, info in manifest["models"].items():
        stored = info.get("est_path", "")          # may be absolute from local training
        # sanitize: always load from our MODEL_DIR using only the basename
        candidate = MODEL_DIR / Path(stored).name

        # fallback 1: expected pattern from our trainer
        expected = MODEL_DIR / f"{fingerprint}_{name.replace(' ', '_')}.joblib"
        if not candidate.exists() and expected.exists():
            candidate = expected

        # fallback 2: any joblib that contains the model name
        if not candidate.exists():
            matches = list(MODEL_DIR.glob(f"*{name.replace(' ', '_')}*.joblib"))
            if matches:
                candidate = matches[0]

        # final check
        if not candidate.exists():
            raise FileNotFoundError(
                f"Model file for '{name}' not found.\n"
                f"- est_path in manifest: {stored}\n"
                f"- Tried: {candidate}\n"
                f"- Also looked for: {expected.name}\n"
                f"- Present files: {present_files}"
            )

        est = joblib.load(candidate)
        trained[name] = {"estimator": est, "metrics": info.get("metrics", {})}

    # rebuild leaderboard safely
    leaderboard = pd.DataFrame(manifest["leaderboard"])
    leaderboard.index.name = None
    return trained, leaderboard, feature_names, meta


def angle_from_neighbor(our_h: float, neigh_h: float, dist: float) -> float:
    if dist is None or dist <= 0:
        return 0.0
    dh = (neigh_h or 0.0) - (our_h or 0.0)
    if dh <= 0:
        return 0.0
    return math.degrees(math.atan(dh / dist))

def compute_derived(length, width, height,
                    south_h, south_d, east_h, east_d, north_h, north_d, west_h, west_d):
    roof_area = (length or 0.0) * (width or 0.0) - 32.0
    pv_area   = 0.564177304724450 * roof_area - 40.5805869459240
    return {
        "Roof Area": roof_area,
        "PV Area": pv_area,
        "South Angle": angle_from_neighbor(height, south_h, south_d),
        "East":        angle_from_neighbor(height, east_h,  east_d),
        "North":       angle_from_neighbor(height, north_h, north_d),
        "West":        angle_from_neighbor(height, west_h,  west_d),
    }

def predict_for_models(trained_models: dict, X_row: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, obj in trained_models.items():
        pred = float(obj["estimator"].predict(X_row)[0])
        rows.append({"Model": name, "Predicted PV Annual results (kWh)": pred})
    return pd.DataFrame(rows).sort_values("Predicted PV Annual results (kWh)", ascending=False)

def get_feature_importances(estimator, feature_names):
    if hasattr(estimator, "feature_importances_"):
        return pd.Series(estimator.feature_importances_, index=feature_names).sort_values(ascending=False)
    return None

# ---------- Load latest cached bundle ----------
bundle_path, bundle_meta = find_latest_bundle(MODEL_DIR)
if not bundle_path:
    st.error("No cached models found in `models_cache/`. Commit your bundle_*.json and .joblib files.")
    st.stop()

trained_models, leaderboard, feature_names, meta = load_bundle(bundle_path)
st.success(f"Loaded cached models: `{bundle_path.name}` (fingerprint: {meta.get('fingerprint')})")


# ---------- Inputs ----------
st.markdown("---")
st.header("🧮 Enter Building Inputs")

colA, colB, colC, colD = st.columns(4)
with colA:
    height = st.number_input("Height (m)", min_value=0.0, value=20.0, step=0.5)
    length = st.number_input("Length (m)", min_value=0.0, value=30.0, step=0.5)
with colB:
    width  = st.number_input("Width (m)", min_value=0.0, value=20.0, step=0.5)
    floors = st.number_input("No. of floors", min_value=1, value=5, step=1)

st.markdown("#### Surroundings (Heights & Distances)")
c1, c2, c3, c4 = st.columns(4)
with c1:
    south_h = st.number_input("South Height (m)", min_value=0.0, value=15.0, step=0.5)
    south_d = st.number_input("South Distance (m)", min_value=0.0, value=10.0, step=0.5)
with c2:
    east_h  = st.number_input("East Height (m)", min_value=0.0, value=15.0, step=0.5)
    east_d  = st.number_input("East Distance (m)", min_value=0.0, value=10.0, step=0.5)
with c3:
    north_h = st.number_input("North Height (m)", min_value=0.0, value=15.0, step=0.5)
    north_d = st.number_input("North Distance (m)", min_value=0.0, value=10.0, step=0.5)
with c4:
    west_h  = st.number_input("West Height (m)", min_value=0.0, value=15.0, step=0.5)
    west_d  = st.number_input("West Distance (m)", min_value=0.0, value=10.0, step=0.5)

derived = compute_derived(length, width, height,
                          south_h, south_d, east_h, east_d, north_h, north_d, west_h, west_d)

with st.expander("📐 Derived Features (per your formulas)", expanded=True):
    st.dataframe(pd.DataFrame([{
        "Roof Area": derived["Roof Area"],
        "PV Area": derived["PV Area"],
        "South Angle (deg)": derived["South Angle"],
        "East (deg)": derived["East"],
        "North (deg)": derived["North"],
        "West (deg)": derived["West"],
    }]).style.format("{:,.3f}"), use_container_width=True)
if derived["Roof Area"] < 0:
    st.warning("Roof Area is negative (L×W − 32). Consider adjusting inputs.")

# ---------- Choose models ----------
st.markdown("### 📦 Choose Models to Compare")
default_models = list(trained_models.keys())
choices = st.multiselect("Select one or more models", options=list(trained_models.keys()), default=default_models)
if not choices:
    st.warning("Select at least one model.")
    st.stop()
selected = {k: v for k, v in trained_models.items() if k in choices}

# ---------- Predict ----------
X_infer = pd.DataFrame([{
    "Height": height,
    "Length": length,
    "Width": width,
    "No. of floors": floors,
    "South Angle": derived["South Angle"],
    "East": derived["East"],
    "North": derived["North"],
    "West": derived["West"],
    "Roof Area": derived["Roof Area"],
    "PV Area": derived["PV Area"],
}])[feature_names]

st.markdown("---")
st.header("📈 Predictions")
pred_df = predict_for_models(selected, X_infer)
st.dataframe(pred_df.style.format({"Predicted PV Annual results (kWh)": "{:,.2f}"}), use_container_width=True)
st.bar_chart(pred_df.set_index("Model"))

# ---------- Feature importances ----------
st.markdown("---")
st.header("🧭 Feature Importances")
tabs = st.tabs(list(selected.keys()))
for tab, (name, obj) in zip(tabs, selected.items()):
    with tab:
        imp = get_feature_importances(obj["estimator"], feature_names)
        if imp is None:
            st.info(f"{name}: Feature importances not available.")
        else:
            st.write(imp.to_frame("importance"))
            st.bar_chart(imp)

st.caption("Angles: atan((neighbor − our)/distance) in degrees; if neighbor ≤ our height or distance ≤ 0 → 0. "
           "Roof Area = L×W − 32. PV Area = 0.564177304724450 × Roof Area − 40.5805869459240. "
           "This app loads cached models from models_cache and does not retrain on startup.")
