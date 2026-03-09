import os, json, joblib, glob, math
from pathlib import Path
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

BASE_DIR  = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models_cache"   

st.set_page_config(page_title="PV Annual Results Predictor", layout="wide")
st.title("🔆 PV Annual Results (kWh) — Model Comparison")

# ---------- Utilities ----------
def find_latest_bundle(model_dir: Path):
    if not model_dir.is_dir():
        return None, None
    bundles = sorted(model_dir.glob("bundle_*.json"))
    if not bundles:
        return None, None
    best, best_meta = None, None
    for bp in bundles:
        try:
            with open(bp, "r", encoding="utf-8") as f:
                manifest = json.load(f)
            meta = manifest.get("meta", {})
            if best_meta is None or int(meta.get("trained_at", 0)) > int(best_meta.get("trained_at", 0)):
                best, best_meta = bp, meta
        except Exception:
            continue
    return best, best_meta

def _sanitize_name(s: str) -> str:
    return s.replace(" ", "_").replace("(", "").replace(")", "").lower()

def load_bundle(bundle_path: Path):
    with open(bundle_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    feature_names = manifest["feature_names"]
    meta = manifest.get("meta", {})
    fingerprint = meta.get("fingerprint", "")

    present_files = sorted(p.name for p in MODEL_DIR.glob("*.joblib"))

    present_map = {p: (MODEL_DIR / p) for p in present_files}
    used_files = set()

    trained = {}
    for name, info in manifest["models"].items():
        stored = info.get("est_path", "")
        base = os.path.basename(stored)
        cand = present_map.get(base) if base in present_map else None

        if cand is None:
            expected = f"{fingerprint}_{name.replace(' ', '_')}.joblib"
            if expected in present_map:
                cand = present_map[expected]

        if cand is None:
            target_key = _sanitize_name(name)
            matches = [present_map[p] for p in present_files
                       if _sanitize_name(p).find(target_key) != -1 and p not in used_files]
            if matches:
                cand = matches[0]

        if cand is None:
            leftovers = [present_map[p] for p in present_files if p not in used_files]
            if len(leftovers) == (len(manifest["models"]) - len(trained)) == 1:
                cand = leftovers[0]

        if cand is None or not cand.exists():
            raise FileNotFoundError(
                f"Model file for '{name}' not found.\n"
                f"- est_path in manifest: {stored}\n"
                f"- Tried basename: {base}\n"
                f"- Also expected: {fingerprint}_{name.replace(' ', '_')}.joblib\n"
                f"- Present files: {present_files}"
            )

        est = joblib.load(cand)
        used_files.add(cand.name)
        trained[name] = {"estimator": est, "metrics": info.get("metrics", {})}

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
                    south_h, south_d, east_h, east_d, north_h, north_d, west_h, west_d, No_of_floors):
    roof_area = (length or 0.0) * (width or 0.0) - 32.0
    total_floor_area = ((length or 0.0) * (width or 0.0) - 32) * No_of_floors
    roof_area_total_floor_area = (roof_area / total_floor_area) if total_floor_area > 0 else 0.0 
    
    return {
        "Roof Area": roof_area,
        "South Angle": angle_from_neighbor(height, south_h, south_d),
        "East":        angle_from_neighbor(height, east_h,  east_d),
        "North":       angle_from_neighbor(height, north_h, north_d),
        "West":        angle_from_neighbor(height, west_h,  west_d),
        "Total Floor Area": total_floor_area,
        "% roof Area/total floor area": roof_area_total_floor_area
    }

def predict_for_models(trained_models: dict, X_row: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, obj in trained_models.items():
        pred = float(obj["estimator"].predict(X_row)[0])
        rows.append({"Model": name, "PV/EUI%": pred})
    return pd.DataFrame(rows).sort_values("PV/EUI%", ascending=False)

def get_feature_importances(estimator, feature_names):
    if hasattr(estimator, "feature_importances_"):
        return pd.Series(estimator.feature_importances_, index=feature_names).sort_values(ascending=False)
    return None

def make_cube_trace(x_center, y_center, z_base, dx, dy, dz, color, name):
    x = [x_center - dx/2, x_center + dx/2, x_center + dx/2, x_center - dx/2,
         x_center - dx/2, x_center + dx/2, x_center + dx/2, x_center - dx/2]
    y = [y_center - dy/2, y_center - dy/2, y_center + dy/2, y_center + dy/2,
         y_center - dy/2, y_center - dy/2, y_center + dy/2, y_center + dy/2]
    z = [z_base, z_base, z_base, z_base,
         z_base + dz, z_base + dz, z_base + dz, z_base + dz]

    i = [7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2]
    j = [3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3]
    k = [0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6]

    return go.Mesh3d(
        x=x, y=y, z=z,
        i=i, j=j, k=k,
        opacity=0.8,
        color=color,
        name=name,
        flatshading=True,
        hoverinfo='name+text',
        text=f"Height: {dz}m<br>Width: {dx}m<br>Length: {dy}m"
    )

# ---------- Load latest cached bundle ----------
bundle_path, bundle_meta = find_latest_bundle(MODEL_DIR)
if not bundle_path:
    st.error("No cached models found in models_cache/. Commit bundle_*.json and the .joblib files.")
    st.stop()

trained_models, leaderboard, feature_names, meta = load_bundle(bundle_path)

# Filter out ANN and SVM models
clean_models = {}
for name, data in trained_models.items():
    name_upper = name.upper()
    if "ANN" not in name_upper and "SVM" not in name_upper:
        clean_models[name] = data
trained_models = clean_models

# ---------- Inputs ----------
st.markdown("---")
st.header("🧮 Enter Building Inputs")

colA, colB, colC, colD = st.columns(4)
with colA:
    height = st.number_input("Height (m)", min_value=0.0, value=20.0, step=0.5)
    length = st.number_input("Length (m)", min_value=0.0, value=30.0, step=0.5)
with colB:
    width  = st.number_input("Width (m)", min_value=0.0, value=20.0, step=0.5)
    No_of_floors = st.number_input("No. of floors", min_value=1, value=5, step=1)

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

# ---------- 3D Plotly Visualization ----------
st.markdown("#### 🏙️ 3D Site Context")
fig = go.Figure()

# Main building in the center
fig.add_trace(make_cube_trace(0, 0, 0, width, length, height, "blue", "Main Building"))

# South neighbor (negative y direction)
fig.add_trace(make_cube_trace(0, -(length/2 + south_d + 5), 0, width, 10, south_h, "gray", "South Neighbor"))

# North neighbor (positive y direction)
fig.add_trace(make_cube_trace(0, (length/2 + north_d + 5), 0, width, 10, north_h, "gray", "North Neighbor"))

# East neighbor (positive x direction)
fig.add_trace(make_cube_trace((width/2 + east_d + 5), 0, 0, 10, length, east_h, "gray", "East Neighbor"))

# West neighbor (negative x direction)
fig.add_trace(make_cube_trace(-(width/2 + west_d + 5), 0, 0, 10, length, west_h, "gray", "West Neighbor"))

fig.update_layout(
    scene=dict(aspectmode='data'),
    margin=dict(l=0, r=0, b=0, t=0),
    height=500
)
st.plotly_chart(fig, use_container_width=True)

# ---------- Derived Features & Predictions ----------
derived = compute_derived(length, width, height,
                    south_h, south_d, east_h, east_d, north_h, north_d, west_h, west_d, No_of_floors)

with st.expander("📐 Derived Features (per your formulas)", expanded=True):
    st.dataframe(pd.DataFrame([{
        "Roof Area": derived["Roof Area"],
        "South Angle (deg)": derived["South Angle"],
        "East (deg)": derived["East"],
        "North (deg)": derived["North"],
        "West (deg)": derived["West"],
        "Total Floor Area": derived["Total Floor Area"],
        "% roof Area/total floor area": derived["% roof Area/total floor area"]
    }]).style.format("{:,.3f}"), use_container_width=True)
    
if derived["Roof Area"] < 0:
    st.warning("Roof Area is negative (L×W − 32). Consider adjusting inputs.")

st.markdown("### 📦 Choose Models to Compare")
default_models = list(trained_models.keys())
choices = st.multiselect("Select one or more models", options=list(trained_models.keys()), default=default_models)
if not choices:
    st.warning("Select at least one model.")
    st.stop()
selected = {k: v for k, v in trained_models.items() if k in choices}

X_infer = pd.DataFrame([{
        "% roof Area/total floor area": derived["% roof Area/total floor area"],
        "No. of floors": No_of_floors,
        "Total Floor Area": derived["Total Floor Area"],
        "South Angle": derived["South Angle"],
        "East": derived["East"],
        "West": derived["West"],
        "North": derived["North"]
    }])[feature_names] 

st.markdown("---")
st.header("📈 Predictions")
pred_df = predict_for_models(selected, X_infer)
st.dataframe(pred_df.style.format({"PV/EUI%": "{:,.2f}"}), use_container_width=True)
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
