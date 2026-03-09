import os, json, joblib, glob, math
from pathlib import Path
import pandas as pd
import streamlit as st
import pydeck as pdk  # This replaces plotly

BASE_DIR  = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models_cache"   

st.set_page_config(page_title="PV Annual Results Predictor", layout="wide", page_icon="🔆")

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
                f"Model file for '{name}' not found. Please check models_cache folder."
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

def compute_derived(length, width, height, south_h, south_d, east_h, east_d, north_h, north_d, west_h, west_d, No_of_floors):
    roof_area = (length or 0.0) * (width or 0.0) - 32.0
    pv_area = 0.564177304724450 * roof_area - 40.5805869459240
    total_floor_area = ((length or 0.0) * (width or 0.0) - 32) * No_of_floors
    roof_area_total_floor_area = (roof_area / total_floor_area) if total_floor_area > 0 else 0.0 
    
    return {
        "Roof Area": roof_area,
        "PV Area": pv_area,
        "South Angle": angle_from_neighbor(height, south_h, south_d),
        "East":        angle_from_neighbor(height, east_h,  east_d),
        "North":       angle_from_neighbor(height, north_h, north_d),
        "West":        angle_from_neighbor(height, west_h,  west_d),
        "Total Floor Area": total_floor_area,
        "% roof Area/total floor area": roof_area_total_floor_area,
        "Length": length
    }

def predict_for_models(trained_models: dict, X_row: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, obj in trained_models.items():
        pred = float(obj["estimator"].predict(X_row)[0])
        rows.append({"Model": name, "PV Annual results (kWh)": pred})
    return pd.DataFrame(rows).sort_values("PV Annual results (kWh)", ascending=False)

def get_feature_importances(estimator, feature_names):
    if hasattr(estimator, "feature_importances_"):
        return pd.Series(estimator.feature_importances_, index=feature_names).sort_values(ascending=False)
    return None

def get_geo_polygon(cx, cy, w, l):
    # Base coordinates (Cairo, Egypt)
    base_lat = 30.061035160155033
    base_lon = 31.33993948956984
    
    
    # Convert meters to map degrees
    lat_per_m = 1.0 / 111111.0
    lon_per_m = 1.0 / (111111.0 * math.cos(math.radians(base_lat)))
    
    # Calculate the 4 corners of the building
    corners = [
        (cx - w/2, cy - l/2),
        (cx + w/2, cy - l/2),
        (cx + w/2, cy + l/2),
        (cx - w/2, cy + l/2),
        (cx - w/2, cy - l/2)
    ]
    
    return [[base_lon + (x * lon_per_m), base_lat + (y * lat_per_m)] for x, y in corners]

def calculate_lcoe(e0, capacity_kw, capex_per_kw, om_per_kw=5.6, r=0.10, g=0.006, n=25):
    if capacity_kw <= 0 or e0 <= 0:
        return 0.0
    
    initial_investment = capacity_kw * capex_per_kw
    total_cost = initial_investment
    total_energy = 0.0
    
    for t in range(1, n + 1):
        # O&M cost discounted for year t
        yearly_om = capacity_kw * om_per_kw
        total_cost += yearly_om / ((1 + r) ** t)
        
        # Energy produced in year t with degradation, discounted
        yearly_energy = e0 * ((1 - g) ** (t - 1))
        total_energy += yearly_energy / ((1 + r) ** t)
        
    return total_cost / total_energy if total_energy > 0 else 0.0

# ---------- Load Models ----------
bundle_path, bundle_meta = find_latest_bundle(MODEL_DIR)
if not bundle_path:
    st.error("No cached models found in models_cache/. Please add the files.")
    st.stop()

trained_models, leaderboard, feature_names, meta = load_bundle(bundle_path)

# ==========================================
# SIDEBAR: Inputs
# ==========================================
st.sidebar.title("Building Parameters")

st.sidebar.markdown("### Main Building")
height = st.sidebar.number_input("Height (m)", value=20.0, step=0.5)
length = st.sidebar.number_input("Length (m)", value=30.0, step=0.5)
width  = st.sidebar.number_input("Width (m)", value=20.0, step=0.5)
No_of_floors = st.sidebar.number_input("No. of floors", value=5, step=1)

st.sidebar.markdown("### Surroundings")
south_h = st.sidebar.number_input("South Height (m)", value=15.0, step=0.5)
south_d = st.sidebar.number_input("South Distance (m)", value=10.0, step=0.5)

east_h  = st.sidebar.number_input("East Height (m)", value=15.0, step=0.5)
east_d  = st.sidebar.number_input("East Distance (m)", value=10.0, step=0.5)

north_h = st.sidebar.number_input("North Height (m)", value=15.0, step=0.5)
north_d = st.sidebar.number_input("North Distance (m)", value=10.0, step=0.5)

west_h  = st.sidebar.number_input("West Height (m)", value=15.0, step=0.5)
west_d  = st.sidebar.number_input("West Distance (m)", value=10.0, step=0.5)

st.sidebar.markdown("---")
st.sidebar.markdown("### Model Selection")
choices = st.sidebar.multiselect("Models to Compare", options=list(trained_models.keys()), default=list(trained_models.keys()))

if not choices:
    st.warning("Please select at least one model in the sidebar.")
    st.stop()

selected = {k: v for k, v in trained_models.items() if k in choices}

# ==========================================
# MAIN PAGE
# ==========================================
st.title("🔆 PV Annual Results Predictor")
st.markdown("Estimate the annual photovoltaic yield (kWh) and economic viability based on building geometry and urban context.")

# Calculate features
derived = compute_derived(length, width, height, south_h, south_d, east_h, east_d, north_h, north_d, west_h, west_d, No_of_floors)
if derived["Roof Area"] < 0:
    st.error("Warning: Roof Area is negative. Please check your length and width inputs.")

# Pass the exact features expected by the new notebook models
X_infer = pd.DataFrame([{
        "Length": derived["Length"],
        "South Angle": derived["South Angle"],
        "East": derived["East"],
        "North": derived["North"],
        "West": derived["West"],
        "Roof Area": derived["Roof Area"],
        "PV Area": derived["PV Area"]
    }])[feature_names] 

pred_df = predict_for_models(selected, X_infer)

# Top metric display
best_model_name = pred_df.iloc[0]["Model"]
best_model_val = pred_df.iloc[0]["PV Annual results (kWh)"]

col1, col2, col3 = st.columns(3)
col1.metric("Highest Predicted Yield", f"{best_model_val:,.2f} kWh", f"Model: {best_model_name}")
col2.metric("Available Roof Area", f"{derived['Roof Area']:,.1f} m²")
col3.metric("PV Area", f"{derived['PV Area']:,.1f} m²")

st.markdown("---")

# Estimate a default capacity using the new PV Area feature
estimated_pv_area = max(0, derived["PV Area"])
default_kw = round(estimated_pv_area * 0.2, 1) if estimated_pv_area > 0 else 10.0

# Sidebar input for System Capacity
st.sidebar.markdown("### Economic Parameters")
sys_capacity = st.sidebar.number_input("PV System Capacity (kW)", min_value=1.0, value=float(default_kw), step=1.0, help="Used for LCOE calculations.")

# Use Tabs to keep the UI clean
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Results", "💰 Economic Viability (LCOE)", "🏙️ 3D Site Context", "🧭 Feature Importances", "📝 Methodology"])

with tab1:
    st.subheader("Model Predictions")
    c1, c2 = st.columns([2, 3])
    with c1:
        st.dataframe(pred_df.style.format({"PV Annual results (kWh)": "{:,.2f}"}), use_container_width=True)
        csv = pred_df.to_csv(index=False).encode('utf-8')
        st.download_button(label="📥 Download Results as CSV", data=csv, file_name='pv_predictions.csv', mime='text/csv')
    with c2:
        st.bar_chart(pred_df.set_index("Model"))

with tab2:
    st.subheader("Levelized Cost of Electricity (LCOE)")
    st.markdown(f"Comparing the predicted energy output against the commercial grid tariff of **$0.0466/kWh (2.33 EGP)**. System assumed at **{sys_capacity} kW**.")
    
    lcoe_data = []
    for index, row in pred_df.iterrows():
        e0_val = row["PV Annual results (kWh)"]
        # Calculate optimistic scenario (600 USD/KW)
        lcoe_opt = calculate_lcoe(e0_val, sys_capacity, capex_per_kw=600)
        # Calculate pessimistic scenario (880 USD/KW)
        lcoe_pes = calculate_lcoe(e0_val, sys_capacity, capex_per_kw=880)
        
        lcoe_data.append({
            "Model": row["Model"],
            "Annual Output (kWh)": e0_val,
            "Optimistic LCOE ($/kWh)": lcoe_opt,
            "Pessimistic LCOE ($/kWh)": lcoe_pes,
            "Optimistic Viable?": "✅ Yes" if lcoe_opt < 0.1466 else "❌ No",
            "Pessimistic Viable?": "✅ Yes" if lcoe_pes < 0.1466 else "❌ No"
        })
        
    lcoe_df = pd.DataFrame(lcoe_data)
    st.dataframe(lcoe_df.style.format({
        "Annual Output (kWh)": "{:,.2f}",
        "Optimistic LCOE ($/kWh)": "${:.4f}",
        "Pessimistic LCOE ($/kWh)": "${:.4f}"
    }), use_container_width=True)
    
    st.info("Assumptions: 25-year lifespan, 10% discount rate, 0.6% annual degradation, and $5.60/kW yearly O&M cost.")

with tab3:
    st.subheader("Urban Context Visualization")
    
    # Create the data for the main building and neighbors
    buildings_data = [
        {"name": "Main Building", "height": height, "color": [210, 210, 210, 255], 
         "polygon": get_geo_polygon(0, 0, width, length)},
        
        {"name": "South Neighbor", "height": south_h, "color": [140, 150, 160, 255], 
         "polygon": get_geo_polygon(0, -(length/2 + south_d + 5), width, 10)},
        
        {"name": "North Neighbor", "height": north_h, "color": [140, 150, 160, 255], 
         "polygon": get_geo_polygon(0, (length/2 + north_d + 5), width, 10)},
        
        {"name": "East Neighbor", "height": east_h, "color": [140, 150, 160, 255], 
         "polygon": get_geo_polygon((width/2 + east_d + 5), 0, 10, length)},
        
        {"name": "West Neighbor", "height": west_h, "color": [140, 150, 160, 255], 
         "polygon": get_geo_polygon(-(width/2 + west_d + 5), 0, 10, length)}
    ]
    
    # Draw rows of solar panels instead of one big block
    num_rows = 5
    row_width = width * 0.8
    row_length = (length * 0.8) / (num_rows * 2) 
    
    start_y = - (length * 0.4) + (row_length / 2)
    step_y = (length * 0.8) / num_rows
    
    for i in range(num_rows):
        cy = start_y + (i * step_y)
        buildings_data.append({
            "name": f"Solar Row {i+1}", 
            "height": height + 0.5, 
            "color": [10, 50, 120, 255], 
            "polygon": get_geo_polygon(0, cy, row_width, row_length)
        })
    
    df_map = pd.DataFrame(buildings_data)
    
    # Setup the 3D map layer
    layer = pdk.Layer(
        "PolygonLayer",
        data=df_map,
        get_polygon="polygon",
        get_elevation="height",
        get_fill_color="color",
        extruded=True,
        wireframe=True,
        pickable=True
    )
    
    # Set the starting camera angle
    view_state = pdk.ViewState(
        latitude=30.061035160155033,
        longitude=31.33993948956984,
        zoom=18.5,
        pitch=60,
        bearing=45
    )
    
    # Draw the map
    r = pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        map_style="light",
        tooltip={"text": "{name}\nHeight: {height}m"}
    )
    
    st.pydeck_chart(r, use_container_width=True)
with tab4:
    st.subheader("What drives these predictions?")
    st.markdown("The charts below show which features the models rely on the most.")
    feat_tabs = st.tabs(list(selected.keys()))
    for f_tab, (name, obj) in zip(feat_tabs, selected.items()):
        with f_tab:
            imp = get_feature_importances(obj["estimator"], feature_names)
            if imp is None:
                st.info(f"Feature importances not available for {name}.")
            else:
                st.bar_chart(imp)

with tab5:
    st.subheader("Methodology")
    st.write("This tool predicts the PV output based on building dimensions and the shading angles of surrounding structures.")
    st.write("The Levelized Cost of Electricity (LCOE) evaluates the economic potential of the PV system over its 25-year lifecycle. The cost includes the initial setup ($600 or $880 per kW) and annual O&M ($5.60 per kW), discounted at 10% per year. Energy output drops by 0.6% annually due to hardware degradation.")
    
    with st.expander("View Full Derived Input Variables"):
        st.json(derived)
