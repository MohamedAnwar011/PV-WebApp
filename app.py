import os, json, joblib, glob, math
from pathlib import Path
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

BASE_DIR  = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models_cache"   

st.set_page_config(page_title="PV Annual Results Predictor", layout="wide", page_icon="🔆")

# --- (Keep all your helper functions here: find_latest_bundle, load_bundle, compute_derived, predict_for_models, get_feature_importances, make_cube_trace) ---

# [Helper functions omitted for brevity, keep them exactly as they were]

# ---------- Load Models ----------
bundle_path, bundle_meta = find_latest_bundle(MODEL_DIR)
if not bundle_path:
    st.error("No cached models found in models_cache/. Please add the files.")
    st.stop()

trained_models, leaderboard, feature_names, meta = load_bundle(bundle_path)

# Filter out ANN and SVM models
clean_models = {k: v for k, v in trained_models.items() if "ANN" not in k.upper() and "SVM" not in k.upper()}
trained_models = clean_models

# ==========================================
# SIDEBAR: Inputs
# ==========================================
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Solar_panel_icon.svg/512px-Solar_panel_icon.svg.png", width=50) # Optional logo
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
st.markdown("Estimate the annual photovoltaic yield (kWh) based on building geometry and urban context.")

# Calculate features
derived = compute_derived(length, width, height, south_h, south_d, east_h, east_d, north_h, north_d, west_h, west_d, No_of_floors)
if derived["Roof Area"] < 0:
    st.error("Warning: Roof Area is negative. Please check your length and width inputs.")

X_infer = pd.DataFrame([{
        "% roof Area/total floor area": derived["% roof Area/total floor area"],
        "No. of floors": No_of_floors,
        "Total Floor Area": derived["Total Floor Area"],
        "South Angle": derived["South Angle"],
        "East": derived["East"],
        "West": derived["West"],
        "North": derived["North"]
    }])[feature_names] 

pred_df = predict_for_models(selected, X_infer)

# Top metric display
best_model_name = pred_df.iloc[0]["Model"]
best_model_val = pred_df.iloc[0]["PV/EUI%"]

col1, col2, col3 = st.columns(3)
col1.metric("Highest Predicted Yield", f"{best_model_val:,.2f} kWh", f"Model: {best_model_name}")
col2.metric("Available Roof Area", f"{derived['Roof Area']:,.1f} m²")
col3.metric("Total Floor Area", f"{derived['Total Floor Area']:,.1f} m²")

st.markdown("---")

# Use Tabs to keep the UI clean
tab1, tab2, tab3, tab4 = st.tabs(["📊 Results", "🏙️ 3D Site Context", "🧭 Feature Importances", "📝 Methodology"])

with tab1:
    st.subheader("Model Predictions")
    c1, c2 = st.columns([2, 3])
    with c1:
        st.dataframe(pred_df.style.format({"PV/EUI%": "{:,.2f}"}), use_container_width=True)
        # Download button for researchers
        csv = pred_df.to_csv(index=False).encode('utf-8')
        st.download_button(label="📥 Download Results as CSV", data=csv, file_name='pv_predictions.csv', mime='text/csv')
    with c2:
        st.bar_chart(pred_df.set_index("Model"))

with tab2:
    st.subheader("Urban Context Visualization")
    fig = go.Figure()
    fig.add_trace(make_cube_trace(0, 0, 0, width, length, height, "blue", "Main Building"))
    fig.add_trace(make_cube_trace(0, -(length/2 + south_d + 5), 0, width, 10, south_h, "gray", "South Neighbor"))
    fig.add_trace(make_cube_trace(0, (length/2 + north_d + 5), 0, width, 10, north_h, "gray", "North Neighbor"))
    fig.add_trace(make_cube_trace((width/2 + east_d + 5), 0, 0, 10, length, east_h, "gray", "East Neighbor"))
    fig.add_trace(make_cube_trace(-(width/2 + west_d + 5), 0, 0, 10, length, west_h, "gray", "West Neighbor"))
    
    fig.update_layout(scene=dict(aspectmode='data'), margin=dict(l=0, r=0, b=0, t=0), height=600)
    st.plotly_chart(fig, use_container_width=True)

with tab3:
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

with tab4:
    st.subheader("How this works")
    st.write("This tool predicts the PV output based on building dimensions and the shading angles of surrounding structures. The angles are calculated using `atan((neighbor_height - building_height) / distance)`.")
    st.write("Roof Area is calculated as `L × W - 32` to account for standard HVAC and rooftop equipment spacing.")
    st.write("**Data Source:** Provide a brief note here about where your training data came from (e.g., EnergyPlus simulations, real-world smart meter data).")
    
    with st.expander("View Full Derived Input Variables"):
        st.json(derived)
