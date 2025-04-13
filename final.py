import streamlit as st
import numpy as np
import joblib
import xgboost as xgb

# --- Load Pre-trained Objects ---
scaler = joblib.load("scaler.pkl")
pca = joblib.load("pca.pkl")
model = joblib.load("xgboost_model.pkl")  # XGBoost Booster

# --- Status Descriptions ---
status_map = {
    0: "🟢 Normal Operation",
    1: "🟡 Light Stress (Heat Dissipation Issue)",
    2: "🟠 Medium Stress (Power Fluctuation)",
    3: "🔴 High Stress (Overstrain)",
    4: "⚙️ Tool Wear",
    5: "🔥 Critical Condition (Major Failure)"
}

# --- Remedies Dictionary ---
remedy = {
    0: [
        "- No issues detected.",
        "- Continue regular maintenance schedules.",
        "- Monitor for anomalies periodically."
    ],
    1: [
        "- Check and clean ventilation systems and cooling fans.",
        "- Inspect temperature sensors for drift.",
        "- Ensure proper airflow around the machine."
    ],
    2: [
        "- Check power supply voltage and grounding.",
        "- Inspect motor load and inverter settings.",
        "- Look for inconsistent current draw or transient spikes."
    ],
    3: [
        "- Reduce machine load or runtime if possible.",
        "- Inspect mechanical connections and shafts.",
        "- Lubricate moving parts and verify torque settings."
    ],
    4: [
        "- Inspect tool edges for dullness or breakage.",
        "- Replace or sharpen tools as needed.",
        "- Recalibrate tool alignment after replacement.",
        "- Log tool usage data to track wear trends."
    ],
    5: [
        "- Immediately stop machine operation.",
        "- Run full diagnostics on all subsystems.",
        "- Check for major electrical or mechanical faults.",
        "- Escalate to technical support or maintenance team.",
        "- Review logs for root cause before restarting."
    ]
}

# --- Page Setup ---
st.set_page_config(page_title="AI Maintenance Assistant", layout="wide")

st.title("🧠 AI Predictive Maintenance Assistant")
st.markdown("Optimize your operations with real-time machine failure prediction and tailored maintenance guidance.")
st.divider()

# --- Class Meaning Section ---
with st.expander("📘 What Do the Failure Classes Mean?"):
    st.markdown("""
    | Class | Condition | Description |
    |-------|-----------|-------------|
    | **0** | 🟢 Normal Operation | No failure detected. Machine is healthy. |
    | **1** | 🟡 Light Stress | Heat dissipation issues. Check temperature systems. |
    | **2** | 🟠 Medium Stress | Power instability. Electrical inspection suggested. |
    | **3** | 🔴 High Stress | Mechanical overstrain. Reduce load, check torque. |
    | **4** | ⚙️ Tool Wear | Tool may be worn. Inspect or replace tool. |
    | **5** | 🔥 Critical Failure | Major anomaly. Immediate shutdown may be needed. |
    """)

# --- Machine Inputs ---
with st.expander("🔧 Input Machine Parameters", expanded=True):
    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        air_temp = st.number_input("🌡️ Air Temperature [K]", 280.0, 420.0, 300.0)
        torque = st.number_input("🌀 Torque [Nm]", 0.0, 100.0, 40.0)

    with col2:
        process_temp = st.number_input("🔥 Process Temperature [K]", 280.0, 450.0, 310.0)
        tool_wear = st.number_input("🛠️ Tool Wear [min]", 0.0, 300.0, 100.0)

    with col3:
        rotational_speed = st.number_input("⚙️ Rotational Speed [rpm]", 1000, 3000, 1500)
        machine_type = st.radio(
            "🏭 Machine Type",
            ["High Performance", "Low Power", "Medium Duty"],
            horizontal=True
        )

# --- One-hot Encoding for Machine Type ---
type_H = 1 if machine_type == "High Performance" else 0
type_L = 1 if machine_type == "Low Power" else 0
type_M = 1 if machine_type == "Medium Duty" else 0

# --- Combine Features ---
input_data = np.array([[air_temp, process_temp, torque, tool_wear,
                        rotational_speed, type_H, type_L, type_M]])

# --- Prediction Section ---
st.markdown("### 📈 Prediction")
st.markdown("Click the button below to run the prediction based on your inputs:")

if st.button("🔍 Predict Now"):
    try:
        # Preprocessing
        scaled = scaler.transform(input_data)
        reduced = pca.transform(scaled)
        dmatrix = xgb.DMatrix(reduced)

        # Predict (get class probabilities)
        prediction_probs = model.predict(dmatrix)
        prediction = int(np.argmax(prediction_probs[0]))  # Most likely class

        # Show prediction results
        status = status_map.get(prediction, "❓ Unknown Condition")
        st.markdown(f"### {status}")

        st.info("🧠 **AI Recommendation:**")
        for item in remedy.get(prediction, ["No advice available."]):
            st.markdown(f"- {item}")

        # Show probability breakdown
        st.markdown("#### 🔢 Class Probabilities")
        for i, prob in enumerate(prediction_probs[0]):
            st.markdown(f"- Class {i}: {prob:.2%}")

    except Exception as e:
        st.error(f"❌ Error during prediction: {str(e)}")

# --- Footer ---
st.divider()
st.caption("Developed with ❤️ using XGBoost + Streamlit for smarter maintenance solutions.")
