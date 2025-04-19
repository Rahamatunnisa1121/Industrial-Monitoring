import streamlit as st
import numpy as np
import joblib
import xgboost as xgb
from transformers import GPT2Tokenizer, GPT2LMHeadModel, pipeline
import torch

# --- Load Pre-trained Objects ---
scaler = joblib.load("scaler.pkl")
pca = joblib.load("pca.pkl")
model = joblib.load("xgboost_model.pkl")  # XGBoost Booster

# ✅ MUST BE FIRST Streamlit command
st.set_page_config(page_title="AI Maintenance Assistant", layout="wide")

# --- GPT-2 Model Loader ---
@st.cache_resource
def load_local_chatbot():
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2-machine-health")
    model = GPT2LMHeadModel.from_pretrained("gpt2-machine-health")
    model.eval()
    return pipeline("text-generation", model=model, tokenizer=tokenizer)

chatbot = load_local_chatbot()

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
        scaled = scaler.transform(input_data)
        reduced = pca.transform(scaled)
        dmatrix = xgb.DMatrix(reduced)

        prediction_probs = model.predict(dmatrix)
        prediction = int(np.argmax(prediction_probs[0]))
        st.session_state.prediction = prediction

        status = status_map.get(prediction, "❓ Unknown Condition")
        st.markdown(f"### {status}")

        st.info("🧠 **AI Recommendation:**")
        for item in remedy.get(prediction, ["No advice available."]):
            st.markdown(f"- {item}")

        st.markdown("#### 🔢 Class Probabilities")
        for i, prob in enumerate(prediction_probs[0]):
            st.markdown(f"- Class {i}: {prob:.2%}")

    except Exception as e:
        st.error(f"❌ Error during prediction: {str(e)}")

# --- Chatbot Section ---
st.markdown("## 💬 Ask AI Assistant")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.text_input("👨‍🔧 Ask about your machine's condition, tips, or next steps:")

if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    # Prediction context
    if "prediction" in st.session_state:
        context = status_map.get(st.session_state.prediction, "Unknown condition")
        suggestions = remedy.get(st.session_state.prediction, ["No suggestions available."])
        suggestion_text = "\n".join(f"- {s}" for s in suggestions)
    else:
        context = "Machine condition not predicted yet."
        suggestion_text = "Please run the prediction to get condition insights."

    prompt = f"""Condition: {context}
Maintenance Suggestions:
{suggestion_text}

User: {user_input}
Assistant:"""

    try:
        output = chatbot(prompt, max_length=150, num_return_sequences=1, do_sample=True)[0]["generated_text"]
        reply = output.split("Assistant:")[-1].strip()
        st.session_state.chat_history.append({"role": "assistant", "content": reply})
    except Exception as e:
        st.error(f"❌ Chatbot error: {str(e)}")

# --- Display chat messages ---
for msg in st.session_state.chat_history:
    if msg["role"] == "user":
        st.markdown(f"👤 **You**: {msg['content']}")
    else:
        st.markdown(f"🤖 **Bot**: {msg['content']}")

# --- Footer ---
st.divider()
st.caption("Developed with ❤️ using GPT-2 + XGBoost + Streamlit for smarter maintenance solutions.")
