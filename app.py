import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Bias Mitigation Evaluation", layout="wide")

# ----------------------------- Introduction -----------------------------
st.title("🔍 Identifying and Mitigating Bias in AI Training Data (IBM)")
st.markdown("""
This project evaluates and compares **bias mitigation techniques** applied to income prediction using racial group fairness.
in ACSincome Dataset We tested:
- **No Mitigation (Baseline)**
- **Pre-processing**: AIF360's *Reweighing*
- **In-processing**: Custom loss to reduce racial disparity
- **Post-processing**: *Calibrated Equalized Odds*

Metrics used:
- **Accuracy**
- **DIR** (Disparate Impact Ratio)
- **DPD** (Demographic Parity Difference)
- **Equal Opportunity Difference**
- **Average Odds Difference**
""")
# ----------------------------- Metric Explanations -----------------------------
with st.expander("ℹ️ What Do These Metrics Mean? Click to Expand"):
    st.markdown("""
### **1. Accuracy**  
- Measures overall correctness of predictions. Ideal is **1.0**, but high accuracy ≠ fairness.  

### **2. Disparate Impact Ratio (DIR)**  
- Checks if outcomes are proportionally equal across groups. Ideal is **1.0** (DIR < 0.8 signals bias).  

### **3. Demographic Parity Difference (DPD)**  
- Tracks gaps in positive prediction rates between groups. Ideal is **0.0** (non-zero = unequal access).  

### **4. Equal Opportunity Difference**  
- Compares true positive rates (e.g., qualified hires). Ideal is **0.0** (non-zero = unfair advantages).  

### **5. Average Odds Difference**  
- Balances gaps in both false positives and true positives. Ideal is **0.0** (bias exists if non-zero).  

**📌 Summary**: All metrics aim for **0.0 or 1.0**; deviations reveal bias. Trade-offs with accuracy are common.
    """)


# ----------------------------- Metrics -----------------------------
results = {
    "Method": ["No Mitigation", "Pre-processing", "In-processing", "Post-processing"],
    "Accuracy": [0.8014, 0.7536, 0.7898, 0.9410],
    "DIR": [0.7047, 0.9072, 0.9142, 0.7960],
    "DPD": [-0.1507, -0.0471, -0.0452, -0.1043],
    "EOD": [-0.0993, None, 0.0014, -0.0073],
    "AOD": [-0.0874, None, 0.0159, -0.0080]
}
df = pd.DataFrame(results)

# ----------------------------- Method Selector -----------------------------
method = st.selectbox("Select a Method to Explore in Detail", df["Method"].tolist())
selected_row = df[df["Method"] == method].iloc[0]

st.subheader(f"📊 Evaluation: {method}")
st.write("### Performance and Fairness Metrics")
metric_cols = st.columns(3)
metric_cols[0].metric("Accuracy", f"{selected_row['Accuracy']:.4f}")
metric_cols[1].metric("DIR", f"{selected_row['DIR']:.4f}")
metric_cols[2].metric("DPD", f"{selected_row['DPD']:.4f}")

metric_cols2 = st.columns(3)
metric_cols2[0].metric("Equal Opportunity Diff", f"{selected_row['EOD']:.4f}" if selected_row["EOD"] is not None else "N/A")
metric_cols2[1].metric("Avg Odds Diff", f"{selected_row['AOD']:.4f}" if selected_row["AOD"] is not None else "N/A")

# ----------------------------- Comparison Chart -----------------------------
st.write("## 📈 Comparison Across Methods")
chart_metric = st.selectbox("Choose metric to compare", ["Accuracy", "DIR", "DPD", "EOD", "AOD"])
fig = px.bar(df, x="Method", y=chart_metric, color="Method", text_auto=True)
st.plotly_chart(fig, use_container_width=True)

# ----------------------------- Interpretation -----------------------------
st.write("### 🔍 Interpretation Summary")
st.markdown("""
- **Baseline** shows high bias (low DIR, high DPD).
- **Pre-processing** significantly improved fairness (DIR ↑).
- **In-processing** achieved a strong balance between accuracy and fairness.
- **Post-processing** has the **highest accuracy** with good fairness.
""")

st.markdown("---")
st.caption("Developed by Rohan Kini,Taran Bhaskar adithya , Vandana S Patil ,Shree Hari. Powered by Streamlit + AIF360 + PyTorch + scikit-learn.")

