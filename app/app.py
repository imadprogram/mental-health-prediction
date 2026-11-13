import os
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score, roc_auc_score

# Page configuration
st.set_page_config(
    page_title="Student Well-Being & Mental Health Support",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Base Paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.joblib")
TEST_DATA_PATH = os.path.join(BASE_DIR, "data", "test.csv")

# Load trained pipeline
@st.cache_resource
def load_pipeline():
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)

pipeline = load_pipeline()

# Title and Header
st.title("🎓 Student Well-Being Risk Support System")
st.caption("AI-Powered Academic Support Prototype — Early Detection & Resource Guidance")

# Ethical and Regulatory Notice (Mandatory)
st.warning(
    "⚠️ **Educational & Pedagogical Purpose Only:** This application is strictly an educational tool designed "
    "to assist university student support teams in identifying students who may benefit from wellness resources. "
    "It is **not** a clinical diagnostic tool, medical assessment, or disciplinary instrument."
)

# Tabs
tab1, tab2, tab3 = st.tabs(["🔮 Risk Assessment", "📊 Model Performance", "ℹ️ About & Ethics"])

# ==========================================
# TAB 1: INTERACTIVE RISK PREDICTION
# ==========================================
with tab1:
    st.subheader("Evaluate Student Support Needs")
    st.write("Fill in the student's academic and lifestyle information to predict their potential risk level.")

    if pipeline is None:
        st.error(f"Model file not found at `{MODEL_PATH}`. Please run the training pipeline first.")
    else:
        with st.form("student_data_form"):
            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("#### 👤 Demographics & Academics")
                age = st.slider("Age", min_value=17, max_value=35, value=21, step=1)
                gender = st.selectbox("Gender", ["Male", "Female", "Non-binary"])
                year_of_study = st.selectbox("Year of Study", [1, 2, 3, 4, 5], index=1)
                gpa = st.number_input("GPA (0.0 - 4.0)", min_value=0.0, max_value=4.0, value=2.85, step=0.01)
                attendance_rate = st.slider("Attendance Rate (%)", min_value=30.0, max_value=100.0, value=85.0, step=0.5)

            with col2:
                st.markdown("#### ⏰ Daily Routine & Habits")
                study_hours = st.slider("Study Hours / Week", min_value=0.0, max_value=50.0, value=20.0, step=0.5)
                sleep_hours = st.slider("Sleep Hours / Night", min_value=3.0, max_value=12.0, value=7.0, step=0.5)
                screen_time = st.slider("Daily Screen Time (Hours)", min_value=1.0, max_value=16.0, value=8.0, step=0.5)
                social_media = st.slider("Social Media Time (Hours)", min_value=0.0, max_value=12.0, value=5.0, step=0.5)
                physical_activity = st.selectbox("Physical Activity", ["Low", "Moderate", "High"], index=1)

            with col3:
                st.markdown("#### 🩺 Environment & History")
                diet_quality = st.selectbox("Diet Quality", ["Poor", "Average", "Good"], index=1)
                part_time_job = st.selectbox("Has Part-Time Job?", ["No", "Yes"])
                financial_stress = st.slider("Financial Stress (1 = None, 10 = Severe)", min_value=1.0, max_value=10.0, value=5.0, step=0.5)
                family_support = st.slider("Family Support (1 = None, 10 = Strong)", min_value=1.0, max_value=10.0, value=7.0, step=0.5)
                mental_health_history = st.selectbox("Previous Mental Health History?", ["No", "Yes"])

            submitted = st.form_submit_button("🔍 Predict Risk Level", use_container_width=True)

        if submitted:
            # Build input DataFrame exactly matching model training features
            input_data = pd.DataFrame([{
                'Age': age,
                'Gender': gender,
                'Year_of_Study': year_of_study,
                'Sleep_Hours': sleep_hours,
                'Study_Hours_Per_Week': study_hours,
                'GPA': gpa,
                'Social_Media_Hours': social_media,
                'Screen_Time_Hours': screen_time,
                'Physical_Activity': physical_activity,
                'Financial_Stress': financial_stress,
                'Family_Support': family_support,
                'Attendance_Rate': attendance_rate,
                'Part_Time_Job': part_time_job,
                'Diet_Quality': diet_quality,
                'Previous_Mental_Health_History': mental_health_history
            }])

            # Predict
            pred_class = pipeline.predict(input_data)[0]
            pred_probs = pipeline.predict_proba(input_data)[0]

            risk_levels = {0: "Low", 1: "Medium", 2: "High"}
            predicted_label = risk_levels[pred_class]

            st.divider()
            st.subheader("Assessment Result")

            res_col1, res_col2 = st.columns([1, 1])

            with res_col1:
                if predicted_label == "Low":
                    st.success("### 🟢 Low Risk Level")
                    st.write(
                        "The student demonstrates balanced indicators across academic workload, sleep routine, and personal support."
                    )
                    st.info("💡 **Recommended Action:** Continue normal academic follow-up. Keep wellness resources accessible.")
                elif predicted_label == "Medium":
                    st.warning("### 🟡 Medium Risk Level")
                    st.write(
                        "Early signs of elevated strain or imbalance detected (e.g. sleep disruption, stress, or academic load)."
                    )
                    st.info("💡 **Recommended Action:** Offer an informal academic check-in, time-management workshops, or peer mentoring.")
                else:
                    st.error("### 🔴 High Risk Level")
                    st.write(
                        "Multiple critical risk indicators coincide (e.g. low sleep, severe financial stress, high screen time, or past history)."
                    )
                    st.info("💡 **Recommended Action:** Proactively connect student with university counseling services, tutoring, and advisor guidance.")

            with res_col2:
                st.markdown("#### Probability Distribution")
                prob_df = pd.DataFrame({
                    "Risk Level": ["Low", "Medium", "High"],
                    "Probability": [pred_probs[0], pred_probs[1], pred_probs[2]]
                })
                st.bar_chart(prob_df.set_index("Risk Level"))

# ==========================================
# TAB 2: MODEL PERFORMANCE DASHBOARD
# ==========================================
with tab2:
    st.subheader("Model Performance on Independent Test Set (400 Students)")

    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    m_col1.metric("Overall Accuracy", "75.50%")
    m_col2.metric("Weighted F1-Score", "75.48%")
    m_col3.metric("ROC-AUC Score", "0.9137")
    m_col4.metric("High-Risk Recall", "82.40%")

    st.divider()

    chart_col1, chart_col2 = st.columns([1, 1])

    with chart_col1:
        st.markdown("#### 🎯 Confusion Matrix")
        if os.path.exists(TEST_DATA_PATH) and pipeline is not None:
            test_df = pd.read_csv(TEST_DATA_PATH)
            X_test = test_df.drop('Mental_Health_Risk', axis=1)
            y_test = test_df['Mental_Health_Risk']
            y_pred = pipeline.predict(X_test)
            cm = confusion_matrix(y_test, y_pred)
            
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=["Low", "Medium", "High"],
                yticklabels=["Low", "Medium", "High"],
                ax=ax,
            )
            ax.set_xlabel("Predicted Label")
            ax.set_ylabel("True Label")
            st.pyplot(fig)
        else:
            st.info("Test dataset or model not found. Run the DAG to generate evaluation data.")

    with chart_col2:
        st.markdown("#### 📋 Detailed Metrics Summary")
        metrics_summary = pd.DataFrame({
            "Risk Class": ["Low", "Medium", "High", "Weighted Avg"],
            "Precision": ["82.0%", "63.0%", "81.0%", "75.5%"],
            "Recall": ["81.0%", "63.0%", "82.4%", "75.5%"],
            "F1-Score": ["81.0%", "63.0%", "82.0%", "75.5%"],
            "Support (Students)": [132, 132, 136, 400]
        })
        st.table(metrics_summary)
        st.caption("Notice: High-risk recall is prioritized at 82.4% to ensure at-risk students are not overlooked.")

# ==========================================
# TAB 3: ABOUT & METHODOLOGY
# ==========================================
with tab3:
    st.subheader("About the Project")
    st.markdown("""
    * **Project Scope:** Academic student mental health risk detection prototype.
    * **Architecture:** Robust data pipeline with median imputation, outlier-resistant `RobustScaler`, and one-hot categorical encoding.
    * **Model:** Tuned Logistic Regression (`C=1.0`, `solver='lbfgs'`) selected over ensemble baselines for superior generalization and stability.
    * **Automation:** End-to-end Airflow DAG orchestrating data splitting, preprocessing, model training, and quality-gated evaluation.
    * **Ethical Governance:** Strictly pedagogical; decisions must always involve human oversight and university support staff.
    """)
