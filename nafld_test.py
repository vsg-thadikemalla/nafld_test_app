import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load the pre-trained machine learning model
with open('models/direct_model_6features.pkl', 'rb') as f:
    model = pickle.load(f)

# Load the scaler
with open('models/sscaler_6features.pkl', 'rb') as f1:
    scaler = pickle.load(f1)

# Parameters required for the model
# params1 = ['Age', 'Gender', 'Wt', 'DM', 'HB', 'TLC', 'PLT', 'Albumin', 'Serum Alkaline Phosphatase', 'Serum ALT /SGPT', 'Serum AST/SGOT', 'Serum Bilirubin Direct', 'Serum Bilirubin Indirect', 'Serum Bilirubin Total', 'Serum GGT', 'Serum Cholesterol', 'Serum Triglycerides', 'Serum HDL  Cholesterol', 'Serum LDL Cholesterol']'''

params1 = ['Age', 'PLT', 'DM', 'Serum AST/SGOT', 'Serum Cholesterol', 'Serum LDL Cholesterol']

params = ['Age (Years); Range: 18 to 71', 'PLT (10^9/L); Range: 20 to 442', 'Diabetes mellitus (Yes/No); (1- Yes, 2-No) ', 'Serum AST/SGOT (U/L); Range: 11 to 484', 'Serum Cholesterol (mg/L); Range: 34 to 357', 'Serum LDL Cholesterol (mg/L); Range: 9.3 to 233']


def update_color_and_status(value):
    st.markdown(
        f"""
        <style>
        .progress {{
            position: relative;
            width: 100%;
            height: 30px;
            background-color: #e0e0df;
            border-radius: 5px;
        }}
        .progress-bar {{
            position: absolute;
            height: 100%;
            width: {value}%;
            background-color: {"green" if value <= 35 else "#e9d460" if value <= 75 else "red"};
            border-radius: 5px;
        }}
        .progress-line {{
            position: absolute;
            height: 100%;
            width: 2px;
            background-color: black;
        }}
        .line-35 {{
            left: 35%;
        }}
        .line-75 {{
            left: 75%;
        }}
        </style>
        <div class="progress">
            <div class="progress-bar"></div>
            <div class="progress-line line-35"></div>
            <div class="progress-line line-75"></div>
        </div>
        """, unsafe_allow_html=True)

    if value <= 35:
        st.success("NAF - High prediction confidence")
    elif 35 < value <= 75:
        st.warning("Weak prediction confidence")
    else:
        st.error("AF - High prediction confidence")

def make_prediction(param_values):
    try:
        param_values = [float(value) if value != "" else None for value in param_values]
        param_values = np.array(param_values).reshape(1, -1)
        X_test = pd.DataFrame(param_values, columns=params1)
        X_t = scaler.transform(X_test)
        X_t = pd.DataFrame(X_t, columns=X_test.columns, index=X_test.index)

        predictions = model.predict(X_t)

        if int(predictions[0]) == 0:
            outcome = 'No Advanced Fibrosis'
            score = model.predict_proba(X_t)[0][1]
        else:
            outcome = 'Advanced Fibrosis'
            score = model.predict_proba(X_t)[0][1]

        return outcome, score * 100

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        return None, None

# Initialize session state
if 'param_values' not in st.session_state:
    st.session_state.param_values = {param: "" for param in params}

# Set up Streamlit app interface with title and instructions to enter parameter values
st.title("NAFLD Prediction App")
st.markdown("Please enter the following information:")

# Collect user inputs for each parameter using text inputs
for param in params:
    # Use param as the key and default value as the value retrieved from session state
    st.session_state.param_values[param] = st.text_input(param, key=param.lower().replace(' ', '_'), value=st.session_state.param_values[param])

if st.button("Predict"):
    try:
        param_values_float = [float(value) if value else None for value in st.session_state.param_values.values()]
        outcome, score = make_prediction(param_values_float)
        if outcome is not None and score is not None:
            st.write(f"Prediction: {outcome}")
            st.write(f"Prediction Score: {score:.2f}%")
            update_color_and_status(score)
    except ValueError:
        st.error("Please enter valid numerical values for all parameters.")
