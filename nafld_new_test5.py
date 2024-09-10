import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load the pre-trained machine learning model
with open('models/direct_model_5features.pkl', 'rb') as f:
    model = pickle.load(f)

# Load the scaler
with open('models/sscaler_5features.pkl', 'rb') as f1:
    scaler = pickle.load(f1)

# Parameters required for the model
params1 = ['Age', 'PLT', 'DM', 'Serum AST/SGOT', 'Serum Cholesterol']
params = ['Age', 'PLT', 'Diabetes mellitus (yes/no)', 'Serum AST/SGOT', 'Serum Cholesterol']


def update_color_and_status(value):
    if value <= 35:
        pred = "strong"
    elif 35 < value <= 75:
        pred = "weak"
    else:
        pred = "strong"
        
    return pred

def make_prediction(param_values,unit_values):
    try:
        
        # Convert 'Diabetes mellitus' to numerical values
        if param_values[2] == "Yes":
            param_values[2] = 1
        elif param_values[2] == "No":
            param_values[2] = 2
        
        # Convert select box inputs to binary values for PLT, Serum Cholesterol, and Serum LDL Cholesterol

        if unit_values[0] == "mmol/L":
            param_values[4] = float(param_values[4]) * 18.018
        
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
st.title("FAET: Fibrosis Assessment using Extra Tree")
st.markdown("Please enter the following information:")
unit_values = []
# Collect user inputs for each parameter using text inputs or select boxes
for i, param in enumerate(params):
    # 'Diabetes mellitus' as select box
    if param.startswith('Diabetes mellitus'):
        st.session_state.param_values[param] = st.selectbox(param, options=["Yes", "No"], key=param.lower().replace(' ', '_'))
    # Select boxes for PLT, Serum Cholesterol, Serum LDL Cholesterol
    elif i in [0]:
        # Text input for numeric values and select box for Yes/No
        col1, col2 = st.columns([2, 1])
        with col1:
            st.session_state.param_values[param] = st.text_input(param, key=param.lower().replace(' ', '_'), value=st.session_state.param_values[param])
        with col2:
            st.text_input(' ',"Years",disabled = True)
    elif i in [1]:
        # Text input for numeric values and select box for Yes/No
        col1, col2 = st.columns([2, 1])
        with col1:
            st.session_state.param_values[param] = st.text_input(param, key=param.lower().replace(' ', '_'), value=st.session_state.param_values[param])
        with col2:
            st.selectbox(" ", options=["10⁹/L", "10³ / µL"], key=(param.lower() + "_select").replace(' ', '_'))

    elif i in [3]:
        # Text input for numeric values and select box for Yes/No
        col1, col2 = st.columns([2, 1])
        with col1:
            st.session_state.param_values[param] = st.text_input(param, key=param.lower().replace(' ', '_'), value=st.session_state.param_values[param])
        with col2:
            st.text_input(' ',"U/L",disabled = True)

    elif i in [4]:
        # Text input for numeric values and select box for Yes/No
        col1, col2 = st.columns([2, 1])
        with col1:
            st.session_state.param_values[param] = st.text_input(param, key=param.lower().replace(' ', '_'), value=st.session_state.param_values[param])
        with col2:
            unit_values.append(st.selectbox(" ", options=["mg/dl", "mmol/L"], key=(param.lower() + "_select").replace(' ', '_')))
    else:
        st.session_state.param_values[param] = st.text_input(param, key=param.lower().replace(' ', '_'), value=st.session_state.param_values[param])


if st.button("Predict"):
    try:   
        param_values_float = [st.session_state.param_values[param] for param in params]
        outcome, score = make_prediction(param_values_float, unit_values)

        if outcome is not None and score is not None:
            outcome_new = f"{outcome}"
            score_new = f"{score:.2f}%"
            pred = update_color_and_status(score)
            pred_new = f"{pred}"
            
            # Display predictions in two rows and two columns
            col1, col2 = st.columns([1, 1])
            with col1:
                st.metric("Outcome", outcome_new)
                st.metric("Prediction confidence", pred_new)
            with col2:
                st.metric("Prediction Score:", score_new)
            
    except ValueError:
        st.error("Please enter valid numerical values for all parameters.")
