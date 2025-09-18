import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
from datetime import datetime
import sys

# --- Explicitly import libraries required by loaded models ---
# Crucial for resolving ModuleNotFoundError when Streamlit runs
try:
    import catboost
    import xgboost
    import sklearn # Specifically for RandomForestClassifier, SVC etc.
    print(f"Successfully imported CatBoost: {catboost.__version__}")
    print(f"Successfully imported XGBoost: {xgboost.__version__}")
    print(f"Successfully imported Scikit-learn: {sklearn.__version__}")
except ImportError as e:
    st.error(f"CRITICAL ERROR: Could not import a required ML library. Please ensure 'catboost', 'xgboost', and 'scikit-learn' are correctly installed in the environment Streamlit is using (Python: {sys.executable}). Error: {e}")
    st.stop()
# ------------------------------------------

# --- Configuration ---
ARTIFACTS_DIR = "streamlit_artifacts" # Directory where artifacts are saved
MODEL_NAMES = ["Random Forest", "XGBoost", "CatBoost"] # Models we want to load

st.set_page_config(layout="wide", page_title="Multi-Model Smuggling Predictor")

# --- Load Artifacts ---
@st.cache_resource # Cache resource loading for efficiency
def load_artifacts(artifacts_dir):
    """Loads the saved models, scaler, columns, and other artifacts."""
    loaded_models = {}
    model_performance = {}
    scaler = None
    training_columns = None
    categorical_values = None
    numerical_cols_to_scale = None
    all_loaded = True

    try:
        # Load Models
        for name in MODEL_NAMES:
            filename = f"{name.lower().replace(' ', '_')}_model.joblib"
            path = os.path.join(artifacts_dir, filename)
            if os.path.exists(path):
                loaded_models[name] = joblib.load(path)
                print(f"Loaded model: {name}")
            else:
                st.error(f"Artifact Error: Model file not found at {path}")
                all_loaded = False

        # Load Scaler
        scaler_path = os.path.join(artifacts_dir, 'scaler.joblib')
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
            print("Loaded scaler.")
        else:
            st.error(f"Artifact Error: Scaler file not found at {scaler_path}")
            all_loaded = False

        # Load Training Columns
        columns_path = os.path.join(artifacts_dir, 'training_columns.json')
        if os.path.exists(columns_path):
            with open(columns_path, 'r') as f:
                training_columns = json.load(f)
            print("Loaded training columns.")
        else:
            st.error(f"Artifact Error: Training columns file not found at {columns_path}")
            all_loaded = False

        # Load Categorical Values
        cat_values_path = os.path.join(artifacts_dir, 'categorical_values.json')
        if os.path.exists(cat_values_path):
            with open(cat_values_path, 'r') as f:
                categorical_values = json.load(f)
            print("Loaded categorical values.")
        else:
            st.error(f"Artifact Error: Categorical values file not found at {cat_values_path}")
            all_loaded = False

        # Load Performance Metrics
        metrics_path = os.path.join(artifacts_dir, 'all_model_performance.json')
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                model_performance = json.load(f)
            print("Loaded model performance metrics.")
        else:
            st.warning(f"Artifact Warning: Model performance file not found at {metrics_path}. Metrics will not be displayed.")
            model_performance = {} # Use empty dict if file is missing

        # Load Numerical Columns to Scale
        num_cols_path = os.path.join(artifacts_dir, 'numerical_cols_to_scale.json')
        if os.path.exists(num_cols_path):
             with open(num_cols_path, 'r') as f:
                 numerical_cols_to_scale = json.load(f)
             print("Loaded numerical columns to scale list.")
        else:
            st.error(f"Artifact Error: Numerical columns file not found at {num_cols_path}")
            all_loaded = False


        if not all_loaded:
            st.stop() # Stop execution if essential files are missing

        return loaded_models, scaler, training_columns, categorical_values, model_performance, numerical_cols_to_scale

    except Exception as e:
        st.error(f"An unexpected error occurred during artifact loading: {e}")
        st.error(f"Please ensure all required artifact files exist in the '{artifacts_dir}' directory and that necessary libraries (catboost, xgboost, scikit-learn) are installed in the correct environment.")
        st.stop()

# --- Load the artifacts ---
models, scaler, training_columns, categorical_values, model_performance, numerical_cols_to_scale = load_artifacts(ARTIFACTS_DIR)


# --- Define original columns (used for input form structure) ---
original_categorical_cols = list(categorical_values.keys()) if categorical_values else []
original_numerical_cols = ['poverty_rate', 'distance_to_checkpoint', 'population_density']

# --- Helper Function for Preprocessing (Identical to previous version) ---
def preprocess_data(df_input, scaler, training_columns, numerical_cols_to_scale, categorical_values):
    """Preprocesses the input DataFrame to match the training data format."""
    df = df_input.copy()
    if not original_categorical_cols: # Check if loading failed
        st.error("Preprocessing cannot proceed: Categorical values list not loaded.")
        return None

    # 1. Date Handling & Feature Engineering
    try:
        if 'date' in df.columns and 'time' in df.columns:
             df['datetime'] = pd.to_datetime(df['date'].astype(str) + ' ' + df['time'].astype(str), errors='coerce')
        elif 'date' in df.columns:
             df['datetime'] = pd.to_datetime(df['date'], errors='coerce')
        else:
            st.error("Input data must contain a 'date' column or 'date' and 'time' columns.")
            return None

        if df['datetime'].isnull().any():
            st.warning(f"Could not parse {df['datetime'].isnull().sum()} date/time value(s). Rows with invalid dates will be dropped.")
            df.dropna(subset=['datetime'], inplace=True)
            if df.empty:
                 st.error("No valid data remaining after dropping rows with invalid dates.")
                 return None

        df['hour'] = df['datetime'].dt.hour
        df['day_of_week'] = df['datetime'].dt.dayofweek
        df['day_of_month'] = df['datetime'].dt.day
        df['month'] = df['datetime'].dt.month
        df['year'] = df['datetime'].dt.year
    except Exception as e:
        st.error(f"Error processing date/time features: {e}")
        return None

    # 2. Handle Missing Categorical Values (Example: Impute with Mode)
    for col in original_categorical_cols:
        if col in df.columns:
            if df[col].isnull().any():
                # Use mode of the *original* training data's distribution if available,
                # otherwise use mode of current input batch or a placeholder.
                # For simplicity here, using mode of current batch or 'Unknown'.
                mode_val = df[col].mode()[0] if not df[col].mode().empty else 'Unknown'
                df[col].fillna(mode_val, inplace=True)
        else:
            st.error(f"Preprocessing Error: Expected categorical column '{col}' not found in input.")
            return None # Stop if required columns are missing

    # 3. One-Hot Encode Categorical Features
    try:
        # Ensure columns to encode exist
        cols_to_encode = [col for col in original_categorical_cols if col in df.columns]
        df = pd.get_dummies(df, columns=cols_to_encode, drop_first=True)
    except Exception as e:
        st.error(f"Error during one-hot encoding: {e}")
        return None

    # 4. Align Columns with Training Data
    try:
        df = df.reindex(columns=training_columns, fill_value=0)
    except Exception as e:
         st.error(f"Error aligning columns with training data: {e}")
         return None

    # 5. Scale Numerical Features
    cols_to_scale_present = [col for col in numerical_cols_to_scale if col in df.columns]
    if not cols_to_scale_present:
        st.warning("No numerical columns designated for scaling were found in the input.")
    else:
        try:
            df[cols_to_scale_present] = scaler.transform(df[cols_to_scale_present])
        except ValueError as e:
             st.error(f"Error scaling numerical features: {e}. Check if input data has unexpected values or structure after encoding.")
             # st.dataframe(df[cols_to_scale_present].head()) # Optional: show data causing error
             return None
        except Exception as e:
             st.error(f"An unexpected error occurred during scaling: {e}")
             return None

    # Ensure final columns match exactly (order matters for some models)
    return df[training_columns]

# --- Streamlit UI ---
st.title("🌍 Multi-Model Smuggling Incident Prediction")
st.markdown("Predict the likelihood of a smuggling incident using Random Forest, XGBoost, and CatBoost models.")
st.markdown("---")

# --- Sidebar ---
st.sidebar.header("Input Options")
input_method = st.sidebar.radio("Choose Input Method:", ("Manual Entry", "Upload CSV File"))

st.sidebar.header("Model Performance")
st.sidebar.markdown("_Metrics based on test set evaluation during training._")

# Display metrics for each model loaded
for model_name in MODEL_NAMES:
    if model_name in model_performance:
        metrics = model_performance[model_name]
        with st.sidebar.expander(f"📊 {model_name} Metrics", expanded=False):
            st.metric(label="ROC AUC", value=f"{metrics.get('roc_auc', 0):.4f}")
            st.metric(label="PR AUC", value=f"{metrics.get('pr_auc', 0):.4f}")
            st.metric(label="F1 Score (Smuggling)", value=f"{metrics.get('f1_score_positive_class', 0):.4f}")
    else:
        st.sidebar.warning(f"Performance metrics for {model_name} not found.")


# --- Main Area ---
input_df = None
analysis_requested = False # Flag to trigger prediction

if input_method == "Manual Entry":
    st.header("✍️ Manual Feature Input")
    col1, col2 = st.columns(2)

    form_inputs = {} # Dictionary to store inputs

    with col1:
        st.subheader("Contextual Factors")
        if 'location' in categorical_values:
            form_inputs['location'] = st.selectbox("📍 Location", options=categorical_values['location'])
        if 'weather_condition' in categorical_values:
            form_inputs['weather_condition'] = st.selectbox("🌦️ Weather Condition", options=categorical_values['weather_condition'])
        if 'geopolitical_event' in categorical_values:
            form_inputs['geopolitical_event'] = st.selectbox("⚖️ Geopolitical Event", options=categorical_values['geopolitical_event'])
        if 'terrain_type' in categorical_values:
            form_inputs['terrain_type'] = st.selectbox("🏞️ Terrain Type", options=categorical_values['terrain_type'])

        c1, c2 = st.columns(2)
        with c1:
             input_date = st.date_input("🗓️ Date")
        with c2:
             input_time = st.time_input("⏰ Time")
        # Combine date and time for preprocessing
        form_inputs['date'] = datetime.combine(input_date, input_time)

    with col2:
        st.subheader("Socio-Economic & Geographic")
        form_inputs['poverty_rate'] = st.number_input("💸 Poverty Rate (%)", min_value=0.0, max_value=100.0, value=20.0, step=0.1)
        form_inputs['distance_to_checkpoint'] = st.number_input("📏 Distance to Checkpoint (km)", min_value=0.0, value=15.0, step=0.1)
        form_inputs['population_density'] = st.number_input("👨‍👩‍👧‍👦 Population Density (per sq km)", min_value=0.0, value=300.0, step=1.0)

    if st.button("Predict Smuggling Likelihood", key="manual_predict"):
        input_df = pd.DataFrame([form_inputs]) # Create DataFrame from single entry
        analysis_requested = True


elif input_method == "Upload CSV File":
    st.header("📤 Upload Data File")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        try:
            input_df_uploaded = pd.read_csv(uploaded_file)
            st.write("Uploaded Data Preview:")
            st.dataframe(input_df_uploaded.head())

            # Basic Validation (Check if essential columns exist)
            # Include 'date' OR ('date', 'time') depending on expected format
            required_cols = original_categorical_cols + original_numerical_cols
            if 'date' not in input_df_uploaded.columns and ('date' not in input_df_uploaded.columns or 'time' not in input_df_uploaded.columns):
                 st.error("Uploaded CSV must contain a 'date' column, or both 'date' and 'time' columns.")
                 input_df = None
            else:
                 # Check for other required columns
                 missing_cols = [col for col in required_cols if col not in input_df_uploaded.columns]
                 if missing_cols:
                     st.error(f"Uploaded CSV is missing required columns: {', '.join(missing_cols)}")
                     input_df = None
                 else:
                     if st.button("Predict Smuggling Likelihood for Uploaded Data", key="file_predict"):
                          input_df = input_df_uploaded # Assign the uploaded df
                          analysis_requested = True

        except Exception as e:
            st.error(f"Error reading or validating CSV file: {e}")
            input_df = None


# --- Prediction and Display ---
if analysis_requested and input_df is not None:
    st.markdown("---")
    st.header("📈 Prediction Results")

    # Preprocess the input data (same preprocessing for all models)
    processed_df = preprocess_data(input_df, scaler, training_columns, numerical_cols_to_scale, categorical_values)

    if processed_df is not None and not processed_df.empty:
        try:
            # --- Get predictions from all models ---
            all_predictions = {}
            all_probabilities = {}
            results_df = input_df.copy() # Start with original input data for display

            with st.spinner("Running predictions..."):
                for name, model in models.items():
                    if name in MODEL_NAMES: # Only predict for loaded models
                        predictions = model.predict(processed_df)
                        probabilities = model.predict_proba(processed_df)[:, 1] # Probability of class '1'
                        all_predictions[name] = predictions
                        all_probabilities[name] = probabilities

                        # Add results to the DataFrame
                        results_df[f'{name} Prediction'] = predictions
                        results_df[f'{name} Probability (%)'] = (probabilities * 100).round(2)


            # --- Display Results ---
            if len(results_df) == 1: # Single manual prediction
                st.subheader("Model Predictions:")
                cols = st.columns(len(MODEL_NAMES))
                i = 0
                for name in MODEL_NAMES:
                    with cols[i]:
                        pred = all_predictions[name][0]
                        prob = all_probabilities[name][0] * 100
                        st.metric(label=f"{name}", value="Yes" if pred == 1 else "No", delta=f"{prob:.1f}% Probability", delta_color="inverse" if pred==1 else "normal")
                    i += 1

                # Show input features used (optional)
                with st.expander("Show Input Features Used for Prediction"):
                    st.dataframe(input_df)

            else: # Batch prediction from file
                st.subheader("Predictions for Uploaded Data:")

                # Add an agreement column (enhancement)
                pred_cols = [f'{name} Prediction' for name in MODEL_NAMES]
                results_df['Model Agreement (Yes)'] = results_df[pred_cols].sum(axis=1) # Counts number of models predicting 1
                results_df['Model Agreement (No)'] = len(MODEL_NAMES) - results_df['Model Agreement (Yes)']

                # Display the DataFrame with results
                st.dataframe(results_df)

                # Show aggregate stats
                st.subheader("Prediction Summary")
                cols = st.columns(len(MODEL_NAMES) + 1) # One column per model + 1 for overall
                with cols[0]:
                    st.metric("Total Records", len(results_df))

                for i, name in enumerate(MODEL_NAMES):
                    with cols[i+1]:
                        num_incidents = results_df[f'{name} Prediction'].sum()
                        avg_prob = results_df[f'{name} Probability (%)'].mean()
                        st.metric(f"{name}: Predicted Incidents", f"{num_incidents}")
                        st.caption(f"Avg. Probability: {avg_prob:.2f}%")


                # Download results
                @st.cache_data # Cache the conversion to CSV
                def convert_df_to_csv(df):
                    return df.to_csv(index=False).encode('utf-8')

                csv_data = convert_df_to_csv(results_df)
                st.download_button(
                    label="Download Predictions as CSV",
                    data=csv_data,
                    file_name='multi_model_smuggling_predictions.csv',
                    mime='text/csv',
                )

        except AttributeError as e:
             st.error(f"Prediction Error: An issue occurred, possibly because a model object wasn't loaded correctly or doesn't have a required method (like predict_proba). Error: {e}")
             st.error(f"Models loaded: {list(models.keys())}")
        except Exception as e:
            st.error(f"An unexpected error occurred during prediction: {e}")
            st.error("Please ensure the input data format is correct and compatible with the models.")
            st.write("Columns fed into the models after preprocessing:")
            st.dataframe(processed_df.head()) # Show what was fed to the model

    elif processed_df is not None and processed_df.empty:
         st.warning("No valid data remained after preprocessing (e.g., due to invalid dates or missing required columns). Cannot make predictions.")
    else:
         st.error("Data preprocessing failed. Cannot make predictions.") # Error message likely shown in preprocess_data

elif not analysis_requested:
     st.info("Enter features manually or upload a CSV file and click 'Predict' to get started.")

st.markdown("---")
st.caption("Disclaimer: Predictions are based on machine learning models trained on historical data and should be used for informational purposes only. Evaluate results from all models.")