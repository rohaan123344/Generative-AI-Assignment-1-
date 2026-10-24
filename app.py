
import os
import random
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf

from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ==========================================
# 1. CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="ANN Salary Regression",
    page_icon="💰",
    layout="wide"
)

RANDOM_STATE = 42
DATA_PATH = "Churn_Modelling.csv"
MODEL_PATH = "salary_model.keras"
PREPROCESSOR_PATH = "salary_preprocessor.joblib"
SCALER_PATH = "salary_target_scaler.joblib"

random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)


# ==========================================
# 2. TRAIN OR LOAD MODEL
# ==========================================

@st.cache_resource
def load_or_train_model():

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"{DATA_PATH} not found. "
            "Place Churn_Modelling.csv beside app.py."
        )

    data = pd.read_csv(DATA_PATH)

    if "EstimatedSalary" not in data.columns:
        raise ValueError(
            "EstimatedSalary column was not found in the dataset."
        )

    # Remove identifiers, target and unrelated churn outcome
    drop_columns = [
        "RowNumber",
        "CustomerId",
        "Surname",
        "Exited",
        "EstimatedSalary"
    ]

    X = data.drop(
        columns=[
            col for col in drop_columns
            if col in data.columns
        ]
    ).copy()

    y = pd.to_numeric(
        data["EstimatedSalary"],
        errors="coerce"
    )

    # Remove rows with invalid target values
    valid = y.notna()
    X = X.loc[valid].reset_index(drop=True)
    y = y.loc[valid].reset_index(drop=True)

    if len(X) < 10:
        raise ValueError("Not enough valid rows to train the model.")

    # Split into training, validation and test sets
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=0.20,
        random_state=RANDOM_STATE
    )

    # Identify feature types
    numeric_features = X_train.select_dtypes(
        include=["number", "bool"]
    ).columns.tolist()

    categorical_features = [
        col for col in X_train.columns
        if col not in numeric_features
    ]

    # Numerical preprocessing
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    # Categorical preprocessing
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    transformers = []

    if numeric_features:
        transformers.append(
            ("numeric", numeric_pipeline, numeric_features)
        )

    if categorical_features:
        transformers.append(
            ("categorical", categorical_pipeline,
             categorical_features)
        )

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop"
    )

    # Fit preprocessing on training data only
    X_train_processed = preprocessor.fit_transform(X_train)
    X_val_processed = preprocessor.transform(X_val)
    X_test_processed = preprocessor.transform(X_test)

    # Convert sparse matrices to dense arrays if necessary
    def to_dense_float32(values):
        if hasattr(values, "toarray"):
            values = values.toarray()
        return np.asarray(values, dtype=np.float32)

    X_train_processed = to_dense_float32(X_train_processed)
    X_val_processed = to_dense_float32(X_val_processed)
    X_test_processed = to_dense_float32(X_test_processed)

    # Scale target values
    target_scaler = StandardScaler()

    y_train_scaled = target_scaler.fit_transform(
        y_train.to_numpy().reshape(-1, 1)
    ).astype(np.float32)

    y_val_scaled = target_scaler.transform(
        y_val.to_numpy().reshape(-1, 1)
    ).astype(np.float32)

    # Build ANN regression model
    model = keras.Sequential([
        keras.Input(shape=(X_train_processed.shape[1],)),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dropout(0.20),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dense(1, activation="linear")
    ])

    # MSE is suitable for regression
    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=0.001
        ),
        loss="mse",
        metrics=["mae"]
    )

    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True
    )

    # Train ANN
    model.fit(
        X_train_processed,
        y_train_scaled,
        validation_data=(X_val_processed, y_val_scaled),
        epochs=100,
        batch_size=32,
        callbacks=[early_stopping],
        verbose=0
    )

    # Evaluate on untouched test data
    pred_scaled = model.predict(
        X_test_processed,
        verbose=0
    )

    predictions = target_scaler.inverse_transform(
        pred_scaled
    ).ravel()

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    # Save artifacts
    model.save(MODEL_PATH)
    joblib.dump(preprocessor, PREPROCESSOR_PATH)
    joblib.dump(target_scaler, SCALER_PATH)

    metrics = {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "R2": float(r2)
    }

    return (
        model,
        preprocessor,
        target_scaler,
        X,
        metrics,
        True
    )


# ==========================================
# 3. STREAMLIT INTERFACE
# ==========================================

st.title("💰 ANN Salary Regression")
st.write(
    "Predict Estimated Salary using an Artificial Neural Network "
    "trained on Churn_Modelling.csv."
)

st.info(
    "The target is EstimatedSalary. It is excluded from the "
    "input features to prevent target leakage."
)

try:
    with st.spinner(
        "Loading model or training the ANN for the first time..."
    ):
        (
            model,
            preprocessor,
            target_scaler,
            X_reference,
            metrics,
            newly_trained
        ) = load_or_train_model()

except Exception as error:
    st.error(f"Unable to load or train the model: {error}")
    st.stop()


# ==========================================
# 4. SHOW MODEL PERFORMANCE
# ==========================================

if newly_trained:
    st.success("ANN trained and saved successfully.")

st.subheader("Model Evaluation")

col1, col2, col3 = st.columns(3)

col1.metric("MAE", f"{metrics['MAE']:,.2f}")
col2.metric("RMSE", f"{metrics['RMSE']:,.2f}")
col3.metric("R² Score", f"{metrics['R2']:.4f}")

st.caption(
    "Metrics are calculated on the held-out test set during training. "
    "Lower MAE and RMSE generally indicate smaller errors; "
    "R² closer to 1 generally indicates a better fit."
)


# ==========================================
# 5. USER INPUT FORM
# ==========================================

st.divider()
st.subheader("Enter Customer Details")

st.write(
    "Enter the customer's information below to estimate their salary."
)

input_values = {}

with st.form("salary_prediction_form"):

    col1, col2 = st.columns(2)

    for index, column in enumerate(X_reference.columns):

        container = col1 if index % 2 == 0 else col2
        series = X_reference[column]

        with container:

            if (
                pd.api.types.is_numeric_dtype(series)
                or pd.api.types.is_bool_dtype(series)
            ):
                default_value = series.median()

                if pd.isna(default_value):
                    default_value = 0

                input_values[column] = st.number_input(
                    column,
                    value=float(default_value),
                    key=f"input_{column}"
                )

            else:
                choices = (
                    series.dropna()
                    .astype(str)
                    .unique()
                    .tolist()
                )

                if not choices:
                    choices = ["Unknown"]

                input_values[column] = st.selectbox(
                    column,
                    options=choices,
                    key=f"input_{column}"
                )

    submitted = st.form_submit_button(
        "Predict Estimated Salary",
        type="primary",
        use_container_width=True
    )


# ==========================================
# 6. PREDICTION
# ==========================================

if submitted:

    input_df = pd.DataFrame(
        [input_values],
        columns=X_reference.columns
    )

    try:
        processed_input = preprocessor.transform(input_df)

        if hasattr(processed_input, "toarray"):
            processed_input = processed_input.toarray()

        processed_input = np.asarray(
            processed_input,
            dtype=np.float32
        )

        prediction_scaled = model.predict(
            processed_input,
            verbose=0
        )

        prediction = target_scaler.inverse_transform(
            prediction_scaled
        )[0, 0]

        st.divider()
        st.subheader("Salary Prediction")

        st.success("Prediction completed successfully.")

        st.metric(
            label="Estimated Salary",
            value=f"{prediction:,.2f}"
        )

        st.write("Input details:")
        st.dataframe(input_df, use_container_width=True)

        st.caption(
            "This is a model estimate, not a guaranteed salary."
        )

    except Exception as error:
        st.error(f"Prediction failed: {error}")


# ==========================================
# 7. FOOTER
# ==========================================

st.divider()
st.caption(
    "Assignment: ANN Salary Regression | "
    "Dataset: Churn_Modelling.csv | "
    "Output: One neuron with linear activation | "
    "Loss: Mean Squared Error (MSE)"
)
