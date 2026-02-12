import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import mean_absolute_error, r2_score

# Load model
model = joblib.load("model.pkl")

st.title("📊 Amazon Sales Revenue Prediction")

st.write("Upload production.csv to evaluate model performance")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.write("Preview of Uploaded Data")
    st.dataframe(df.head())

    if "total_revenue" in df.columns:
        X = df.drop(columns=["total_revenue"])
        y_true = df["total_revenue"]

        predictions = model.predict(X)

        df["Predicted_Revenue"] = predictions

        mae = mean_absolute_error(y_true, predictions)
        r2 = r2_score(y_true, predictions)

        st.subheader("📈 Model Performance on Production Data")
        st.write(f"MAE: {mae}")
        st.write(f"R2 Score: {r2}")

        st.write("Sample Predictions")
        st.dataframe(df.head())

    else:
        st.error("CSV must contain 'total_revenue' column for accuracy testing")
    