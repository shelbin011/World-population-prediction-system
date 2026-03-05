import streamlit as st
import pandas as pd
from ml_model import prepare_data, train_polynomial_model, predict_years, plot_fit

st.set_page_config(page_title="Population Prediction", layout="centered")
st.title("Population Prediction (Polynomial Regression)")

uploaded = st.file_uploader("Upload world_population.csv (or select example below)", type='csv')

if uploaded is None:
    st.info("Upload a CSV file with columns like 'Country/Territory' and '1970 Population', '1980 Population', ...")

country = st.text_input("Country", value="India")
degree = st.slider("Polynomial degree", 1, 10, 3)
future_years_input = st.text_input("Future years (comma-separated)", value="2030,2040,2050")

if st.button("Train and Predict"):
    if uploaded is None:
        st.error("Please upload a CSV file first")
    else:
        try:
            df = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"Could not read CSV: {e}")
            st.stop()

        try:
            new_df = prepare_data(df, country=country)
        except Exception as e:
            st.error(str(e))
            st.stop()

        poly, model = train_polynomial_model(new_df, degree=degree)

        # parse future years
        years = []
        for part in future_years_input.split(','):
            p = part.strip()
            if not p:
                continue
            try:
                years.append(int(p))
            except ValueError:
                st.warning(f"Skipping invalid year: {p}")

        preds = predict_years(poly, model, years) if years else []

        st.subheader("Training data")
        st.dataframe(new_df)

        if years:
            st.subheader("Predictions")
            for y, p in zip(years, preds):
                st.write(f"**{y}**: {int(p):,}")

        fig = plot_fit(new_df, poly, model)
        st.pyplot(fig)
