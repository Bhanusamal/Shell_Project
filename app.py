import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.cluster import KMeans  # Only if you're adding clustering

# Load saved models
selected_features = joblib.load("selected_features.pkl")
pca = joblib.load("pca_model.pkl")

st.title("🌍 Climate Feature Explorer")
st.markdown("Project environmental data into PCA space to explore patterns and trends.")

st.sidebar.header("Enter Environmental Data")
user_input = {}
for feature in selected_features:
    user_input[feature] = st.sidebar.number_input(f"{feature}", value=0.0)

input_df = pd.DataFrame([user_input])
input_df.fillna(0, inplace=True)

if st.button("Project into PCA Space"):
    projection = pca.transform(input_df)
    st.success("✅ Projection complete!")
    st.write("📍 PCA Coordinates:", {f"PC{i+1}": round(val, 4) for i, val in enumerate(projection[0])})

    # Optional: Load historical PCA data
    try:
        historical_df = pd.read_csv("pca_transformed.csv")
        historical_df.columns = [f"PC{i+1}" for i in range(historical_df.shape[1])]
        user_point = pd.DataFrame(projection, columns=historical_df.columns)
        combined = pd.concat([historical_df, user_point], ignore_index=True)

        st.subheader("📊 PCA Projection with Historical Context")
        st.scatter_chart(combined.iloc[:, :2])  # PC1 vs PC2

        # Optional clustering
        kmeans = KMeans(n_clusters=3, random_state=42)
        clusters = kmeans.fit_predict(historical_df)
        user_cluster = kmeans.predict(user_point)[0]
        st.info(f"🧭 Your input falls into Cluster {user_cluster}")

    except FileNotFoundError:
        st.warning("No historical PCA data found. Upload 'pca_transformed.csv' to enable comparison.")
