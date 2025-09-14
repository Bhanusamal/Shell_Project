# 🌍 Climate Feature Explorer

An interactive Streamlit app for projecting environmental data into PCA space to explore climate patterns, detect anomalies, and visualize trends. Designed for unsupervised workflows where labeled outcomes (like fire risk) are unavailable.

---

## 🚀 Features

- Input up to 12 environmental variables via sidebar
- Project data into PCA space using a trained model
- Visualize PCA coordinates and compare with historical data
- Upload CSVs for batch projection
- Optional clustering and anomaly detection modules
- Download PCA-transformed results

---

## 🛠️ Tech Stack

- Python 3.8+
- Streamlit
- scikit-learn
- pandas
- joblib

---

## 📁 Files Included

- `app.py` — Streamlit interface  
- `feature_selection.py` — Selects high-variance features  
- `save_selected_features.py` — Saves selected features and PCA model  
- `selected_features.pkl` — Saved feature list  
- `pca_model.pkl` — Trained PCA model  
- `climate_risk_dataset.csv` — Input dataset  
- `pca_transformed.csv` — Optional historical PCA data  
- `requirements.txt` — Python dependencies  

---

## ▶️ How to Run

```bash
pip install -r requirements.txt
streamlit run app.py
