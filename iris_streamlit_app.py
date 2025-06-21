# iris_streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Load dataset
data = pd.read_csv("Iris.csv")
data.drop("Id", axis=1, inplace=True)

# Encode target labels
label_encoder = LabelEncoder()
data["Species"] = label_encoder.fit_transform(data["Species"])

# Features and labels
X = data.drop("Species", axis=1)
y = data["Species"]

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# Streamlit UI
st.title("🌸 Iris Flower Species Predictor")
st.write("Enter flower measurements below to predict the species")

# Input fields
sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.1)
sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.5)
petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 1.4)
petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 0.2)

# Predict
if st.button("Predict Species"):
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(input_data)[0]
    species = label_encoder.inverse_transform([prediction])[0]
    st.success(f"Predicted Iris Species: **{species}**")
