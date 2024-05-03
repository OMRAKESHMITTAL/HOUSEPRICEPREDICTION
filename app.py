import pickle
import streamlit as st
import pandas as pd

st.header("House Price Prediction")
best_model = pickle.load(open("best_model.pkl", 'rb'))

sqft_living = st.number_input("Enter square footage of living space:", value=0, step=1)
bedrooms = st.number_input("Enter number of bedrooms", value=0, step=1)
bathrooms = st.number_input("Enter number of bathrooms:", value=0, step=1)
floors = st.number_input("Enter number of floors:", value=0, step=1)
view = st.number_input("Enter view rating (0 to 5):",  min_value=0, max_value=5, step=1)
sqft_above = st.number_input("Enter square footage above ground:", value=0, step=1)
sqft_basement = st.number_input("Enter square footage of basement:", value=0, step=1)
yr_built = st.number_input("Enter the year the house was built:", value=0, step=1)
yr_renovated = st.number_input("Enter the year the house was renovated (if applicable):", value=0, step=1)

input_data = pd.DataFrame ({
    'sqft_living': [sqft_living],
    'bedrooms': [bedrooms],
    'bathrooms': [bathrooms],
    'floors': [floors],
    'view': [view],
    'sqft_above': [sqft_above],
    'sqft_basement': [sqft_basement],
    'yr_built': [yr_built],
    'yr_renovated': [yr_renovated]
})


if(st.button("Predict")):
    prediction = best_model.predict(input_data)
    st.text(f"The model predicts that the rate of the house should be $:{prediction}")

