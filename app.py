import streamlit as st
from src.model import prediction

# Set page title
st.title("Stock Price Predictor")

# Dropdown menu for stock symbols
stock_symbol = st.selectbox("Select a stock symbol:", ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA'])

# Window size input (optional)
window_size = st.number_input("Enter window size (e.g., for LSTM):", min_value=5, max_value=15, value=5)

# Predict button
if st.button("Predict"):
    # Placeholder for actual prediction logic
    # For now, just display selected values
    st.write(f"Predicting stock price for: **{stock_symbol}**")
    st.write(f"Using window size: **{window_size}**")
    
    # Dummy prediction result
    predicted_price = prediction.predict(window_size, stock_symbol)
    st.success(predicted_price[0][0])
