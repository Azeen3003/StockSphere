import streamlit as st
import yfinance as yf
import plotly.express as px

# Streamlit UI setup
st.title("Welcome to Your Stock Dashboard")

# User input for stock ticker
ticker = st.text_input("Enter Stock Ticker (e.g., AAPL, TSLA, MSFT)", "AAPL").upper()

# Period selection (Checkbox or Slider)
period_options = {"1 Month": "1mo", "3 Months": "3mo", "6 Months": "6mo", "1 Year": "1y", "2 Years": "2y", "5 Years": "5y", "Max": "max"}
period = st.selectbox("Select Time Period", list(period_options.keys()))

# Fetch data from Yahoo Finance
if ticker:
    stock = yf.Ticker(ticker)
    data = stock.history(period=period_options[period])
    
    if data.empty:
        st.error("No data found for the given ticker. Please enter a valid stock symbol.")
    else:
        st.subheader(f"Stock Price Data for {ticker}")
        
        # Plot price movement
        fig = px.line(data, x=data.index, y='Close', title=f"{ticker} Stock Price Movement")
        st.plotly_chart(fig)

        st.text('Raw data')
        st.write(data)
