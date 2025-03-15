import streamlit as st
import numpy as np
import yfinance as yf
import plotly.express as px
from stocknews import StockNews
from textblob import TextBlob
from prophet import Prophet

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
        data['% Change'] = data['Close'].pct_change() * 100
        tabs = st.tabs(["Pricing Data", "Fundamental Data", "Stock News", "Stock Prediction"])
        
        with tabs[0]:
            st.subheader(f"Stock Price for {ticker}")
            fig = px.line(data, x=data.index, y='Close', title=f'{ticker} Stock Price Movement')
            st.plotly_chart(fig)

            st.subheader('Raw data')
            st.write(data)
            
            st.subheader("Stock Statistics")
            annual_return = data['% Change'].mean() * 252
            std_dev = np.std(data['% Change']) * np.sqrt(252)
            risk_adjusted_return = annual_return / std_dev if std_dev != 0 else 0
            
            st.write(f"**Annual Return:** {annual_return:.2f}%")
            st.write(f"**Standard Deviation:** {std_dev:.2f}%")
            st.write(f"**Risk-Adjusted Return:** {risk_adjusted_return:.2f}")
        
        with tabs[1]:
            st.subheader("Fundamental Data")
            info = stock.info

            currency = info.get('financialCurrency', 'N/A')
            
            def format_large_number(value):
                if value is None:
                    return "N/A"
                elif value >= 1e12:
                    return f"{value / 1e12:.2f} Trillion {currency}"
                elif value >= 1e9:
                    return f"{value / 1e9:.2f} Billion {currency}"
                elif value >= 1e6:
                    return f"{value / 1e6:.2f} Million {currency}"
                else:
                    return f"{value:,} {currency}"
            
            st.write(f"**Sector:** {info.get('sector', 'N/A')}")
            st.write(f"**Industry:** {info.get('industry', 'N/A')}")
            st.write(f"**Market Cap:** {format_large_number(info.get('marketCap'))}")
            st.write(f"**Revenue:** {format_large_number(info.get('totalRevenue'))}")
            st.write(f"**Net Income:** {format_large_number(info.get('netIncomeToCommon'))}")
            st.write(f"**Dividend Yield:** {info.get('dividendYield', 'N/A')}")
            
            
            st.subheader("Financial Statements")
            st.write("**Balance Sheet:**")
            st.write(stock.balance_sheet)
            
            st.write("**Income Statement:**")
            st.write(stock.financials)
            
            st.write("**Cash Flow Statement:**")
            st.write(stock.cashflow)
        
        with tabs[2]:
            st.subheader("Stock News")
            news = stock.news
            
            if news:
                for article in news:
                    content = article.get('content', {})
                    title = content.get('title', 'No Title Available')
                    publisher = content.get('provider', {}).get('displayName', 'Unknown Publisher')
                    link = content.get('canonicalUrl', {}).get('url', '#')
                    publish_time = content.get('pubDate', 'N/A')
                    sentiment = TextBlob(title).sentiment.polarity
                    
                    if sentiment > 0:
                        sentiment_label = "Positive"
                    elif sentiment < 0:
                        sentiment_label = "Negative"
                    else:
                        sentiment_label = "Neutral"
                    
                    st.write(f"### {title}")
                    st.write(f"{publisher} - Published: {publish_time}")
                    st.write(f"**Sentiment:** {sentiment_label}")
                    st.write(f"[Read more]({link})")
                    st.write("---")
            else:
                st.write("No news available for this stock.")

        with tabs[3]:
            st.subheader("Stock Price Prediction")
            forecast_days = st.slider("Select Forecast Period (days)", 30, 365, 180)
            
            df_train = data[['Close']].reset_index()
            df_train.rename(columns={'Date': 'ds', 'Close': 'y'}, inplace=True)
            df_train['ds'] = df_train['ds'].dt.tz_localize(None)  # Removing timezone
            
            model = Prophet()
            model.fit(df_train)
            
            future = model.make_future_dataframe(periods=forecast_days)
            forecast = model.predict(future)
            
            fig_pred = px.line(forecast, x='ds', y='yhat', title=f'{ticker} Stock Price Prediction')
            st.plotly_chart(fig_pred)
            
            st.write("Prediction Components")
            fig_components = model.plot_components(forecast)
            st.pyplot(fig_components)
