import streamlit as st
import yfinance as yf
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from textblob import TextBlob
from prophet import Prophet
import pandas as pd

st.set_page_config(page_title="StockSphere", layout="wide")
left_co, cent_co, last_co = st.columns(3)
with cent_co:
    st.image('src/logo.png', use_container_width=True)
st.markdown(
    """
    <div style="text-align: center;">
        <h2>Welcome to StockSphere</h2>
        <p style="font-size: 18px;">Your all-in-one platform for real-time stock insights, analytics, and predictions.</p>
    </div>
    """,
    unsafe_allow_html=True
)
st.sidebar.header("Stock Selection")
ticker = st.sidebar.text_input("Enter Stock Ticker")
period = st.sidebar.selectbox("Select Time Period", ['1mo', '3mo', '6mo', '1y', '2y', '5y', 'max'])

if ticker:
    with st.spinner("Fetching stock data..."):
        stock = yf.Ticker(ticker)
        data = stock.history(period=period)

    if data.empty:
        st.error("No data found for the given ticker.")
    else:
        data['% Change'] = data['Close'].pct_change() * 100

        st.markdown("###")
        tabs = st.tabs(["📊 Pricing Data", "📈 Fundamental Data", "📰 Stock News", "🔮 Stock Prediction"])

        with tabs[0]:
            st.subheader(f"Stock Price for {ticker}")

            fig = go.Figure(data=[go.Candlestick(x=data.index,
                                        open=data['Open'],
                                        high=data['High'],
                                        low=data['Low'],
                                        close=data['Close'])])
            fig.update_layout(
                title=f'{ticker} Stock Price Movement',
                xaxis_title='Date',
                yaxis_title='Price',
                xaxis=dict(
                   rangeslider=dict(
                       visible=False  # Remove range slider
                    )
                ),
                dragmode='select',  # Enable rectangular selection zoom
                plot_bgcolor='rgba(0,0,0,0)',  # Make plot background transparent
                paper_bgcolor='rgba(0,0,0,0)', # Make paper background transparent
                font=dict(color="#FFFFFF"), # Set font color to white
                selectdirection='horizontal', # Only allow horizontal select
                # Add zoom behavior
                xaxis_fixedrange=False,
                yaxis_fixedrange=False
            )
            
            fig.update_xaxes(gridcolor='lightgrey')
            fig.update_yaxes(gridcolor='lightgrey')

            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Stock Statistics")
            col1, col2, col3 = st.columns(3)
            annual_return = data['% Change'].mean() * 252
            std_dev = np.std(data['% Change']) * np.sqrt(252)
            risk_adjusted_return = annual_return / std_dev if std_dev != 0 else 0

            col1.metric("Annual Return", f"{annual_return:.2f}%")
            col2.metric("Standard Deviation", f"{std_dev:.2f}%")
            col3.metric("Risk-Adjusted Return", f"{risk_adjusted_return:.2f}")

            st.write("### Raw Data")
            st.write(data)

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

            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Sector:** {info.get('sector', 'N/A')}")
                st.write(f"**Industry:** {info.get('industry', 'N/A')}")
                st.write(f"**Market Cap:** {format_large_number(info.get('marketCap'))}")
            with col2:
                st.write(f"**Revenue:** {format_large_number(info.get('totalRevenue'))}")
                st.write(f"**Net Income:** {format_large_number(info.get('netIncomeToCommon'))}")
                st.write(f"**Dividend Yield:** {info.get('dividendYield', 'N/A')}")

            st.subheader("Financial Statements")
            st.write("### Balance Sheet")
            st.dataframe(stock.balance_sheet)

            st.write("### Income Statement")
            st.dataframe(stock.financials)

            st.write("### Cash Flow Statement")
            st.dataframe(stock.cashflow)

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
                        sentiment_label = "🟢 Positive"
                    elif sentiment < 0:
                        sentiment_label = "🔴 Negative"
                    else:
                        sentiment_label = "🟡 Neutral"

                    st.markdown(f"### [{title}]({link})")
                    st.write(f"{publisher} - Published: {publish_time}")
                    st.write(f"**Sentiment:** {sentiment_label}")
                    st.write("---")
            else:
                st.write("No news available for this stock.")

        with tabs[3]:
            st.subheader("Stock Price Prediction")
            st.write("Here, we use the Prophet model to predict future stock prices based on historical data.")
            st.write("Because weekly and daily seasonal patterns are often unreliable for stock predictions, we have disabled them. You can adjust the strength of the *yearly* seasonal pattern.")

            seasonality_strength = st.slider("Yearly Seasonality Strength", 0.01, 0.5, 0.1)
            forecast_days = st.slider("Forecast Horizon (days)", 30, 365, 180)

            df_train = data[['Close']].reset_index()
            df_train.rename(columns={'Date': 'ds', 'Close': 'y'}, inplace=True)
            df_train['ds'] = df_train['ds'].dt.tz_localize(None)

            model = Prophet(seasonality_prior_scale=seasonality_strength,
                            weekly_seasonality=False,
                            daily_seasonality=False)
            model.fit(df_train)

            future = model.make_future_dataframe(periods=forecast_days)
            forecast = model.predict(future)

            fig_pred = px.line(forecast, x='ds', y='yhat', title=f'{ticker} Stock Price Prediction',
                               labels={'ds': 'Date', 'yhat': 'Predicted Price'},
                               color_discrete_sequence=['#FF5733'])
            fig_pred.add_trace(go.Scatter(x=df_train['ds'], y=df_train['y'], mode='lines',
                                           name='Historical Data'))
            st.plotly_chart(fig_pred)

            st.write("The orange/red line represents the predicted stock price, while the blue line presents the historical data.")
            st.write("---")
            st.write("The model attempts to capture longer-term trends and any yearly patterns in the stock price.  The slider above controls how strongly the yearly pattern influences the forecast.")
            st.write("Please note that these predictions are based on historical data and may not be accurate. Use this information for informational purposes only and not as financial advice.")