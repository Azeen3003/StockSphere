# StockSphere

StockSphere is a comprehensive stock dashboard providing real-time stock insights, analytics, and predictions. Built using Streamlit, it offers an interactive interface for users to explore stock prices, fundamental data, news sentiment analysis, and future price predictions.

## Features
- **Real-time stock data** using Yahoo Finance API
- **Interactive charts** for stock price movements
- **Fundamental data** including market cap, revenue, and industry details
- **News sentiment analysis** to gauge market sentiment
- **Stock price predictions** powered by Prophet
- **User-friendly interface** with an intuitive layout

## Deployment
StockSphere is live at: [StockSphere Web App](https://stocksphere-az.streamlit.app/)

## Installation
To run this project locally, follow these steps:

```bash
# Clone the repository
git clone https://github.com/Azeen3003/StockSphere.git
cd StockSphere

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run main.py
```

## Requirements
Ensure you have the following dependencies installed:

```txt
streamlit
yfinance
numpy
pandas
plotly
textblob
prophet
```

## Usage
1. Enter a stock ticker symbol in the sidebar.
2. Select the desired time period.
3. View interactive stock charts and fundamental data.
4. Explore sentiment analysis of stock-related news.
5. Forecast stock prices using AI-powered predictions.

## Contributing
Contributions are welcome! Feel free to open issues or submit pull requests.

## License
This project is open-source and available under the MIT License.
