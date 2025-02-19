import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from sklearn.svm import SVR
from xgboost import XGBRegressor

st.cache_data.clear()

# Load and preprocess the data
file_path = "C:/Users/Ahmed/Desktop/COM724/Clustering.csv" 
price_data = pd.read_csv(file_path)

if "Cluster" in price_data.columns:
    price_data = price_data.drop(columns=["Cluster"])

if "Ticker" in price_data.columns:
    price_data = price_data.set_index("Ticker")

price_data.columns = pd.to_datetime(price_data.columns, errors='coerce')

# Representatives for analysis
representatives = ["SOL-USD", "BTC-USD", "ETH-USD", "XMR-USD"]


# Create tabs for navigation
tab1, tab2, tab3, tab4 = st.tabs(["Exploratory Data Analysis (EDA)", "Forecasting & Trading Signals", "News Feed", "Correlation Analysis"])

### TAB 1: EDA ###
with tab1:
    st.header("Exploratory Data Analysis (EDA)")
    selected_rep = st.selectbox("Select a Cryptocurrency for EDA", representatives)
    
    if selected_rep:
        # Historical Price Trend
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(price_data.columns, price_data.loc[selected_rep], linewidth=2, label=f"Price Trend for {selected_rep}")
        ax.set_title(f"Historical Price Trend: {selected_rep}")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.legend()
        plt.xticks(rotation=45)
        st.pyplot(fig)
        
        # Price Distribution
        st.write("### Price Distribution")
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.histplot(price_data.loc[selected_rep], bins=50, kde=True, ax=ax)
        ax.set_title(f"Price Distribution: {selected_rep}")
        ax.set_xlabel("Price")
        ax.set_ylabel("Frequency")
        st.pyplot(fig)

        # Daily Percentage Returns
        st.write("### Daily Percentage Returns")
        price_returns = price_data.pct_change()
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(price_returns.columns, price_returns.loc[selected_rep], label=f"Daily Returns for {selected_rep}")
        ax.set_title(f"Daily Percentage Returns: {selected_rep}")
        ax.set_xlabel("Date")
        ax.set_ylabel("Daily Returns")
        ax.legend()
        st.pyplot(fig)
        
        # Distribution of Daily Returns
        st.write("### Distribution of Daily Returns")
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.histplot(price_returns.loc[selected_rep].dropna(), bins=50, kde=True, ax=ax)
        ax.set_title(f"Distribution of Daily Returns: {selected_rep}")
        ax.set_xlabel("Daily Return (%)")
        ax.set_ylabel("Frequency")
        st.pyplot(fig)

### TAB 2: Forecasting & Trading Signals ###
with tab2:
    st.header("Forecasting & Trading Signals")
    selected_rep = st.selectbox("Select a Cryptocurrency for Forecasting", representatives, key="forecast_rep")
    selected_model = st.selectbox("Choose a Model", ["Prophet", "ARIMA", "Holt-Winters", "SVR", "XGBoost"])

    # Ensure the selected representative exists in the dataset
    if selected_rep:
        # Extract historical prices
        prices = price_data.loc[selected_rep].dropna()
        dates = pd.to_datetime(prices.index, errors="coerce")
        prices = prices.reset_index(drop=True)
        
        
         # Limit the forecast horizon
        forecast_horizon = 30
        last_date = dates[-1] if not dates.empty else pd.Timestamp.now()
        future_dates = pd.date_range(start=last_date, periods=forecast_horizon + 1, freq="D")[1:]

        # Initialize placeholders for forecast, moving averages, and trading signals
        forecast_prices = None
        short_ma = prices.rolling(window=5).mean()  # Short-term moving average (5 days)
        long_ma = prices.rolling(window=20).mean()  # Long-term moving average (20 days)
        buy_signals = []
        sell_signals = []

        # Prophet Model
        if selected_model == "Prophet":
            df = pd.DataFrame({"ds": dates, "y": prices})
            model = Prophet()
            model.fit(df)
            future = model.make_future_dataframe(periods=forecast_horizon)
            forecast = model.predict(future)

            # Extract predicted values
            forecast_prices = forecast["yhat"].iloc[-forecast_horizon:].values
            forecast_dates = forecast["ds"].iloc[-forecast_horizon:].values

        # ARIMA Model
        elif selected_model == "ARIMA":
            model = ARIMA(prices, order=(1, 1, 1))
            fitted = model.fit()
            forecast_prices = fitted.forecast(steps=forecast_horizon).values
            forecast_dates = future_dates

        # Holt-Winters Model
        elif selected_model == "Holt-Winters":
            model = ExponentialSmoothing(prices, trend="add", seasonal="add", seasonal_periods=12)
            fitted = model.fit()
            forecast_prices = fitted.forecast(forecast_horizon)
            forecast_dates = future_dates

        # Support Vector Regression (SVR)
        elif selected_model == "SVR":
            svr = SVR(kernel="rbf", C=1e3, gamma=0.1)
            svr.fit(np.arange(len(prices)).reshape(-1, 1), prices)
            forecast_prices = svr.predict(np.arange(len(prices), len(prices) + forecast_horizon).reshape(-1, 1))
            forecast_dates = future_dates

        # XGBoost Model
        elif selected_model == "XGBoost":
            xgb = XGBRegressor()
            xgb.fit(np.arange(len(prices)).reshape(-1, 1), prices)
            forecast_prices = xgb.predict(np.arange(len(prices), len(prices) + forecast_horizon).reshape(-1, 1))
            forecast_dates = future_dates

       
        # Generate buy/sell signals based on forecast prices
        forecast_prices = list(forecast_prices)
        forecast_dates = list(future_dates)
        buy_signals = []
        sell_signals = []
        for i in range(1, len(forecast_prices)):
            if forecast_prices[i] > forecast_prices[i - 1]:  # Buy signal
                buy_signals.append((forecast_dates[i], forecast_prices[i]))
            elif forecast_prices[i] < forecast_prices[i - 1]:  # Sell signal
                sell_signals.append((forecast_dates[i], forecast_prices[i]))

        # Predictive high and low for the next 30 days
        predicted_high = max(forecast_prices)
        predicted_low = min(forecast_prices)

        # Display confidence level (mocked for demonstration)
        confidence_level = np.random.uniform(85, 95)

        # Visualization with Plotly
        fig = go.Figure()

        # Plot historical prices
        fig.add_trace(go.Scatter(
            x=dates, y=prices, mode="lines", name="Historical Prices", line=dict(color="blue")
        ))

        # Plot forecasted prices
        fig.add_trace(go.Scatter(
            x=forecast_dates, y=forecast_prices, mode="lines", name="Forecast Prices", line=dict(dash="dot", color="green")
        ))

        # Plot moving averages
        fig.add_trace(go.Scatter(
            x=dates, y=short_ma, mode="lines", name="Short-term MA", line=dict(color="orange")
        ))
        fig.add_trace(go.Scatter(
            x=dates, y=long_ma, mode="lines", name="Long-term MA", line=dict(color="purple")
        ))

        # Plot buy signals
        if buy_signals:
            buy_dates, buy_prices = zip(*buy_signals)
            fig.add_trace(go.Scatter(
                x=buy_dates, y=buy_prices, mode="markers", name="Buy Signals", marker=dict(color="green", size=10)
            ))

        # Plot sell signals
        if sell_signals:
            sell_dates, sell_prices = zip(*sell_signals)
            fig.add_trace(go.Scatter(
                x=sell_dates, y=sell_prices, mode="markers", name="Sell Signals", marker=dict(color="red", size=10)
            ))

        # Update layout
        fig.update_layout(
            title=f"Forecasting & Trading Signals for {selected_rep} ({selected_model})",
            xaxis_title="Date",
            yaxis_title="Price",
            legend_title="Legend",
            template="plotly_dark"
        )

        # Display the Plotly chart
        st.plotly_chart(fig, use_container_width=True)

        # Display additional analysis
        st.write(f"**Predicted High:** ${predicted_high:.2f}")
        st.write(f"**Predicted Low:** ${predicted_low:.2f}")
        st.write(f"**Confidence Level:** {confidence_level:.2f}%")

### TAB 3: News Feed ###
import requests

with tab3:
    st.header("News Feed")
    st.write("### Latest Cryptocurrency News")

    api_key = "7ed907df58294d8e8fd6d42f45d213be"
    base_url = "https://newsapi.org/v2/everything?q=cryptocurrency&language=en&sortBy=publishedAt&apiKey=7ed907df58294d8e8fd6d42f45d213be"

    # Query parameters
    query_params = {
        "q": "cryptocurrency OR bitcoin OR ethereum",  # Keywords
        "language": "en",  # Language of the news
        "sortBy": "publishedAt",  # Sort by most recent
        "7ed907df58294d8e8fd6d42f45d213be": api_key,  # Your API key
    }

    # Fetch data from the News API
    try:
        response = requests.get(base_url, params=query_params)
        response.raise_for_status()  # Raise HTTPError for bad responses
        news_data = response.json()

        # Parse and display the news
        if "articles" in news_data:
            for article in news_data["articles"][:10]: 
                st.write(f"**{article['title']}**")
                st.write(f"Published at: {article['publishedAt']}")
                st.write(f"[Read more]({article['url']})")
                st.write("---")
        else:
            st.write("No news articles found.")
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching news: {e}")


### TAB 4: Correlation Analysis ###
with tab4:
    st.header("Correlation Analysis")

    file_path = "C:/Users/Ahmed/Desktop/COM724/Clustering.csv" 
    price_data = pd.read_csv(file_path)

    if "Cluster" in price_data.columns:
        price_data = price_data.drop(columns=["Cluster"])

    if "Ticker" in price_data.columns:
        price_data = price_data.set_index("Ticker")

    price_data_t = price_data.transpose()

    correlation_matrix = price_data_t.corr()

    representatives = ["SOL-USD", "BTC-USD", "ETH-USD", "XMR-USD"]
    valid_representatives = [rep for rep in representatives if rep in correlation_matrix.columns]

    selected_rep = st.selectbox("Select a Cryptocurrency Representative", valid_representatives)

    st.write("#### Cryptocurrency Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(correlation_matrix, cmap="coolwarm", annot=False, ax=ax)
    ax.set_title("Cryptocurrency Correlation Heatmap")
    st.pyplot(fig)

    if selected_rep:
        st.write(f"#### Top Correlations for {selected_rep}")
        
        correlations = correlation_matrix[selected_rep]
        top_positive = correlations.sort_values(ascending=False).iloc[1:5]  # Skip self-correlation
        top_negative = correlations.sort_values(ascending=True).iloc[:4]

        col1, col2 = st.columns(2)
        with col1:
            st.write("**Top Positive Correlations**")
            st.dataframe(top_positive)
        with col2:
            st.write("**Top Negative Correlations**")
            st.dataframe(top_negative)
