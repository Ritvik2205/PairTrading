from AlgorithmImports import *
from itertools import combinations
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from hmmlearn import hmm

class PairTradingWithLSTMHMM(QCAlgorithm):
    def Initialize(self):
        self.SetStartDate(2020, 1, 1)  # Set the backtest start date
        self.SetEndDate(2025, 1, 1)    # Set the backtest end date
        self.SetCash(100000)           # Set starting cash
        
        # Define sector and fetch tickers (replace with your tickers)
        self.financial_tickers = ['AFL', 'ALL', 'AXP', 'AIG', 'AMP', 'AON', 'ACGL', 'AJG', 'AIZ', 'BAC', 'BRK.B', 'BLK', 'BX', 'BK', 'BRO', 'COF', 'CBOE', 'SCHW', 'CB', 'CINF', 'C']          

        self.symbols = [self.AddEquity(ticker, Resolution.Daily).Symbol for ticker in self.financial_tickers]
        
        # Initialize data containers
        self.min_dict = {}
        self.max_dict = {}
        self.normclose = {}
        self.diff_df = {}
        self.sq_sum_diff = {}
        self.least_ssd_pairs = []
        self.pair_positions = {}  # To track open positions for each pair

        # Parameters for LSTM
        self.lookback = 60  # Number of days to look back for LSTM prediction
        self.epochs = 10    # Number of training epochs for LSTM
        self.batch_size = 32

        # Parameters for HMM
        self.n_components = 3  # Number of hidden states for HMM

        
    
    def OnData(self, data):
        # Ensure sufficient data exists for calculations
        # if not self.History(self.symbols, 200).empty:
        self.CalculatePairTradingSignals()
    
    def CalculatePairTradingSignals(self):
        # Fetch historical data
        history = self.History(self.symbols, 200, Resolution.Daily)
        close = history.unstack(level=0)['close']  # Pivot to get close prices per ticker
        close = close.dropna()
        
        # Store min and max values for normalization
        for symbol in self.symbols:
            ticker = str(symbol)
            self.min_dict[ticker] = close[ticker].min()
            self.max_dict[ticker] = close[ticker].max()
            self.normclose[ticker] = (close[ticker] - self.min_dict[ticker]) / (self.max_dict[ticker] - self.min_dict[ticker])
        
        # Calculate pair differences and sum of squared differences
        norm_df = pd.DataFrame(self.normclose)
        for pair in combinations(self.financial_tickers, 2):
            pair_str = f"{pair[0]} - {pair[1]}"
            self.diff_df[pair_str] = norm_df[pair[1]] - norm_df[pair[0]]
            self.sq_sum_diff[pair_str] = (self.diff_df[pair_str] ** 2).sum()
        
        # Select pairs with the least sum of squared differences
        self.sq_sum_diff = dict(sorted(self.sq_sum_diff.items(), key=lambda x: x[1]))
        self.least_ssd_pairs = list(self.sq_sum_diff.keys())[:2]
        
        # Train LSTM and HMM for the selected pairs
        self.TrainLSTMAndHMM(close)
        
        # Generate signals for the selected pairs
        self.GenerateTradingSignals(close)
    
    def TrainLSTMAndHMM(self, close):
        # Train LSTM for each selected pair
        self.lstm_models = {}
        for pair in self.least_ssd_pairs:
            stock1, stock2 = pair.split(" - ")
            prices = close[[stock1, stock2]].values
            self.lstm_models[pair] = self.TrainLSTM(prices)
        
        # Train HMM on the returns of the selected pairs
        returns = close.pct_change().dropna()
        self.hmm_model = self.TrainHMM(returns)

    def TrainLSTM(self, prices):
        # Preprocess data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_prices = scaler.fit_transform(prices)

        # Prepare training data
        X, y = [], []
        for i in range(self.lookback, len(scaled_prices)):
            X.append(scaled_prices[i-self.lookback:i, :])
            y.append(scaled_prices[i, 0])  # Predict the first stock's price
        X, y = np.array(X), np.array(y)

        # Build LSTM model
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
        model.add(LSTM(units=50, return_sequences=False))
        model.add(Dense(units=25))
        model.add(Dense(units=1))

        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X, y, epochs=self.epochs, batch_size=self.batch_size)

        return model

    def TrainHMM(self, returns):
        # Reshape returns for HMM
        returns = returns.values.reshape(-1, 1)

        # Train the HMM
        model = hmm.GaussianHMM(n_components=self.n_components, covariance_type="diag", n_iter=1000)
        model.fit(returns)

        return model

    def GenerateTradingSignals(self, close):  # Added close parameter
        for pair in self.least_ssd_pairs:
            stock1, stock2 = pair.split(" - ")
            spread = self.diff_df[pair]
            
            mean = spread.mean()
            std = spread.std()
            upper_bound = mean + 2 * std
            lower_bound = mean - 2 * std
            
            # Get current spread value
            current_spread = spread.iloc[-1]
            
            # Fetch recent data for the pair
            recent_data = close[[stock1, stock2]].values[-self.lookback:]
            if recent_data.shape[0] < self.lookback or recent_data.shape[1] != 2:
                self.Debug(f"Insufficient data for {pair}: {recent_data.shape}")
                continue
            
            # Normalize recent data
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(recent_data)
            
            # Reshape for LSTM input (1, lookback, num_features)
            X = scaled_data.reshape(1, self.lookback, 2)
            
            # Predict next price using LSTM
            predicted_price_scaled = self.lstm_models[pair].predict(X)
            predicted_price = scaler.inverse_transform(
                np.hstack((predicted_price_scaled, np.zeros((1, 1))))  # Add dummy column
            )[0, 0]
            
            # Predict the most likely hidden state sequence using HMM
            recent_returns = pd.DataFrame(recent_data).pct_change().dropna().values.reshape(-1, 1)
            hidden_states = self.hmm_model.predict(recent_returns)
            current_state = hidden_states[-1]
            
            # Signal conditions
            if current_spread > upper_bound and (current_state == 0 or predicted_price < current_spread):  # Bullish regime and spread is wide
                self.GoShort(pair, stock1, stock2)
            elif current_spread < lower_bound and (current_state == 1 or predicted_price > current_spread):  # Bearish regime and spread is narrow
                self.GoLong(pair, stock1, stock2)
            elif mean - std <= current_spread <= mean + std:
                self.ClosePosition(pair, stock1, stock2)

    def GoShort(self, pair, stock1, stock2):
        if pair not in self.pair_positions or self.pair_positions[pair] != "short":
            self.Debug(f"Opening SHORT position for pair: {pair}")
            self.SetHoldings(stock1, -0.5)  # Short stock1
            self.SetHoldings(stock2, 0.5)   # Long stock2
            self.pair_positions[pair] = "short"

    def GoLong(self, pair, stock1, stock2):
        if pair not in self.pair_positions or self.pair_positions[pair] != "long":
            self.Debug(f"Opening LONG position for pair: {pair}")
            self.SetHoldings(stock1, 0.5)   # Long stock1
            self.SetHoldings(stock2, -0.5)  # Short stock2
            self.pair_positions[pair] = "long"

    def ClosePosition(self, pair, stock1, stock2):
        if pair in self.pair_positions:
            self.Debug(f"Closing position for pair: {pair}")
            self.Liquidate(stock1)
            self.Liquidate(stock2)
            del self.pair_positions[pair]

    def OnEndOfAlgorithm(self):
        # Log final information at the end of the backtest
        self.Debug(f"Final least SSD pairs: {self.least_ssd_pairs}")
