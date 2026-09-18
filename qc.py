from AlgorithmImports import *
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint

class PairsTradingAlgorithm(QCAlgorithm):
    def Initialize(self):
        # Set start and end dates
        self.SetStartDate(2020, 1, 1)
        self.SetEndDate(2025, 1, 1)
        self.SetCash(100000)  # Set initial cash

        # Parameters
        self.formation_period = 252  # 12 months formation period
        self.trading_period = 126    # 6 months trading period
        self.num_pairs = 20          # Number of pairs to trade
        self.threshold = 2           # Threshold for opening trades (in standard deviations)
        
        # List of stock symbols to trade
        self.symbols = [self.AddEquity(ticker).Symbol for ticker in ["AAPL", "MSFT", "AMZN", "TSLA", "NVDA", "INTC", "CSCO", "ORCL"]]
        
        # Data structures to store pairs and spread statistics
        self.pairs = []              # List to store selected pairs
        self.spread_mean = {}        # Mean of the spread for each pair
        self.spread_std = {}         # Standard deviation of the spread for each pair

        # Schedule rebalancing every 6 months
        self.Schedule.On(self.DateRules.MonthStart(self.symbols[0]), 
                        self.TimeRules.AfterMarketOpen(self.symbols[0], 10), 
                        self.Rebalance)

    def Rebalance(self):
        # Rebalance every 6 months
        if self.Time.month % 6 != 0:
            return

        # Fetch historical data for the formation period
        history = self.History(self.symbols, self.formation_period, Resolution.Daily)
        prices = history['close'].unstack(level=0)
        normalized_prices = prices / prices.iloc[0]  # Normalize prices

        # Step 1: Distance Method - Calculate sum of squared differences (SSD)
        ssd = {}
        for i in range(len(self.symbols)):
            for j in range(i + 1, len(self.symbols)):
                ssd[(self.symbols[i], self.symbols[j])] = np.sum((normalized_prices[self.symbols[i]] - normalized_prices[self.symbols[j]]) ** 2)

        # Step 2: Cointegration Method - Test for cointegration
        cointegrated_pairs = []
        for pair in ssd:
            score, pvalue, _ = coint(normalized_prices[pair[0]], normalized_prices[pair[1]])
            if pvalue < 0.05:  # If cointegrated
                cointegrated_pairs.append(pair)

        # Step 3: Select top N pairs based on SSD and cointegration
        self.pairs = sorted(cointegrated_pairs, key=ssd.get)[:self.num_pairs]

        # Step 4: Calculate spread mean and std for each pair
        for pair in self.pairs:
            spread = normalized_prices[pair[0]] - normalized_prices[pair[1]]
            self.spread_mean[pair] = np.mean(spread)
            self.spread_std[pair] = np.std(spread)

    def OnData(self, data):
        # Monitor spreads and execute trades
        for pair in self.pairs:
            if pair[0] not in data or pair[1] not in data:
                continue

            # Calculate current spread
            spread = data[pair[0]].Price - data[pair[1]].Price
            normalized_spread = (spread - self.spread_mean[pair]) / self.spread_std[pair]

            # Step 5: Execute trades based on spread divergence
            if normalized_spread > self.threshold:
                # Short the overvalued stock and long the undervalued stock
                self.SetHoldings(pair[0], -1)  # Short
                self.SetHoldings(pair[1], 1)   # Long
            elif normalized_spread < -self.threshold:
                # Long the undervalued stock and short the overvalued stock
                self.SetHoldings(pair[0], 1)   # Long
                self.SetHoldings(pair[1], -1)  # Short
            elif abs(normalized_spread) < 0.5:  # Close positions when spread converges
                self.Liquidate(pair[0])
                self.Liquidate(pair[1])
