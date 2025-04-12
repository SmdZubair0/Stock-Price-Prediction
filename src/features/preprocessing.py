import sys
import logging
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from my_logger import create_log_path, CustomLogger

warnings.filterwarnings('ignore')


log_path = create_log_path("Data_Preprocessing")
logger = CustomLogger("Data_preprocessing", log_path)


class FeaturesForStocksData:

    def __init__(self):
        self.Ticker = None
        self.data = None

    def load_data(self,
                  filepath : Path,
                  ticker : str) -> None:
        """
            Reads CSV file and save it
            
            parameters:
            - filepath : the path where the file is stored
            - ticker : The ticker of the company of which the stock is colected

            returns
            - None
        """
        
        self.Ticker = ticker
        self.data = pd.read_csv(filepath)
        logger.save_logs("Data loaded for preprocessing")

    def get_data(self) -> pd.DataFrame:
        """
            return stored dataframe

            parameters:
            - None

            returns:
            - None
        """
        return self.data
    
    def save_data(self,
                  filepath : Path) -> None:
        """
            save data to given file

            parameters:
            - filepath : where data is stored

            returns:
            None
        """
        self.data.to_csv(filepath, index=False)

        logger.save_logs(f"Data saved at {filepath}.")

    def create_Split_ratio(self) -> pd.DataFrame:
        """
            Create a dataframe with date and split_ratio columns for the ticker

            parameters:
            - None

            returns
            - splits : Dataframe containing date and split_ratio
        """

        stocks = yf.Ticker(self.Ticker)
        splits = stocks.splits
        splits = splits.reset_index()
        splits.columns = ['date', 'split_ratio']

        splits['date'] = splits["date"].dt.tz_localize(None)

        logger.save_logs("Split_ratio generated")

        return splits


    def create_Adj_close(self) -> None:
        """
            Adds following features to dataset
            - split_ratio
            - adj_factor
            - adj_close

            parameters:
            - None

            returns
            - None
        """

        splits = self.create_Split_ratio()
        self.data['date'] = pd.to_datetime(self.data['date'])

        df = self.data.merge(splits, on='date', how='left')

        df['split_ratio'].fillna(1, inplace=True)

        df['adj_factor'] = df['split_ratio'].cumprod()

        df['adj_close'] = df['adj_factor'] * df['close']

        self.data = df

        logger.save_logs("Adj_close added along with Adj_factor, split_ratio")


    def create_Moving_averages(self) -> None:
        """
            Adds following feature to dataset
            - rolling_ma

            parameters:
            - None

            returns
            - None
        """

        self.data['rolling_ma'] = self.data['adj_close'].ewm(span=10, adjust=False).mean()

        logger.save_logs("Rolling MA added")


    def create_RSI(self) -> None:
        """
            Adds following features to dataset
            - change
            - gain
            - loss
            - avg_gain
            - avg_loss
            - relative_strength
            - RSI

            parameters:
            - None

            returns
            - None
        """

        # curr close value - past close value
        self.data['change'] = self.data['adj_close'].diff()

        # take gain if gain, else take 0
        self.data['gain'] = np.where(self.data['change'] > 0, self.data['change'], 0)
        # take loss if loss, else take 0
        self.data['loss'] = np.where(self.data['change'] < 0, -self.data['change'], 0)

        # calculate average gain and loss usign exponentially moving average
        self.data['avg_gain'] = self.data['gain'].ewm(span=14, adjust=False).mean()
        self.data['avg_loss'] = self.data['loss'].ewm(span=14, adjust=False).mean()

        self.data['relative_strength'] = self.data['avg_gain'] / self.data['avg_loss']

        self.data['RSI'] = 100 - (100 / (1 + self.data["relative_strength"]))
        
        logger.save_logs("RSI added along with change, gain, loss, avg_gain, avg_loss, relative_strength")

    
    def create_MACD(self) -> None:
        """
            Adds following features to dataset
            - EMA_12
            - EMA_26
            - MACD
            - Signal_Line
            - MACD_histogram

            parameters:
            - None

            returns
            - None
        """
        

        self.data["EMA_12"] = self.data["adj_close"].ewm(span=12, adjust=False).mean()
        self.data["EMA_26"] = self.data["adj_close"].ewm(span=26, adjust=False).mean()

        # Calculate MACD Line
        self.data["MACD"] = self.data["EMA_12"] - self.data["EMA_26"]

        # Calculate Signal Line (9-day EMA of MACD)
        self.data["Signal_Line"] = self.data["MACD"].ewm(span=9, adjust=False).mean()

        # Calculate MACD Histogram
        self.data["MACD_Histogram"] = self.data["MACD"] - self.data["Signal_Line"]

        logger.save_logs("MACD added along with EMA_12, EMA_26, single_line, MACD_histogram")


    def create_Bollinger_bands(self) -> None:
        """
            Adds following features to dataset
            - SMA_20
            - STD
            - upper_band
            - lower_band

            parameters:
            - None

            returns
            - None
        """

        self.data["SMA_20"] = self.data["adj_close"].ewm(span=20, adjust=False).mean()

        # Standard Deviation
        self.data["STD"] = self.data["adj_close"].ewm(span=20, adjust=False).std()

        # Calculate Bollinger Bands
        self.data["upper_band"] = self.data["SMA_20"] + (2 * self.data["STD"])
        self.data["lower_band"] = self.data["SMA_20"] - (2 * self.data["STD"])

        logger.save_logs("Added upper_band, lower_band, STD, SMA_20")


    def create_date_features(self) -> None:
        """
            Adds following features to dataset
            - year
            - month
            - day
            - weekday

            parameters:
            - None

            returns
            - None
        """

        self.data["year"] = self.data["date"].dt.year
        self.data["month"] = self.data["date"].dt.month
        self.data["day"] = self.data["date"].dt.day
        self.data["weekday"] = self.data["date"].dt.weekday

        logger.save_logs("Added year, month, day, weekday")


    def create_Seasonality_features(self) -> None:
        """
            Adds following features to dataset
            - Fourier_Sin_7
            - Fourier_Cos_7
            - Fourier_Sin_30
            - Fourier_Cos_30

            parameters:
            - None

            returns
            - None
        """
    
        self.data["Fourier_Sin_7"] = np.sin(2 * np.pi * self.data.date.dt.day_of_year / 7)
        self.data["Fourier_Cos_7"] = np.cos(2 * np.pi * self.data.date.dt.day_of_year / 7)
        self.data["Fourier_Sin_30"] = np.sin(2 * np.pi * self.data.date.dt.day_of_year / 30)
        self.data["Fourier_Cos_30"] = np.cos(2 * np.pi * self.data.date.dt.day_of_year / 30)

        logger.save_logs("Added Fourier_Sin_7, Fourier_Cos_7, Fourier_Sin_30, Fourier_Cos_30")


    def drop_null_values(self) -> None:
        """
            removes all the records with null values

            parameters:
            - None
            

            returns:
            - None
        """

        self.data.dropna(inplace=True)

        logger.save_logs("Dropped NAs")

    def generate_all_features(self) -> None:
        """
            Generates all the features

            parameters:
            - None

            returns:
            - None
        """
        self.create_Adj_close()
        self.create_Moving_averages()
        self.create_RSI()
        self.create_MACD()
        self.create_Bollinger_bands()
        self.create_date_features()
        self.create_Seasonality_features()


class FeaturesForNewsData:

    def __init__(self):
        self.data = None

    def load_data(self,
                  filepath : Path) -> None:
        """
            Reads CSV file and save it
            
            parameters:
            - filepath : the path where the file is stored

            returns
            - None
        """

        self.data = pd.read_csv(filepath)
        logger.save_logs("Data loaded for preprocessing")

    def save_data(self,
                  filepath : Path) -> None:
        """
            save data to given file

            parameters:
            - filepath : where data is stored

            returns:
            None
        """
        self.data.to_csv(filepath, index=False)

        logger.save_logs(f"Data saved at {filepath}.")
    


if __name__ == "__main__":

    ticker = sys.argv[1]

    root_dir = Path(__file__).parent.parent.parent
    input_data_path = root_dir / "data" / "raw"
    output_data_path = root_dir / "data" / "interim"

    stocks_file = input_data_path / "stocks_1.csv"
    news_file = input_data_path / "news_1.csv"

    stocks_file2 = output_data_path / "stocks_2.csv"
    news_file2 = output_data_path / "news_2.csv"

    stockprocessor = FeaturesForStocksData()
    newsprocessor = FeaturesForNewsData()

    stockprocessor.load_data(stocks_file,
                             ticker=ticker)
    
    stockprocessor.generate_all_features()
    stockprocessor.save_data(stocks_file2)
    
    newsprocessor.load_data(news_file)
    newsprocessor.save_data(news_file2)  