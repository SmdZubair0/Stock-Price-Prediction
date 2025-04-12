import os
import sys
import json
import logging
import datetime
import requests
import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path
from yaml import safe_load
from bs4 import BeautifulSoup

sys.path.append(str(Path(__file__).resolve().parent.parent))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from api_key import API_KEY
from my_logger import create_log_path, CustomLogger

logger_path = create_log_path("make_dataset")
logger = CustomLogger("make_dataset", logger_path)
logger.set_log_level(logging.INFO)

COMPANY = {
    'AAPL' : 'apple',
    'TSLA' : 'tesla'
}

class StockDataCollector:
    
    def __init__(self):
        self.data = None

    
    def load_data(self,
                  ticker : str,
                  start_date : str,
                  end_date : str,
                  filepath : Path) -> None:
        
        """
            loads stocks data from yfinance api
            
            parameters:
            - ticker : company of which data is loaded
            - start_date : date from which data is loaded
            - end_date : date upto which data is loaded
            - filepath : file where the loaded data is stored

            returns
            - None
        """

        self.data = yf.download(tickers = [ticker], start=start_date, end=end_date)

        self.format_data()

        self.data.to_csv(filepath, index=False)

        logger.save_logs(f"stocks data stored in {filepath}")



    def format_data(self) -> None:
        """
            Format the dataset into required format

            parameters:
            - None

            returns:
            - None
        """
        self.data = self.data.reset_index()
        self.data = self.data.droplevel(level=0,axis=1)
        cols = ['date','close','high','low','open','volume']
        self.data.columns = cols

        logger.save_logs("stocks data formatted")

    def get_dateset(self) -> pd.DataFrame:
        """
            Returns dataset

            parameters:
            - None

            returns:
            - dataframe
        """
        return self.data



class NewsDataCollector:

    def __init__(self):
        self.data = None

    def load_data(self,
                    tikcer : str,
                    start_date : str,
                    end_date :str,
                    filepath : Path) -> None:
        """
            loads dataset
            
            parameters:
            - ticker : company of which data is loaded
            - start_date : date from which data is loaded
            - end_date : date upto which data is loaded
            - filepath : file where the loaded data is stored

            returns
            - None
        """
        

        self.data = self.fetch_data(ticker,
                                    start_date,
                                    end_date)

        self.data = self.data.sort_values(by='publish_date')

        self.data.to_csv(filepath, index=False)

        logger.save_logs(f"News dataset stored at {filepath}")

    def fetch_data(self,
                   ticker : str,
                   start_date : str,
                   end_date : str):
        """
            fetch data from newsAPI            
            
            parameters:
            - ticker : company of which data is loaded
            - start_date : date from which data is loaded
            - end_date : date upto which data is loaded

            returns
            - None
        """
        response = requests.get(f"https://newsapi.org/v2/everything?q={COMPANY[ticker]}&from={start_date}&to={end_date}&sortBy=popularity&apiKey={API_KEY}")

        soup = BeautifulSoup(response.content, 'html.parser')

        jsonstring = soup.get_text()

        data = json.loads(jsonstring)

        headline_text = []
        date = []

        for i in data['articles']:
            try:
                headline_text.append(i['title'] + " " + i['description'])
                date.append(i['publishedAt'].split('T')[0])
            except:
                pass

        return pd.DataFrame({
            'headline_text' : headline_text,
            'publish_date' : date
        })


    def get_dataset(self) -> pd.DataFrame:
        """
            Returns dataset

            parameters:
            - None

            returns:
            - dataframe
        """
        return self.data


def params(filepath):

    try:
        with open(filepath) as f:
            params_file = safe_load(f)
            
    except FileNotFoundError as e:
        logger.save_logs(msg='Parameters file not found, Switching to default values for train test split',
                                 log_level='error')
        
        current_date = datetime.date.today().strftime("%Y-%m-%d")
        date_15_days_ago = datetime.date.today() - datetime.timedelta(days=15)
        formatted_date = date_15_days_ago.strftime("%Y-%m-%d")

        return current_date, formatted_date
        
    else:
        logger.save_logs(msg=f'Parameters file read successfully',
                                    log_level='info')
        
        # read the parameters from the parameters file
        test_size = params_file['data']['start_date']
        random_state = params_file['data']['end_date']

        return test_size, random_state

if __name__ == "__main__" :

    ticker = sys.argv[1]
    stock_filepath = "stocks_1.csv"
    news_filepath = "news_1.csv"

    root_dir = Path(__file__).parent.parent.parent
    input_data_path = root_dir / "data" / "raw"

    params_file = root_dir / "config.yaml"

    start_date, end_date = params(params_file)

    stockcollector = StockDataCollector()
    newscollector = NewsDataCollector()

    stock_filepath = input_data_path / stock_filepath
    news_filepath = input_data_path / news_filepath
    
    stockcollector.load_data(ticker,
                             start_date= start_date,
                             end_date=end_date,
                             filepath=stock_filepath)
    
    newscollector.load_data(tikcer=ticker,
                            start_date=start_date,
                            end_date=end_date,
                            filepath=news_filepath)

