import os
import sys
import joblib
import warnings
import datetime
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings('ignore')

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.data_collection.make_dataset import NewsDataCollector

import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
from transformers import pipeline

class Model:

    def __init__(self):
        self.finbert = self.load_finbert_model()
        self.model = self.load_final_model()

    def load_final_model(self):
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        model_path = os.path.join(base_dir, 'models', 'final_model.h5')
        model = load_model(model_path)
        return model
    
    def load_finbert_model(self):
        return pipeline(task = "text-classification",  model="yiyanghkust/finbert-tone", tokenizer="yiyanghkust/finbert-tone", framework="tf", return_all_scores = True)
    
    def get_final_model(self):
        return self.model
    
    def get_finbert_model(self):
        return self.finbert


class Predictor:

    def __init__(self, window_size):
        self.stocks = None
        self.news = None
        self.window_size = window_size
        self.X1 = None
        self.X2 = None
        self.y = None
        self.stock_scaler = None
        self.news_scaler = None
        self.target_scaler = None
        self.get_scalers()

        self.model = Model()


    def get_data_dir(self, name = 'interim'):
        root_dir = Path(__file__).parent.parent.parent

        input_data_path = root_dir / "data" / name

        return input_data_path



    def get_sentiment_features(self, text):
        """
            Get sentiment scores and labels using FinBERT
        """
        result = self.model.finbert.predict(text)[0]
        
        # store the values in dict as label : score
        scores = {res['label']: res['score'] for res in result}

        # Get highest score label as output label for this text
        label = max(scores, key=scores.get)


        # return positive, negative, neutral and label    
        return scores["Positive"], scores["Negative"], scores["Neutral"], label


    def find_sentiment(self):

        label = []
        pos = []
        neu = []
        neg = []

        for i in self.news['headline_text']:
            v = self.get_sentiment_features(i)
            label.append(v[3])
            pos.append(v[0])
            neg.append(v[1])
            neu.append(v[2])

        self.news['label'] = label
        self.news['pos'] = pos
        self.news['neg'] = neg
        self.news['neu'] = neu

    def generate_group_data(self):
        pos = self.news['pos'].mean()
        neg = self.news['neg'].mean()
        neu = self.news['neu'].mean()
        label = self.news['label'].mode().values[0]

        if label == "Neutral":
            label = 0
        elif label == "Negative":
            label = -1
        else:
            label = 1

        self.news = pd.DataFrame({
            'pos' : [pos],
            'neg' : [neg],
            'neu' : [neu],
            'label' : [0]
        })

        self.X2 = self.news.values


    def make_sequence(self):
       
        self.X1 = self.stocks[['open', 'rolling_ma', 'RSI', 'MACD', 'Signal_Line','MACD_Histogram', 'SMA_20', 'upper_band', 'lower_band']].values
        self.X1 = np.array([self.X1])

    def get_scalers(self):
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        scaler_path = os.path.join(base_dir, 'models', 'artifacts', 'stock_scaler.pkl')
        self.stock_scaler = joblib.load(scaler_path)
        scaler_path = os.path.join(base_dir, 'models', 'artifacts', 'news_scaler.pkl')
        self.news_scaler = joblib.load(scaler_path)
        scaler_path = os.path.join(base_dir, 'models', 'artifacts', 'target_scaler.pkl')
        self.target_scaler = joblib.load(scaler_path)

    def scale_data(self):
        # convert the data into numpy array
        # self.X1 = np.array(self.X1.tolist())
        samples, timesteps, features = self.X1.shape

        # reshape to 2d since standard scaler won't work with 3d
        self.X1 = self.X1.reshape((samples, timesteps * features))

        self.X1 = self.stock_scaler.transform(self.X1)
        # again reshape to 3d
        self.X1 = self.X1.reshape((samples, timesteps, features))

        self.X2 = self.news_scaler.transform(self.X2.reshape(1, -1))


    def get_data(self, ticker):

        data_dir = self.get_data_dir()
        self.stocks = pd.read_csv(data_dir / "stocks_2.csv").tail(self.window_size)

        today = datetime.date.today().strftime("%Y-%m-%d")
        yesterday = datetime.date.today() - datetime.timedelta(1)
        yesterday = yesterday.strftime("%Y-%m-%d")


        newscollector = NewsDataCollector()
        self.news = newscollector.fetch_data(ticker, yesterday, today)

        self.find_sentiment()
        self.generate_group_data()

        self.make_sequence()

        self.scale_data()

    def predict(self):
        ypred = self.model.model.predict([self.X1, self.X2])
        return self.target_scaler.inverse_transform(ypred)


    
def predict(window_size, ticker):
    predictor = Predictor(window_size=5)

    predictor.get_data(ticker)

    print(predictor.predict())
    return predictor.predict()


if __name__ == "__main__":
    predict(5, 'AAPL')