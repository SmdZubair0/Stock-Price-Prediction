# 📈 Stock Price Prediction with News Sentiment

This project predicts stock prices by combining historical market data and news sentiment analysis. It uses LSTM models trained separately on both types of data and integrates them using the Keras Functional API.

---

## 📂 Folder Structure

<pre>
📦 stock-price-prediction/
    ├── notebooks/
    │ ├── Data_Collection_News.ipynb
    │ ├── Data_Collection_Stocks.ipynb
    │ ├── EDA.ipynb
    │ ├── Final_data.ipynb
    │ ├── Model_training_Stocks.ipynb
    │ └── New_Sentiment.ipynb
    ├── src/
    │ ├── data_collection/
    │ │ └── make_dataset.py
    │ ├── features/
    │ │ └── preprocessing.py
    │ ├── model/
    │ │ └── Prediction.py
    │ ├── visualizations/
    │ │ └── Visualization.py
    │ └── my_logger.py
    ├── app.py
    ├── api_key.py
    ├── config.yaml
    ├── dvc.yaml
    ├── requirements.txt
    ├── Dockerfile
    ├── setup.py
    ├── setup-project.sh
    ├── test.yaml
    ├── .gitignore
    └── README.md
</pre>

  
---

## ⚙️ Features

- 📊 Collects and cleans historical stock data and news articles
- 🧠 Performs sentiment analysis on news data
- 🧮 Trains LSTM models on both data streams
- 🔄 Merges both models using Keras Functional API
- 🌐 Flask-based web app for stock price prediction
- 🐳 Docker and DVC integrated for reproducibility and deployment

---

## 🔍 Workflow Overview

1. **Data Collection**:
   - `Data_Collection_Stocks.ipynb`: Gets historical stock prices from Yahoo Finance.
   - `Data_Collection_News.ipynb`: Fetches news articles for the same time period.

2. **Preprocessing**:
   - Clean and merge both datasets.
   - Perform sentiment analysis on news headlines using a custom lexicon-based or ML-based approach.

3. **Modeling**:
   - Train separate LSTM models for stock data and sentiment scores.
   - Use the Keras Functional API to combine them.

4. **Deployment**:
   - Flask app (`app.py`) serves the final model.
   - Dockerfile and `setup-project.sh` included for containerization and easy setup.

---

## 🧠 Tech Stack

- Python (pandas, numpy, scikit-learn, TensorFlow/Keras)
- Flask (for web interface)
- Docker + DVC (for reproducibility and deployment)
- BeautifulSoup / requests (for scraping news)
- YAML (for configurations)

---

## 🚀 How to Run

1. Clone the repository:
```bash
git clone https://github.com/yourusername/stock-price-prediction.git
cd stock-price-prediction

pip install -r requirements.txt

python app.py
```

Open browser and navigate to
http://127.0.0.1:5000/
