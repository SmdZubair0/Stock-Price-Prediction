# Set project name
# PROJECT_NAME="Stock-Price-Prediction"

# # Create the main project directory
# # mkdir $PROJECT_NAME && cd $PROJECT_NAME

# echo "Creating project structure..."

# # Create folders
# mkdir -p .github/workflows data/raw data/processed data/interim models notebooks src tests

# # Create essential files
# touch config.yaml dvc.yaml Dockerfile .gitignore README.md requirements.txt setup.py

# echo "Writing .gitignore..."
# cat <<EOL > .gitignore
# newVenv/
# __pycache__/
# *.log
# *.csv
# *.h5
# *.pkl
# models/
# data/
# EOL

# echo "Writing config.yaml..."
# cat <<EOL > config.yaml
# data:
#   stock_api: "yfinance"
#   news_api_key: "#"
#   start_date: "1980-12-12"
#   end_date: "2025-04-02"

# model:
#   lstm_units: [50, 50]   # LSTM layers with 50 units each
#   return_sequences: [True, False]  # Matches the layers
#   dense_units: [25, 1]   # Dense layers with 25 and 1 neuron
#   batch_size: 32
#   epochs: 20
#   optimizer: adam
#   learning_rate: 0.001
#   input_shape: [5, 9]
# EOL

# echo "Writing dvc.yaml..."
# cat <<EOL > dvc.yaml
# stages:
#   data_ingestion:
#     cmd: python src/data_ingestion.py
#     deps:
#       - src/data_ingestion.py
#     outs:
#       - data/raw/

#   preprocessing:
#     cmd: python src/data_preprocessing.py
#     deps:
#       - src/data_preprocessing.py
#       - data/raw/
#     outs:
#       - data/processed/

#   training:
#     cmd: python src/model_training.py
#     deps:
#       - src/model_training.py
#       - data/processed/
#     outs:
#       - models/
# EOL

# echo "Writing Dockerfile..."
# cat <<EOL > Dockerfile
# FROM python:3.9

# WORKDIR /app
# COPY requirements.txt .
# RUN pip install -r requirements.txt

# COPY . .
# CMD ["python", "src/model_training.py"]
# EOL

# echo "📝 Writing GitHub Actions workflow..."
# mkdir -p .github/workflows
# cat <<EOL > .github/workflows/mlops.yml
# name: MLOps Pipeline

# on:
#   push:
#     branches:
#       - main

# jobs:
#   train_model:
#     runs-on: ubuntu-latest
#     steps:
#       - name: Checkout Code
#         uses: actions/checkout@v3

#       - name: Setup Python
#         uses: actions/setup-python@v4
#         with:
#           python-version: 3.9

#       - name: Install Dependencies
#         run: pip install -r requirements.txt

#       - name: Run Tests
#         run: pytest tests/

#       - name: Train Model
#         run: python src/model_training.py
# EOL

# echo "📝 Writing setup.py..."
# cat <<EOL > setup.py
# from setuptools import setup, find_packages

# setup(
#     name="stock_prediction",
#     version="0.1",
#     packages=find_packages(),
#     install_requires=open("requirements.txt").read().splitlines(),
# )
# EOL

# echo "Writing requirements.txt..."
# cat <<EOL > requirements.txt
# # Core Libraries
# numpy==1.23.5
# pandas==1.5.3
# scipy==1.10.1

# # Data Collection
# yfinance==0.2.31
# newsapi-python==0.2.7
# requests==2.31.0

# # Data Processing
# scikit-learn==1.3.0
# matplotlib==3.7.1
# seaborn==0.12.2

# # Time Series & Feature Engineering
# statsmodels==0.14.0
# pmdarima==2.0.3

# # NLP (Sentiment Analysis)
# nltk==3.8.1
# textblob==0.17.1
# transformers==4.38.2
# torch==2.2.0

# # Deep Learning (LSTM)
# tensorflow==2.13.0
# keras==2.13.1

# # Other Utilities
# tqdm==4.66.1
# jupyterlab==4.0.5
# notebook==7.0.6
# EOL

# echo "🔧 Setting up virtual environment..."
# python3 -m venv venv
# source venv/bin/activate

# echo "📦 Installing dependencies..."
# pip install --upgrade pip
# pip install -r requirements.txt

# echo "🔀 Initializing Git..."
# git init
# git add .
# git commit -m "Initial MLOps project setup"

# echo "🗂 Initializing DVC..."
# pip install dvc
# dvc init
# dvc add data/raw/

# echo "✅ Setup complete! Activate your virtual environment using:"
# echo "source venv/bin/activate"
