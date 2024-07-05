# Boosting Financial Market Stability by Predicting Fake News with Deep Learning and Machine Learning

This project aims to enhance financial market stability by detecting fake news using advanced machine learning and deep learning techniques. By leveraging models like LSTM, Random Forest, Logistic Regression, and Gradient Boosting Classifier, we strive to identify fake news with high accuracy, thus contributing to a more informed and stable market environment.

## Introduction

The proliferation of fake news poses a significant threat to financial markets, influencing investor decisions and market dynamics. This project utilizes various machine learning models to predict and identify fake news, thereby mitigating its adverse effects on financial stability.

## Models Trained
- **LSTM (Long Short-Term Memory)**
- **Random Forest**
- **Logistic Regression**
- **Gradient Boosting Classifier**

## Evaluation Metrics
- **Root Mean Squared Error (RMSE)**
- **Mean Absolute Error (MAE)**
- **R-squared (R²) score**
- **Accuracy**
- **Loss**

### LSTM Model Performance
- **Architecture:**
  - Embedding Layer
  - Spatial Dropout Layer
  - LSTM Layer
  - Dense Layer
- **Total Parameters:** 25,998,005
- **Trainable Parameters:** 25,998,005

#### Training and Validation Metrics
- **Epoch 1:**
  - Loss: 0.0999
  - Accuracy: 96.40%
  - Validation Loss: 0.0640
  - Validation Accuracy: 98.27%
- **Epoch 2:**
  - Loss: 0.0196
  - Accuracy: 99.53%
  - Validation Loss: 0.0099
  - Validation Accuracy: 99.80%
- **Epoch 3:**
  - Loss: 0.1749
  - Accuracy: 93.98%
  - Validation Loss: 0.0792
  - Validation Accuracy: 97.14%
- **Epoch 4:**
  - Loss: 0.0607
  - Accuracy: 97.82%
  - Validation Loss: 0.0963
  - Validation Accuracy: 96.40%
- **Epoch 5:**
  - Loss: 0.0315
  - Accuracy: 99.10%
  - Validation Loss: 0.0601
  - Validation Accuracy: 98.49%

#### Test Metrics
- **Test Loss:** 0.0566
- **Test Accuracy:** 98.46%

### Comparison with Other Models
1. **Random Forest:**
   - **RMSE:** 0.0497
   - **MAE:** 0.0025
   - **R² score:** 99.01%
2. **Logistic Regression:**
   - **RMSE:** 0.1209
   - **MAE:** 0.0146
   - **R² score:** 94.15%
3. **Gradient Boosting Classifier:**
   - **RMSE:** 0.0727
   - **MAE:** 0.0053
   - **R² score:** 97.88%
4. **LSTM:**
   - **Test Loss:** 0.0566
   - **Test Accuracy:** 98.46%

### Final Verdict
The **Random Forest model** stands out as the best-performing model based on RMSE, MAE, and R² score metrics. However, the **LSTM model** also performs exceptionally well, especially in terms of accuracy. If the primary goal is accuracy, the LSTM model is highly competitive. If considering overall error metrics (RMSE, MAE, R²), the Random Forest model remains the best choice.

## Requirements
- **Python 3.12**

### Python Libraries
- `pandas==1.5.3`
- `numpy==1.24.3`
- `scikit-learn==1.2.2`
- `nltk==3.8.1`
- `matplotlib==3.7.1`
- `seaborn==0.12.2`
- `scipy==1.10.1`
- `tensorflow==2.13.0`

## Setup Instructions

### Prerequisites
- Ensure you have [Anaconda](https://www.anaconda.com/products/distribution) installed.
- Install [VS Code](https://code.visualstudio.com/) if you haven't already.

### Steps to Set Up the Project

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Sudheer-Kumar124/FakeNews_Detection.git
   cd FakeNews_Detection
   git checkout Krishnashis_Das
   ```

2. **Create and activate the virtual environment:**
   ```bash
   conda create --name fake_news_detection python=3.12
   conda activate fake_news_detection
   ```

3. **Install the required packages:**
   ```bash
   pip install -r requirements.txt
   ```

## Conclusion
This project demonstrates the application of various machine learning and deep learning models to detect fake news, contributing to financial market stability. The comparison of models highlights the strengths of each approach, with the Random Forest model performing the best overall.However, the **LSTM model** also performs exceptionally well, especially in terms of accuracy.

## Acknowledgements
This project was developed with the help of various open-source libraries and tools. Special thanks to the developers and the community for their contributions.
