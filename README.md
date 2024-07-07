# Detecting Fake News on Social Media Content - LinkedIn, Twitter, Facebook & Instagram

![image](https://github.com/Sudheer-Kumar124/FakeNews_Detection/assets/91627540/47dff5b2-2697-48a1-9365-a2cecf245145)


---

## Description

In today's digital age, social media platforms such as LinkedIn, Twitter, Facebook, and Instagram play a pivotal role in information dissemination. However, the rapid spread of fake news across these platforms poses significant challenges, including misinformation and its potential societal impact. This project aims to develop robust machine learning models to detect and combat fake news, leveraging advanced techniques such as Logistic Regression, BERT, and LSTM models.

---

## Dataset Used

#### FakeNewsNet.csv

---

## Data Preprocessing

Data preprocessing is a crucial step in the data analysis and machine learning pipeline. It involves transforming raw data into a clean and usable format. The main objectives are to improve the quality of the data and to prepare it for analysis or model training.

#### Data preprocessing steps include:

- Conversion to lowercase
- Removal of hyperlinks
- Elimination of newline characters (\n)
- Removal of words containing numbers
- Handling of special characters
- Removal of stop words
- Stemming and Lemmatization

---

## Exploratory Data Analysis (EDA)
Exploratory Data Analysis (EDA) involves using data visualization techniques to understand the distribution, patterns, and relationships within a dataset.

Some of the used EDA techniques used in this project for data visualization are are:

  - Count plot
  - Pie Chart
  - Bar Plot
  - Word Cloud
  - Histogram
  - Distribution Plots (KDE Plot)

---

## Models Overview

### 1. Logistic Regression

- Logistic Regression is a statistical method for analyzing a dataset in which there are one or more independent variables that determine an outcome. The outcome is a binary variable (0 or 1, true or false, yes or no).
- Logistic Regression serves as the baseline model for this project. It is chosen for its simplicity and interpretability in classifying news articles based on extracted features.

### 2. BiLSTM (Long Short-Term Memory) Model

- LSTM models are well-suited for sequential data analysis, making them ideal for processing textual information found in social media posts. They excel in capturing long-term dependencies and context within text data.
- A Bidirectional LSTM is an extension of the traditional LSTM architecture that enhances its ability to capture patterns from sequential data by processing it in both forward and backward directions.

### 3. BERT-based Model (DistilBERT Model)

- LSTM models, known for their ability to capture sequential dependencies in data, are particularly suited for analyzing textual data like social media posts. These models excel in understanding context and detecting patterns in the sequence of words, crucial for identifying nuanced misinformation.
- DistilBERT is a smaller, faster, and lighter version of BERT (Bidirectional Encoder Representations from Transformers). It retains 97% of BERT's language understanding while being 60% faster and 40% smaller. DistilBERT is trained using a technique called knowledge distillation, where a smaller network (the student) is trained to reproduce the behavior of a larger network (the teacher, in this case, BERT).

---

## Model Performance comparision
![image](https://github.com/Sudheer-Kumar124/FakeNews_Detection/assets/91627540/cf3a15f2-be74-4baa-ac4e-c582fd0048bd)

---

## Authors

- [Tatwadarshi](https://github.com/Dev7091)

---


