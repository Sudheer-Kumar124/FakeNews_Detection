# Fake News Detection using DEEP LEARNING


## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Models](#models)
- [Results](#Results)
- [Contact](#contact)


## Introduction

The spread of misinformation online necessitates the development of reliable tools to identify fake news. This project tackles this challenge by building a machine learning model to classify news articles as real or fake. We begin by collecting and preparing a dataset of labeled news articles. Text cleaning techniques and feature engineering might be employed to extract relevant information for analysis. Subsequently, a machine learning model will be trained on this data, enabling it to identify patterns indicative of fake news in unseen articles. The project aims to evaluate the model's performance and contribute to the fight against fake news. 

## Features

I'm writing this project to combat the spread of fake news online. Here's how it can be used in three key ways:

1. **Content Moderation on Platforms:**  Imagine social media platforms and online forums using my model. They could automatically flag potentially fake news articles shared by users. This would help:

    - Reduce the spread of misinformation: By identifying fake news early, platforms can prevent it from reaching a wider audience.
    - Improve information quality: Users would be exposed to more reliable information, fostering a healthier online environment.


2. **Fact-Checking Efficiency:** Fact-checking organizations are overwhelmed by the sheer volume of potentially fake news. My model can be a valuable tool to:

    - Prioritize investigations: I can analyze a large pool of articles, ranking them based on their likelihood of being fake.
    - Streamline workflow: Fact-checkers can focus their limited resources on the articles with the highest fake news probability scores, identified by my model.

3. **Educational Tool for Everyone:**  This project can be more than just a detection system. It can also be an educational tool to raise awareness about fake news:

    - Web application: Users can submit news articles and see my model's prediction (real or fake) along with explanations. These explanations could highlight features that led to the prediction (e.g., excessive exclamation points, specific keywords).
    - Browser extension: A browser extension can analyze news articles on the fly, highlighting potential indicators of fake news. This would prompt users to be more critical consumers of online information.

By implementing these use cases, my project has the potential to make a significant impact in the fight against fake news. It can empower platforms, fact-checkers, and everyday users to identify and avoid misinformation. 

 ## Installation

To run this project locally, please follow these steps:

1.	Clone the repository to your local machine using the following command:
This command will install the install the custom_dataset.

   ```
        https://github.com/Sudheer-Kumar124/FakeNews_Detection.git
   ```

## Usage

1. **Data Acquisition:**
   - You likely acquire data from separate sources, potentially containing real and fake news articles.

2. **Data Combination:**
   - The code suggests you might be merging these datasets (real and fake news) into a single DataFrame named `news`.

3. **Label Creation (Potential):**
   - While not explicitly shown, you might be creating a new column (e.g., `"label"`) to label real news articles with a specific value (e.g., 1) and fake news articles with another value (e.g., 0). This step helps the machine learning model understand the distinction between real and fake news during training.

4. **Data Shuffling:**
   - The code snippet `news = news.sample(frac=1)` shuffles the data in your DataFrame. This ensures a good mix of real and fake news examples throughout the DataFrame, which is crucial for training the model effectively.

5. **Text Cleaning:**
   - It's common practice in text analysis to clean the text data before feeding it to a machine learning model. This might involve removing punctuation, stop words (common words like "the", "a"), or unwanted characters (like URLs or HTML tags). While the code snippets you shared don't explicitly show this step, you might be using regular expressions (imported with `import re`) for text cleaning purposes.

6. **Feature Engineering:**
   - Extracting features from the text data can significantly enhance the performance of your model. These features could include:
      - Number of exclamation points or question marks.
      - Presence of specific phrases often used in fake news (e.g., "breaking news", "shocking truth").
      - Reading level analysis (complex vs. simpler language).
      - Sentiment analysis (positive, negative, neutral) of the text.
      - Structural features like length of title or article, presence of uppercase characters in the title, or number of external links.

**Overall, these data processing steps prepare your raw news data for effective machine learning model training, allowing it to learn the characteristics of real and fake news articles.**

**Note:** The specific steps involved might vary depending on your actual data sources and the features you choose for your model.

## Models

**Logistic regression**
Logistic regression is a supervised machine learning algorithm used for classification tasks where the goal is to predict the probability that an instance belongs to a given class or not. Logistic regression is a statistical algorithm which analyze the relationship between two data factors. The article explores the fundamentals of logistic regression, it’s types and implementations.

**decision tree**
A decision tree in machine learning is a versatile, interpretable algorithm used for predictive modelling. It structures decisions based on input data, making it suitable for both classification and regression tasks. This article delves into the components, terminologies, construction, and advantages of decision trees, exploring their applications and learning algorithms.

**Random Forest Classifier**
The Random forest or Random Decision Forest is a supervised Machine learning algorithm used for classification, regression, and other tasks using decision trees. Random Forests are particularly well-suited for handling large and complex datasets, dealing with high-dimensional feature spaces, and providing insights into feature importance. This algorithm’s ability to maintain high predictive accuracy while minimizing overfitting makes it a popular choice across various domains, including finance, healthcare, and image analysis, among others

## Results


![output_Screen](https://github.com/Sudheer-Kumar124/FakeNews_Detection/blob/Koushik_Kyadari/Output/output_screen.jpg)

fig-1.0

## Contact

| Name  | Email id |
| ------------- | ------------- |
| Kyadari Koushik  | kyadarii.koushik@gmail.com  |

