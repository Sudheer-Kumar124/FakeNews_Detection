# Detecting Fake News on Social Media Content
- In today's digital age, the spread of misinformation poses significant challenges to information integrity. This project aims to detect fake news articles using text analysis techniques without employing machine learning models.


# Introduction
- This project explores techniques for detecting fake news articles based on textual features and patterns. Instead of relying on machine learning models, the focus is on analyzing and processing text data using traditional programming and statistical methods.

# Dataset
- The dataset (FakeNewsNet.csv) used in this project consists of news article titles labeled as real or fake. The data is cleaned and preprocessed to prepare it for text analysis.

# Data Preprocessing
Data preprocessing is essential to transform raw data into a clean and usable format, thereby improving data quality and preparing it for analysis or model training. The preprocessing steps include:

- Conversion to Lowercase: Ensures uniformity by converting all text to lowercase.
- Removal of Hyperlinks: Cleans the text by removing any embedded links.
- Elimination of Newline Characters (\n): Converts text into a single line format.
- Removal of Words Containing Numbers: Strips out irrelevant words that contain numbers.
- Handling of Special Characters: Removes special characters to simplify the text.
- Removal of Stop Words: Eliminates common words that do not contribute to the meaning.
- Stemming and Lemmatization: Reduces words to their base or root forms.


# Feature Extraction
Textual data is transformed into feature vectors using TF-IDF (Term Frequency-Inverse Document Frequency) to represent the information numerically.

# Model Building and Evaluation:
- SVM Model: Built a linear SVM classifier (SVC(kernel='linear')), trained it on the training set (X_train, y_train), and evaluated its performance on the test set (X_test, y_test). Performance metrics included accuracy, confusion matrix, and classification report.
- Random Forest Model: Constructed a Random Forest classifier (RandomForestClassifier(n_estimators=100)), trained it, and evaluated its performance similarly.
- Gradient Boosting Model: Developed a Gradient Boosting classifier (GradientBoostingClassifier()), trained it, and assessed its performance.


# Result
The project aimed to detect fake news articles using traditional machine learning classifiers: SVM (Support Vector Machine), Random Forest, and Gradient Boosting. Here are the key findings and performance metrics from each classifier:

### Support Vector Machine (SVM)

Accuracy: 0.83
Precision: 0.85 (fake), 0.81 (real)
Recall: 0.80 (fake), 0.86 (real)
F1-score: 0.82 (fake), 0.83 (real)
### Random Forest

Accuracy: 0.82
Precision: 0.84 (fake), 0.80 (real)
Recall: 0.78 (fake), 0.84 (real)
F1-score: 0.81 (fake), 0.82 (real)
### Gradient Boosting

Accuracy: 0.80
Precision: 0.81 (fake), 0.78 (real)
Recall: 0.76 (fake), 0.82 (real)
F1-score: 0.78 (fake), 0.80 (real)
# Performance Comparison
Accuracy: SVM achieved the highest accuracy of 83%, followed closely by Random Forest with 82% and Gradient Boosting with 80%.
Precision and Recall: SVM showed balanced precision and recall for both fake and real classes, indicating effective classification capability. Random Forest and Gradient Boosting also performed well but showed slightly lower recall for fake news articles.
F1-score: SVM consistently showed the highest F1-scores across both classes, indicating robust performance in terms of precision and recall balance.
# Conclusion
Based on the evaluation metrics, SVM emerges as the most effective classifier for detecting fake news in this project. It demonstrated the highest accuracy and balanced performance in distinguishing between fake and real news articles. Random Forest and Gradient Boosting, while competitive, showed slightly lower performance metrics compared to SVM.

