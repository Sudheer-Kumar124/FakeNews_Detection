# Fake News Detection on Social Media Content Using Deep Learning

## Introduction
The rapid spread of information through social media platforms has made it increasingly difficult to discern credible news from misinformation. The proliferation of fake news has led to widespread public confusion, undermined trust in legitimate sources, and posed significant risks to public health and safety. To address this, artificial intelligence, particularly deep learning, offers a promising solution. This project aims to develop a deep learning model to detect and filter out false information from digital platforms by analyzing textual content. This will help identify misleading information and protect the public from the adverse effects of misinformation.

## Approach

### Data Collection
The team has gathered a comprehensive dataset from various social media platforms, which serves as the foundation for model training and evaluation.

### Data Preprocessing
The collected data is meticulously cleaned and prepared for model training, ensuring that the input features are in a suitable format for the various machine learning and deep learning techniques.

## Model Development
Three distinct models are implemented and trained on the pre-processed data:
- Logistic Regression (LR) Model
- LSTM (Long Short-Term Memory) Model
- Convolutional Neural Networks (CNN)

## Model Performance

### Logistic Regression Performance
- **Accuracy:** 83.26%
- **Precision for Class 0 (Real):** 0.71
- **Precision for Class 1 (Fake):** 0.86
- **Recall for Class 0:** 0.52
- **Recall for Class 1:** 0.93
- **F1-Score for Class 0:** 0.60
- **F1-Score for Class 1:** 0.89

This classification report provides a detailed breakdown of the model's performance, indicating its capability to distinguish between real and fake news effectively. The high precision and recall for Class 1 highlight the model's proficiency in identifying fake news.

### LSTM Model Performance
**Model Architecture:**
- Embedding Layer
- LSTM Layer
- Dropout Layer
- LSTM Layer
- Dropout Layer
- Dense Layer

**Total Parameters:** 2,101,057
**Trainable Parameters:** 2,101,057
**Non-trainable Parameters:** 0

**Training and Validation Metrics:**
- **Epoch 1:**
  - Loss: 1.1436
  - Accuracy: 76.82%
  - Validation Loss: 0.4389
  - Validation Accuracy: 82.50%
- **Epoch 2:**
  - Loss: 0.3763
  - Accuracy: 85.39%
  - Validation Loss: 0.5461
  - Validation Accuracy: 81.34%
- **Epoch 3:**
  - Loss: 0.2936
  - Accuracy: 89.14%
  - Validation Loss: 0.4418
  - Validation Accuracy: 82.32%
- **Epoch 4:**
  - Loss: 0.2613
  - Accuracy: 90.97%
  - Validation Loss: 0.4946
  - Validation Accuracy: 81.70%
- **Epoch 5:**
  - Loss: 0.4488
  - Accuracy: 81.08%

**Test Metrics:**
- Test Accuracy: 81.08%
- Precision: 82.99%
- Recall: 94.30%

### CNN Model Performance
**Epoch Results:**
- **Epoch 10/10:**
  - Loss: 0.0090
  - Accuracy: 0.9972
  - Validation Loss: 0.9461
  - Validation Accuracy: 0.7967

**Final Evaluation Metrics:**
- **Precision:**
  - Fake: 0.94
  - Real: 0.99
- **Recall:**
  - Fake: 0.95
  - Real: 0.98
- **F1-Score:**
  - Fake: 0.95
  - Real: 0.98
- **Support:**
  - Fake: 1461
  - Real: 4539
- **Overall Accuracy:** 97%

## Final Model
The <b>Convolutional Neural Network (CNN)</b> stands out as the best-performing model based on performance metrics when compared with the other two models (LSTM and Logistic Regression) in detecting fake news.

## Future Scope
- **Model Enhancements**
  - Ensemble methods
  - Transfer learning
  - Regular updates and maintenance
  - Model retraining
  - Algorithm optimization
- **Ethical Considerations**
  - Bias mitigation
  - Privacy and security
- **Scalability and Deployment**
  - Scalable solutions
  - Cloud integration

## Conclusion
The <b>CNN model</b> significantly outperforms both the Logistic Regression and LSTM models in terms of accuracy, precision, recall, and F1-score. The high precision and recall for both classes (fake and real news) indicate the model's robustness and reliability in correctly identifying fake news. The CNN's superior performance metrics highlight its effectiveness and efficiency in distinguishing between fake and real news. Therefore, the CNN model stands out as the best choice for fake news detection among the evaluated models.

