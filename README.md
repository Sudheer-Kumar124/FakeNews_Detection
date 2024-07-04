# Fake News Detection using Deep Learning
## Introduction
Fake news has become a significant global problem, negatively impacting various aspects of society, including politics, economics, and social well-being. This project aims to develop a model for detecting fake news on social media by leveraging sentiment analysis of news content and emotion analysis of users' comments.

## Dataset
The project uses the Fakeddit dataset, a large-scale and multi-modal dataset (text and image) collected from the social media platform Reddit. The dataset consists of more than one million posts from various domains, with several features associated with each post, such as images, comments, users, domains, and other metadata. The dataset contains a variety of fake news types, including misleading content, manipulated content, false connection, imposter content, and satire.

### Data Preprocessing
The data preprocessing steps include:

### 1) Sentiment Analysis: 
Analyzing the sentiment polarity of the news titles using the TextBlob library, which provides a sentiment score ranging from -1 (negative) to 1 (positive).
### 2) Text Cleaning:
Removing noise and irrelevant characters from the text, such as punctuation, numbers, and multiple spaces.
### 3) Stop-words Removal and Lemmatization: 
Removing common stop-words and applying lemmatization to reduce words to their base forms.
### 4) Tokenization and Padding:
Converting the preprocessed text into numerical format and padding the sequences to a fixed length to meet the requirements of the deep learning models.

### Unimodal Approaches

### Three unimodal deep learning models are explored:

### Convolutional Neural Network (CNN): 
The CNN model uses multiple convolutional filters of different sizes to capture n-gram features of the text. It also leverages pre-trained GloVe word embeddings to improve performance.

### Bidirectional Long Short-Term Memory (BiLSTM): 
The BiLSTM model combines a Bidirectional LSTM and a CNN to capture both local and long-range dependencies in the text.
### Bidirectional Encoder Representations from Transformers (BERT): 
The BERT model uses pre-trained contextual word embeddings from the BERT language model, which outperforms the other unimodal approaches.
### Multimodal Approach
The multimodal approach combines the text and image inputs using a CNN architecture. The text processing part is similar to the unimodal CNN model, while the image processing part uses a series of convolutional and max-pooling layers to extract visual features. The outputs from the text and image processing are then concatenated and passed through dense layers to perform the final classification.

## Evaluation
The performance of the models is evaluated using various metrics, including recall, precision, F1 score, micro and macro averages, and overall accuracy. The researchers are particularly interested in the models' ability to detect different types of fake news, so they compute additional metrics focused on the 5 fake news classes (excluding the "true" class).

## Results
The key findings from the results are:

1) The multimodal CNN model outperforms all the unimodal approaches, achieving an accuracy of 87% and micro F1 of 87%.
2) Among the unimodal models, the BERT model achieves the best performance, with an accuracy of 78% and micro F1 of 74%.
3) The BiLSTM model with dynamic GloVe vectors is the third-best unimodal approach.
4) The traditional SVM baseline performs the poorest among the models compared.
5) 
## Future Work
As future work, we can explore following:

1) Incorporating pre-trained visual feature extractors like VGG.
2) Experiment with deep learning techniques like LSTM, GRU, and different fusion methods (late fusion).

## Usage
## Usage

To use the project, follow these steps:

1. **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/fake-news-detection.git
    ```
2. **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3. **Preprocess the data:**
    ```bash
    python preprocess.py
    ```
4. **Train and evaluate the models:**
    ```bash
    python train.py
    python evaluate.py
    ```
5. **(Optional) Deploy the trained model for real-world use:**
    ```markdown
    # Example deployment script (deploy.py)
    import pickle
    from flask import Flask, request, jsonify

    # Load the trained model
    with open('model.pkl', 'rb') as file:
        model = pickle.load(file)

    # Initialize the Flask app
    app = Flask(__name__)

    @app.route('/predict', methods=['POST'])
    def predict():
        data = request.get_json(force=True)
        prediction = model.predict([data['text']])
        return jsonify({'prediction': prediction[0]})

    if __name__ == '__main__':
        app.run(debug=True)

    # Run the deployment script
    python deploy.py
    ```

   
References:
1)	[https://iopscience.iop.org/article/10.1088/1757-899X/1099/1/012040]
2)	[https://arxiv.org/pdf/2102.04458]
3)	[https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10006567/]
4)	Boididou, C., Andreadou, K., Papadopoulos, S., Dang-Nguyen, D.-T., Boato, G., Riegler, M., Kompatsiaris, Y. et al. (2015). Verifying Multimedia Use at MediaEval 2015. MediaEval, 3 , 7.
5)	Breiman, L. (2001). Random Forests. Machine Learning, 45 , 5–32. doi:[https: //doi.org/10.1023/A:1010933404324.]
6)	Brownlee, J. (). A Gentle Introduction to Early Stopping to Avoid Overtraining Neural Networks. [https://machinelearningmastery.com/ early-stopping-to-avoid-overtraining-neural-network-models/] 
7)	Deng, L., & Liu, Y. (2018). Deep learning in natural language processing. (1st ed.). Springer.
8)	[https://colab.research.google.com/drive/1edXYIghmzu3Bs4UhWJ8Ho9sk13oaebfz?usp=sharing]
9)	Baheti, P. (2020). Introduction to Multimodal Deep Learning. Retrieved from [https://heartbeat.comet.ml/introduction-to-multimodal-deep-learning-630b259f9291]. Accessed November 13, 2021
10)	Brown, E. (2019). Online fake news is costing us $78 billion globall each year. ZDNet. [https://www.zdnet.com/article/online-fake-news-costing-us-78-billion-globally-each-year/]


