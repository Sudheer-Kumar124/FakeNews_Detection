# Fake News Detection using Deep Learning

## Introduction
Fake news has become a significant global problem, negatively impacting various aspects of society, including politics, economics, and social well-being. This project aims to develop a model for detecting fake news on social media by leveraging sentiment analysis of news content and emotion analysis of users' comments.

## Dataset
The project uses the **Fakeddit** dataset, a large-scale and multi-modal dataset (text and image) collected from the social media platform Reddit. The dataset consists of more than one million posts from various domains, with several features associated with each post, such as images, comments, users, domains, and other metadata. The dataset contains a variety of fake news types, including misleading content, manipulated content, false connection, imposter content, and satire.

### Data Preprocessing
The data preprocessing steps include:

1. **Sentiment Analysis:** 
   Analyzing the sentiment polarity of the news titles using the TextBlob library, which provides a sentiment score ranging from -1 (negative) to 1 (positive).

2. **Text Cleaning:**
   Removing noise and irrelevant characters from the text, such as punctuation, numbers, and multiple spaces.

3. **Stop-words Removal and Lemmatization:** 
   Removing common stop-words and applying lemmatization to reduce words to their base forms.

4. **Tokenization and Padding:**
   Converting the preprocessed text into numerical format and padding the sequences to a fixed length to meet the requirements of the deep learning models.

## Unimodal Approaches
Three unimodal deep learning models are explored:

- **Convolutional Neural Network (CNN):** 
  The CNN model uses multiple convolutional filters of different sizes to capture n-gram features of the text. It also leverages pre-trained GloVe word embeddings to improve performance.

- **Bidirectional Long Short-Term Memory (BiLSTM):** 
  The BiLSTM model combines a Bidirectional LSTM and a CNN to capture both local and long-range dependencies in the text.

- **Bidirectional Encoder Representations from Transformers (BERT):** 
  The BERT model uses pre-trained contextual word embeddings from the BERT language model, which outperforms the other unimodal approaches.

## Multimodal Approach
The multimodal approach combines the text and image inputs using a CNN architecture. The text processing part is similar to the unimodal CNN model, while the image processing part uses a series of convolutional and max-pooling layers to extract visual features. The outputs from the text and image processing are then concatenated and passed through dense layers to perform the final classification.

## Evaluation
The performance of the models is evaluated using various metrics, including recall, precision, F1 score, micro and macro averages, and overall accuracy. The researchers are particularly interested in the models' ability to detect different types of fake news, so they compute additional metrics focused on the 5 fake news classes (excluding the "true" class).

## Results
The key findings from the results are:

1. The multimodal CNN model outperforms all the unimodal approaches, achieving an accuracy of 87% and micro F1 of 87%.
2. Among the unimodal models, the BERT model achieves the best performance, with an accuracy of 78% and micro F1 of 74%.
3. The BiLSTM model with dynamic GloVe vectors is the third-best unimodal approach.
4. The traditional SVM baseline performs the poorest among the models compared.
   ![Models Comparision](https://github.com/Sudheer-Kumar124/FakeNews_Detection/blob/Anushka_Thakur/images/Screenshot%202024-07-04%20231831.png?raw=true) 

## Future Work
As future work, we can explore the following:

1. Incorporating pre-trained visual feature extractors like VGG.
2. Experimenting with deep learning techniques like LSTM, GRU, and different fusion methods (late fusion).

## Usage

echo "# Fake News Detection Project

Welcome to the Fake News Detection project! This guide will walk you through the steps to set up and run the project on your local machine.

## Table of Contents

1. [Project Setup](#project-setup)
2. [Adjusting File Paths](#adjusting-file-paths)
3. [Running the Application](#running-the-application)
4. [Optional: Deploying the Model](#optional-deploying-the-model)
5. [Additional Information](#additional-information)
6. [License](#license)
7. [Contact](#contact)

## Project Setup

### 1. Clone the Repository

First, clone the repository from GitHub to your local machine:

\`\`\`bash
git clone https://github.com/Sudheer-Kumar124/FakeNews_Detection.git
cd FakeNews_Detection
\`\`\`

### 2. Create and Activate a Virtual Environment

Create a virtual environment in the project directory:

\`\`\`bash
python -m venv venv
\`\`\`

Activate the virtual environment:

- **For Windows:**

    \`\`\`bash
    venv\\Scripts\\activate
    \`\`\`

- **For macOS/Linux:**

    \`\`\`bash
    source venv/bin/activate
    \`\`\`

### 3. Install Dependencies

Navigate to the \`Scripts\` folder where \`requirements.txt\` is located and install the required packages:

\`\`\`bash
cd Scripts
pip install -r requirements.txt
\`\`\`

### 4. Verify Installation

Ensure that all required packages are installed correctly. If there are any issues, you may need to manually install missing dependencies.

## Adjusting File Paths

Before running the application, you need to configure the paths to the model and image files. Follow these steps:

### 1. Adjust the Model Path

Open the \`app.py\` file and locate the section where the model is loaded. Update the path to point to your model file:

\`\`\`python
# Example of updating model path in app.py
model_path = 'path/to/your/model/file'
\`\`\`

Replace \`'path/to/your/model/file'\` with the relative path to the model file in your project.

### 2. Adjust the Image Paths

Similarly, adjust the paths for any images used in your application. Open \`app.py\` and find the sections where image paths are specified:

\`\`\`python
# Example of updating image paths in app.py
image_path = 'path/to/your/image/file'
\`\`\`

Replace \`'path/to/your/image/file'\` with the relative path to the image files you are using.

## Running the Application

With all paths correctly set up, you can now run the application. Ensure you are in the \`Scripts\` folder and your virtual environment is activated:

\`\`\`bash
python app.py
\`\`\`

This will start the application, and you should be able to access it through the provided local URL (usually \`http://localhost:8501\`).

## Optional: Deploying the Model

If you want to deploy the trained model for real-world use, you can use a Flask application. Follow these steps:

### 1. Create the Deployment Script

Create a file named \`deploy.py\` with the following content:

\`\`\`python
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
\`\`\`

### 2. Run the Deployment Script

Execute the deployment script to start the Flask server:

\`\`\`bash
python deploy.py
\`\`\`

The server will run and you can make POST requests to the \`/predict\` endpoint to get predictions.

## Additional Information

- **Model Path**: Ensure that the path specified in \`app.py\` matches the location of your model file. If your project structure changes, update this path accordingly.
- **Image Paths**: Ensure all image files are correctly referenced. If the images are in a different folder, adjust the paths in \`app.py\` accordingly.
- **Dependencies**: If you face issues with missing packages, check \`requirements.txt\` and install any missing dependencies manually.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For any questions or issues, please reach out to [Your Name](mailto:your-email@example.com).
" > README.md


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

## References
1. [https://iopscience.iop.org/article/10.1088/1757-899X/1099/1/012040](https://iopscience.iop.org/article/10.1088/1757-899X/1099/1/012040)
2. [https://arxiv.org/pdf/2102.04458](https://arxiv.org/pdf/2102.04458)
3. [https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10006567/](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10006567/)
4. Boididou, C., Andreadou, K., Papadopoulos, S., Dang-Nguyen, D.-T., Boato, G., Riegler, M., Kompatsiaris, Y. et al. (2015). Verifying Multimedia Use at MediaEval 2015. MediaEval, 3, 7.
5. Breiman, L. (2001). Random Forests. Machine Learning, 45, 5–32. doi: [https://doi.org/10.1023/A:1010933404324](https://doi.org/10.1023/A:1010933404324)
6. Brownlee, J. (2020). A Gentle Introduction to Early Stopping to Avoid Overtraining Neural Networks. [https://machinelearningmastery.com/early-stopping-to-avoid-overtraining-neural-network-models/](https://machinelearningmastery.com/early-stopping-to-avoid-overtraining-neural-network-models/)
7. Deng, L., & Liu, Y. (2018). Deep learning in natural language processing. (1st ed.). Springer.
8. [https://colab.research.google.com/drive/1edXYIghmzu3Bs4UhWJ8Ho9sk13oaebfz?usp=sharing](https://colab.research.google.com/drive/1edXYIghmzu3Bs4UhWJ8Ho9sk13oaebfz?usp=sharing)
9. Baheti, P. (2020). Introduction to Multimodal Deep Learning. Retrieved from [https://heartbeat.comet.ml/introduction-to-multimodal-deep-learning-630b259f9291](https://heartbeat.comet.ml/introduction-to-multimodal-deep-learning-630b259f9291). Accessed November 13, 2021.
10. Brown, E. (2019). Online fake news is costing us $78 billion globally each year. ZDNet. [https://www.zdnet.com/article/online-fake-news-costing-us-78-billion-globally-each-year/](https://www.zdnet.com/article/online-fake-news-costing-us-78-billion-globally-each-year/)
