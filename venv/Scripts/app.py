import streamlit as st

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        background-color: #fdeef4;
        
        color: #ffffff;
        font-family: 'Arial', sans-serif;
    }
    .stTextInput, .stTextArea {
        background-color: #0d152d;
        border: 1px solid #ff4b4b;
        color: #ffffff;
    }
    .stTextInput:focus, .stTextArea:focus {
        border-color: #ffffff;
        box-shadow: 0 0 0 0.2rem rgba(255, 75, 75, 0.25);
    }
    .stButton button {
        background-color: #ff4b4b;
        color: #000000;
        border: none;
        padding: 10px 20px;
        font-size: 16px;
        border-radius: 5px;
        transition: background-color 0.3s ease;
        cursor: pointer;
    }
    .stButton button:hover {
        background-color: #ff6b6b;
    }
   .stMarkdown h1 {
    background-color: #0d152d;
    color: #ffffff;
    font-size: 3em;
    text-align: center;
    font-weight: bold;
    font-family: 'Manyavar', sans-serif;
    # padding: 100px; /* Increased padding inside the box */
    # height:30px;
    border: 7px solid #0d152d; 
    background-color: #0d152d; 
    border-radius: 10px; /* Optional: adds rounded corners */
    box-shadow: 0 4px 8px ; /* Optional: adds a shadow for a 3D effect */
}


    .stMarkdown h2 {
        color: #0d152d;
        font-size: 1.5em;
    }
    .feature-box {
        background-color: #2b2b2b;
        border: 1px solid #ff4b4b;
        border-radius: 5px;
        padding: 10px;
        margin: 10px 0;
        color: #000000;
    }
    .feature-box h3 {
        color: #ff4b4b;
        font-size: 1.2em;
    }
    .feature-box p {
        font-size: 1em;
        margin: 5px 0;
    }
    @keyframes pulse {
        0% { transform: scale(1); }
        100% { transform: scale(1.1); }
    }
    .sidebar {
        background-color: #0d152d;
        padding: 20px;
        position: fixed;
        height: 100%;
        width: 200px;
        top: 0;
        left: 0;
        overflow-y: auto;
        z-index: 100;
    }
    .sidebar a {
        color: #0d152d;
        display: block;
        padding: 10px;
        text-decoration: none;
        transition: all 0.3s ease;
        border-radius: 5px;
    }
    .sidebar a:hover {
        background-color: #ff4b4b;
        color: #0d152d;
        padding-left: 15px;
    }
    .sidebar a.active {
        background-color: #ff4b4b;
        color: #0d152d;
    }
    .sidebar h2 {
        color: #ff4b4b;
        font-size: 1.5em;
        margin-bottom: 10px;
    }
    .sidebar hr {
        border-color: #ff4b4b;
    }
    </style>
    """, unsafe_allow_html=True)

import streamlit as st

# CSS for the title and content
st.markdown("""
   <style>
    .title-container {
        text-align: center;
        padding: 20px;
        width: 1500px;
        # height: 20px;
        max-width: 100%; /* Adjust this to fit your needs */
        margin: 0 auto;
    }
    
    .title {
        font-family: 'Arial Black', sans-serif;
        font-size: 0.5em; /* Adjusted size */
        color: #ff4b4b;
        text-shadow: 2px 2px 4px #000000;
        animation: glow 1.5s infinite alternate;
    }
   
    @keyframes glow {
        0% {
            text-shadow: 2px 2px 8px #ff4b4b, -2px -2px 8px #ff4b4b;
        }
        100% {
            text-shadow: 2px 2px 12px #ff4b4b, -2px -2px 12px #ff4b4b;
        }
    }

    .logo {
        display: block;
        margin: 0 auto;
        max-width: 150px; /* Adjust size as needed */
    }
</style>

""", unsafe_allow_html=True)

# Display the logo and title
st.markdown("""
    <div class='title-container'>
        <img class='logo' src='C:/Users/meanu/Documents/infosys Interface/venv/Scripts/logo.png' alt='Logo' />
        <h1 class='title'>FAKE NEWS DETECTOR</h1>
    </div>
""", unsafe_allow_html=True)

st.markdown("<div class='content'>", unsafe_allow_html=True)
# st.markdown("<iframe src='https://www.statista.com/statistics/649221/fake-news-expose-responsible-usa/' width='100%' height='800'></iframe>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)




st.markdown("""
    <style>
    .sidebar {
        background-color: #0d152d; /* Dark royal blue background color for the sidebar */
        padding: 30px; /* Increase padding to ensure content fits */
        border-radius: 10px; /* Optional: add rounded corners */
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2); /* Optional: add a shadow for a 3D effect */
        width: 250px; /* Set a fixed width for the sidebar */
        margin: auto; /* Center the sidebar */
    }
    .sidebar h2 {
        color: #ffffff; /* White color for the heading */
        font-size: 1.5em;
        text-align: center;
        margin: 40px 0 20px 0; /* Increase top margin to shift heading lower */
    }
    .sidebar .option {
        color: #ffffff; /* White color for options */
        font-weight: bold; /* Make the text bold */
        text-align: center; /* Center-align the text */
        padding: 10px 0; /* Add vertical padding for spacing */
        font-size: 1.2em; /* Increase font size for better readability */
        transition: background-color 0.3s, color 0.3s; /* Smooth transition for background color and text color */
        display: block; /* Make each option block-level for dropdown functionality */
    }
    .sidebar .option:hover {
        background-color: #1e2a45; /* Darker shade of royal blue on hover */
        color: #ffffff; /* Keep text color white on hover */
    }
    .sidebar .option.active {
        background-color: #e0e0e0; /* Light grey background for active option */
        color: #0d152d; /* Dark royal blue text color for active option */
        border-radius: 5px; /* Optional: add rounded corners to the active option */
    }
    .divider {
        margin: 20px 0; /* Add vertical margin for spacing */
        border: 0; /* Remove default border */
        border-top: 2px solid #0d152d; /* Dark royal blue color for the horizontal line */
    }
    </style>
    <div class='divider'></div> <!-- Top horizontal line -->
    <div class='sidebar'>
        <h2>Navigation</h2>
        <a href='#' class='option active'>Home</a>
        <a href='#input' class='option'>Input</a>
        <a href='#check-stats' class='option'>Check Stats</a>
        <a href='#workflow' class='option'>Workflow</a>
        <!-- Dropdown for Workflow -->
        <div class='dropdown-content'>
            <a href='https://www.websequencediagrams.com/cgi-bin/cdraw?lz=QHN0YXJ0dW1sCgp0aXRsZSBEZXRhaWxlZCBCRVJUIE1vZGVsIFRyYWluaW5nIGFuZCBUZXN0aW5nIFNlcXVlbmNlCgphY3RvciBVc2VyCnBhcnRpY2lwYW50IERhdGFQcmVwYXJhdGlvbiBhcyBEUAAVDQBZBUluaXRpYWxpegAdCU1JADsNAHYIUHJvY2VzcyBhcyBUADoORXZhbHUAYAUAGQtFAFkTU2F2AIFCBXMgTVMAfA5ldHJpY3NDYWxjdWwAgH8KQwoKVXNlciAtPiBEUCA6IExvYWQgdACCBghkYXRhAA0TdmFsaWQAgW0GAAsXdGVzdAArE1ByZXAAgV0HAEkSU3BsaQAuBwpEUCAtPiBNSQAtB2FyZQCBBgZNSSAtPiBUUCA6IACCTQllIG1vZGVsABAMQ29uZmlndXIAFAcgcGFyYW1ldGVycwA2DERlZmluZSBvcHRpbWl6ZXIAg3gFc2NoZWR1bGVyAEMNcmVhdAB9BiBsb2FkZXJzCgpUAIEfBQCBBwVTdGFydACCLgkADgwAhFQFAIB_B2ZvciBlcG9jaCAxADEMQ29tcHV0ZSBsb3NzAEkMQmFja3Byb3BhZ2EACxNVcGRhdGUAgU4MAIEFCwCEKAdlIG9uAIMhDHNldACBDQ9jayBtAIQTBgCBQwxSZXBlYXQAgSULMgCBZghNUyA6IFNhdgCCUQhzdGF0ZQoKTVMgLT4gRQCEOAkAgwsGRQCDQAUAEQVDb21iaW5lAIRQCmFuAIQ1EgAlC1JlAIR_BQCCLgtmaW5hbACCOQZzAE4MAIFfDACEcQVzZXQAcwdNQyA6IEdlbmVyYXRlIGNsYXNzaWZpYwCHJgZyZXBvcgATF29uZnVzaW9uIG1hdHJpeABDDExvZyBlAIcECSByZXN1bHRzCgpNQyAtPgCIHQUAhWIFb3ZpZGUgZACIYggAgl0HAG0ICkBlbmQAiQ0F&s=default'>BERT Diagram</a>
            <a href='https://www.websequencediagrams.com/cgi-bin/cdraw?lz=dGl0bGUgTW9kZWwgVHJhaW5pbmcsIFZhbGlkYXRpb24sIGFuZCBFdmFsdQALBSBQcm9jZXNzCgpwYXJ0aWNpcGFudCBVc2VyAAQNU3lzdGVtABcNAEoKAC4NAFEKCgpVc2VyLT4ANgY6IEluaXRpYWxpemUgQmlMU1RNIG1vZGVsCgBWBgAeClNldHVwIGVtYmVkZGluZyBsYXllciwALQcsIENOTgCBPwZsaW5lYXIAGQZzADURRGVmaW5lIG9wdGltaXplcgAtBm9zcyBmdW5jAIEMFFN0YXJ0IHQAgi4HIHdpdGggcmFuZG9tAIEDCnMgKwAYCgAMCgCBMxFSdW4gZXBvY2gAgQ8KAIJ6CjoAgwgIZQCBfAYgYWZ0ZXIgZWFjaAAvBgoAgyYKAII1ClByb3ZpZGUgdgCDQwkgYWNjdXJhY3kAgjoJVXNlcjogRGlzcGxheQCBQQphbmQAIxQgZm8AaQ0AgW0jR2xvVmUABYIKAIQLFG5vAIQ6CwCCHYIGACeBbACKcAUAiDcGAIhGBSArAIcmDGRhdGEAiWARQ29tYmluZQCHFxkAIBUAYgYAiDoGAIkmBWMAQAZkAF8FAIdWBTIAiHsIAIp6D1BlcmZvcm0AiD4MAIloBQCKeRYAiywLdGVuc29ycwCMPgVjb3VudACKZRQAiVMJaW4gYmF0Y2hlcyBvZiA2MACLYAkAjHQKOgCNAggAigIIcACBFQZhbmNlCgCNGwoAiXMSAIlyCCwgY29uZnVzaW9uIG1hdHJpeACNWwZjbGFzc2lmaWMAjV8GcmVwb3J0AI0BEACKIAgAin8GZQCOCgpyZXN1bHQAjEoKAIpRBgCKeQgARxUAgg0HAH4OAI16BwCLEAZBbmFseXoAgVUUABoNSW50ZXJwcmV0IFIAgH8HADkMQ2xhc3MgMDogUHJlY2kAgXIFMC44NSwgUmVjYWxsIDAuODgAHhMxACIPMgAmCzcyAE8TMgBUDjc3ACUMNgCBABMzAIEFDjQ2AIEICzI1AIExEzQAgTgMMS4wMACBOwkxLjAwAIFiEzUAgQQPAIFpDDc5AIJeDU92ZXJhbGwAjhcJOiA4NyUAglkOQ29uY2wAhDMFOiBIaWdoZXN0AI4OCgCPCgYAhjcILACPHwdzaG93cyBzdHJvbmcAhR0MIGJ1dCBuZWVkcyByAJBwBW1lbnQAjlcFc3BlY2lmaWMAhQkGZXMuCgo&s=default'>CNN Multimodal</a>
            <a href='https://www.websequencediagrams.com/cgi-bin/cdraw?lz=QHN0YXJ0dW1sCgp0aXRsZSBCaUxTVE0gTW9kZWwgVHJhaW5pbmcgUHJvY2VzcyB3aXRoIERpZmZlcmVudCBTY2VuYXJpb3MKCmFjdG9yIFVzZXIKcGFydGljaXBhbnQgRGF0YVByZXBhcmF0aW9uIGFzIERQABUNAGUFSW5pdGlhbGl6AB0JTUkAOw0AgREGAIEHCWFzIEJMVAA2ElNhdgAYB01TAFgOZXRyaWNzQ2FsY3VsAFsKQwCBIA1FdmFsdQCBHwlFVgoKVXNlciAtPiBEUCA6IExvYWQgdACCCwhkYXRhAA0TdmFsaWQAgWYGAAsXdGVzdAArE1ByZXAAgl8HAEkSU3BsaQAuBwpEUCAtPiBNSQAtB2FyZQCBBgZNSSAtPiBCTFQgOiAAgkcJZSBtb2RlbAoKYWx0IFJhbmRvbSBFbWJlZGRpbmdzICsAg1kKAAwKCiAgICAAQgxTZXQgcgA2BmUAGA5CTFQAawoAhCkFAGoGAIQjBgAjESBhbgCCNwsAQQtlbHNlIEdsb1ZlAGg2ADUGAHAtACMQAH4eAIInFE5vdACBTHNubwCEewkAgh4ZAG4sAIFzSACBCw1uZAoKAIQPDVBlcmZvcm0AhjUKbG9vcAAVDkZvcndhcmQgcGFzcwAvDgCHLghlIGxvAAwQQmFja3Byb3BhZwCIXAUAaA5VcGRhdGUgcGFyYW1ldGVyAEkPVHJhY2sAgQcMAGkQABsGAIdICwB7BQCBUghNUyA6IFNhdgCGXQcgc3RhdGUKCk1TIC0-IEVWAIgqCACGewZFVgANCUNvbWJpbmUAiEEKYW4AiCYSACULUmUAiHAFAIZTB2ZvciBmaW5hbCBlcG9jaHMATgwAiToHZSBvbgCIYQZzZXQAcwdNQyA6IEdlbmVyYXRlIGNsYXNzaWZpYwCLEAZyZXBvcgATF29uZnVzaW9uIG1hdHJpeABDDExvZyBlAIoqCnJlc3VsdHMKCk1DIC0-AIwHBQCJUwVvdmlkZSBkZXRhaWxlZCBtAIsFBgBtCApAZW5kAIx8BQ&s=default>BiLSTM Diagram</a>
        </div>
    </div>
    <div class='divider'></div> <!-- Bottom horizontal line -->
""", unsafe_allow_html=True)










# Description
# st.markdown('<h2 style="font-size: 36px;">About Fake News</h2>', unsafe_allow_html=True)
st.markdown('''
    <style>
    @keyframes glow {
        0% { text-shadow: 0 0 5px #ff0000; }
        50% { text-shadow: 0 0 20px #ff0000; }
        100% { text-shadow: 0 0 5px #ff0000; }
    }
    .glow-text {
        color: #ff0000; /* Adjust the color of the text */
        animation: glow 1.5s infinite;
    }
    </style>
    <h2 class="glow-text">About Fake News</h2>
''', unsafe_allow_html=True)


st.markdown("""
<p style='font-size: 1.3em; color: #000000;'>
Fake news affects people’s daily lives, manipulates their thoughts and feelings, changes their beliefs, and may lead them to make wrong decisions. The propagation of fake news on social media negatively affects society in many domains such as political, economic, social, health, technological, and sports. Social media platforms have presented a virtual environment for posting, discussion, exchange of views, and global interaction among users, without restrictions on location, time, or content volume. In 2021, Facebook announced that about 1.3 billion fake accounts had been closed, and more than 12 million posts containing false information about COVID-19 and vaccines had been deleted.
</p>
""", unsafe_allow_html=True)

# Check Stats section
st.markdown("<h2 id='Some Statistics'>Some Statistics</h2>", unsafe_allow_html=True)



import streamlit as st

# Styling for images with hover effects
image_style = """
    <style>
    .image-container {
        position: relative;
        display: inline-block;
    }
    .image-container img {
        width: 50%; /* Adjust the width as needed */
        height: auto; /* Maintain aspect ratio */
        transition: transform 0.2s ease-in-out;
    }
    .image-container img:hover {
        transform: scale(1.05); /* Slight zoom effect */
    }
    </style>
"""

import streamlit as st

# CSS for images with hover effect and reduced size
image_style = """
    <style>
    .image-container img {
        width: 70%; /* Adjust the width as needed */
        height: auto; /* Maintain aspect ratio */
        transition: transform 0.2s ease-in-out;
    }
    .image-container img:hover {
        transform: scale(1.1); /* Slight zoom effect */
    }
    </style>
"""

import streamlit as st

# CSS for images with hover effect and reduced size
st.markdown("""
    <style>
    .small-image {
        width: 70%; /* Adjust the width as needed */
        height: auto; /* Maintain aspect ratio */
        transition: transform 0.2s ease-in-out;
    }
    .small-image:hover {
        transform: scale(1.1); /* Slight zoom effect */
    }
    </style>
""", unsafe_allow_html=True)

# Percentage of using social media
st.markdown("""
    <h2 style="color: #000000;">Percentage of using social media</h2>
""", unsafe_allow_html=True)

st.markdown("""
    <img class="small-image" src="C:/Users/meanu/Documents/infosys Interface/venv/Scripts/Social media use.jpg" alt="Description of the second statistic" />
""", unsafe_allow_html=True)

st.markdown("""
<p style='color: #000000;'>
The image shows the percentage of people using social media for various purposes. The most common usage is "Stay up-to-date with news and current events" at over 37%. Other popular uses include "Find funny or entertaining content" at around 33% and "Stay in touch with what in-time friends are doing" at around 31%. The least common uses are "Promote and sell products or services" at around 5%, "Follow celebrities, influencers, and public figures" at around 8%, and "Make sure do not miss out on anything" at around 9%. The data suggests that people primarily use social media to stay informed, be entertained, and maintain connections, rather than for commercial or self-promotional purposes.
<a href="https://www.smartinsights.com/social-media-marketing/social-media-strategy/new-global-social-media-research/" style='color: #a1a1a;'>Link to more information</a>
</p>
""", unsafe_allow_html=True)

# Social media role in the spread of Fake news
st.markdown("""
    <h3 style="color: #000000;">Social media role in the spread of Fake news</h3>
""", unsafe_allow_html=True)

st.markdown("""
    <img class="small-image" src="C:/Users/meanu/Documents/infosys Interface/venv/Scripts/social media how responsible that is.jpg" alt="Description of the third statistic" />
""", unsafe_allow_html=True)

st.markdown("""
<p style='color: #000000;'>
The pie chart shows the public's perception of how much social media is responsible for the spread of fake news. According to the chart, 60% of respondents believe that social media is "Mostly responsible" for the spread of fake news, while 6% think it is "Partly responsible but other media sources are more responsible". Additionally, 6% believe that social media is "Not at all responsible", and 5% "Do not know".
<a href="https://www.researchgate.net/publication/379872133_'The_Role_of_Social_Media_in_the_Spread_of_Misinformation_and_Fake_News'" style='color: #a1a1a;'>Link to more information</a>
</p>
""", unsafe_allow_html=True)

st.markdown('<hr style="margin: 3px 0">', unsafe_allow_html=True)
st.markdown('<hr style="margin: 3px 0">', unsafe_allow_html=True)



# import streamlit as st
# import numpy as np
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# from keras.preprocessing.sequence import pad_sequences

# Define the function to calculate metrics
def model_evaluation(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return mae, rmse, r2




# Define the function to calculate metrics
def model_evaluation(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return mae, rmse, r2

def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    st.pyplot()

def plot_roc_curve(y_true, y_pred_prob):
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc='lower right')
    st.pyplot()

def plot_precision_recall_curve(y_true, y_pred_prob):
    precision, recall, _ = precision_recall_curve(y_true, y_pred_prob)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    st.pyplot()

def plot_feature_importance(features, importances):
    plt.figure(figsize=(10, 8))
    sns.barplot(x=importances, y=features, palette='viridis')
    plt.xlabel('Feature Importance')
    plt.title('Feature Importance')
    st.pyplot()

def plot_prediction_distribution(predictions):
    plt.figure(figsize=(8, 6))
    sns.histplot(predictions, bins=30, kde=True, color='skyblue')
    plt.xlabel('Prediction Value')
    plt.ylabel('Frequency')
    plt.title('Distribution of Predictions')
    st.pyplot()


# User input section
st.markdown("<h2 id='input'>Input</h2>", unsafe_allow_html=True)

news_text = st.text_area("Enter News Article Text:", height=200, key="input_text")

# Content for Input section (Added directly under the header)
st.markdown("""
    <div id="input" class="content" style="color: #000000;">
        <p>Enter the news article text in the box above.</p>
    </div>
""", unsafe_allow_html=True)

# Check button to classify news
check_button = st.button("Check")
if check_button:
    if news_text.strip() != "":
        # Predict
        sequences = tokenizer.texts_to_sequences([news_text])
        padded_sequences = pad_sequences(sequences, maxlen=max_len)
        prediction = model.predict(padded_sequences)[0][0]
        predicted_label = "Fake" if prediction < 0.5 else "Real"
        confidence = prediction if predicted_label == "Real" else 1 - prediction

        # Store prediction and confidence in session state
        st.session_state.prediction = prediction
        st.session_state.confidence = confidence
        st.session_state.predicted_label = predicted_label

        st.markdown(f"<h3>Prediction: {predicted_label}</h3>", unsafe_allow_html=True)
        st.markdown(f"<h3>Confidence: {confidence * 100:.2f} %</h3>", unsafe_allow_html=True)

        # LIME explanation
        explainer = LimeTextExplainer(class_names=['Fake', 'Real'])
        explanation = explainer.explain_instance(news_text, model.predict_proba, num_features=10)

        # Show explanation
        st.markdown("<h3>Explanation:</h3>", unsafe_allow_html=True)
        fig = explanation.as_pyplot_figure()
        st.pyplot(fig)
    else:
        st.markdown("<h3>Please enter some text for classification.</h3>", unsafe_allow_html=True)

# Add the Show Analysis button
show_analysis_button = st.button("Show Analysis")
if show_analysis_button:
    try:
        # Check if prediction is available in session state
        if 'prediction' in st.session_state:
            # Dummy values for demonstration (replace with actual y_true and y_pred)
            y_true = np.array([1])  # Replace with actual true values
            y_pred = np.array([st.session_state.prediction])  # Use the prediction from the Check button
            y_pred_prob = np.array([st.session_state.prediction])  # For ROC and Precision-Recall curve
            
            # Calculate metrics
            mae, rmse, r2_square = model_evaluation(y_true, y_pred)
            
            # Display metrics
            st.markdown("""
                <h3>Model Training Performance</h3>
                <p><strong>RMSE:</strong> {:.2f}</p>
                <p><strong>MAE:</strong> {:.2f}</p>
                <p><strong>R2 score:</strong> {:.2f}%</p>
            """.format(rmse, mae, r2_square * 100), unsafe_allow_html=True)
            
            # Visualizations
            st.markdown("<h3>Confusion Matrix</h3>", unsafe_allow_html=True)
            plot_confusion_matrix(y_true, y_pred)

            st.markdown("<h3>ROC Curve</h3>", unsafe_allow_html=True)
            plot_roc_curve(y_true, y_pred_prob)

            st.markdown("<h3>Precision-Recall Curve</h3>", unsafe_allow_html=True)
            plot_precision_recall_curve(y_true, y_pred_prob)

            st.markdown("<h3>Feature Importance</h3>", unsafe_allow_html=True)
            # Example feature names and importances (replace with actual values)
            features = ['Feature1', 'Feature2', 'Feature3']  # Replace with actual feature names
            importances = [0.1, 0.2, 0.3]  # Replace with actual feature importances
            plot_feature_importance(features, importances)

            st.markdown("<h3>Distribution of Predictions</h3>", unsafe_allow_html=True)
            # Example predictions (replace with actual values)
            predictions = [0.2, 0.3, 0.8]  # Replace with actual prediction values
            plot_prediction_distribution(predictions)
        else:
            st.markdown("<h3>No prediction available. Please check the news article first.</h3>", unsafe_allow_html=True)
    except Exception as e:
        st.error(f"An error occurred: {e}")





# Footer
st.markdown("<footer style='position: absolute; bottom: 0; left: 0; width: 100%; padding: 10px; text-align: center;'>Powered by Streamlit</footer>", unsafe_allow_html=True)

# Close main content area
st.markdown("</div>", unsafe_allow_html=True)

# Run the Streamlit app
if __name__ == "__main__":
    st.write("Streamlit app is running.")