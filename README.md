# SimpleRNN_IMBD_Movie-Review-system
IMDB Movie Review Sentiment Analysis with Simple RNN

## 🎬 Overview

This project is a web application that classifies movie reviews from IMDB as **Positive** or **Negative** using a **Simple Recurrent Neural Network (RNN)**. The app is built with Streamlit and TensorFlow, providing an interactive and user-friendly interface to test the model's sentiment prediction capabilities.

A key feature of this project is its handling of negation. Words like "not" are combined with the following word (e.g., "not good" becomes "not_good") during preprocessing, allowing the model to better capture nuanced negative sentiment.

## ✨ Features

- **Interactive UI**: A clean and simple interface built with Streamlit.
- **Real-time Sentiment Prediction**: Classify any movie review instantly.
- **Negation Handling**: A custom preprocessing step to improve accuracy on reviews with negations.
- **Example Reviews**: Pre-loaded examples to quickly test positive, negative, and negated sentiments.
- **Visual Feedback**: Displays the sentiment, a prediction score, and a confidence progress bar.

## 🧠 Model Architecture

The sentiment analysis model is a Simple RNN built with `tensorflow.keras`. The architecture consists of:
1.  **Embedding Layer**: Converts the 10,000 most frequent words in the IMDB vocabulary into 128-dimensional dense vectors.
2.  **SimpleRNN Layer**: A recurrent layer with 128 units and ReLU activation to process the sequence of words.
3.  **Dense Output Layer**: A single neuron with a sigmoid activation function that outputs a probability score between 0 (Negative) and 1 (Positive).

The model was trained on the IMDB dataset and saved as `simple_rnn_imdb.h5`.

## 🛠️ Technologies Used

- **Python**
- **TensorFlow / Keras**: For building and training the RNN model.
- **Streamlit**: For creating the interactive web application.
- **NumPy**: For numerical operations.

## 🚀 Getting Started

### Prerequisites

Make sure you have Python 3.8+ installed.

### Installation

1.  **Clone the repository:**
    ```
    git clone https://github.com/your-username/your-repo-name.git
    cd your-repo-name
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required libraries:**
    ```
    pip install -r requirements.txt
    ```
    The `requirements.txt` file should contain:
    ```
    streamlit
    tensorflow
    numpy
    ```

### Running the Application

Once the setup is complete, run the Streamlit app with the following command:
         streamlit run main.py

 

This will open the application in your default web browser at `http://localhost:8501`.

## 🧑‍💻 How to Use

1.  **Enter a Review**: Type or paste a movie review into the text area.
2.  **Use Examples**: Alternatively, select one of the example reviews from the sidebar to see how the model handles different inputs.
3.  **Classify**: Click the **"🔍 Classify Sentiment"** button.
4.  **View Results**: The app will display the predicted sentiment (**Positive** or **Negative**), the raw prediction score, and a visual progress bar.

## 📂 Project Structure

    . 
    ├── main.py                # The Streamlit web application script
    ├── simplernn.py           # The script for training and saving the RNN model 
    ├── simple_rnn_imdb.h5     # The pre-trained Keras model
    ├── requirements.txt       # Python dependencies 
    └── README.md              # You are here!


## 🙏 Acknowledgements

- This project utilizes the **IMDB Movie Review Dataset**.
- Built with the amazing **Streamlit** and **TensorFlow** libraries.


