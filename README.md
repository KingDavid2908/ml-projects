# AI Engineer Showcase Projects (Built in Colab)

## Overview

This repository contains a Google Colab notebook (`.ipynb` file) featuring a series of mini-projects built rapidly to showcase practical skills relevant to the AI Engineer roles. The projects demonstrate capabilities across Natural Language Processing (NLP), Large Language Models (LLMs), Machine Learning (Classification, Evaluation, Comparison), Deep Learning (TensorFlow, PyTorch), and basic Computer Vision (OpenCV).

## Projects Included

The notebook is structured sequentially with the following projects:

1.  **Project 1: AI-Powered Customer Feedback Analyzer**
    * **Goal:** Analyze customer reviews for sentiment and extract keywords.
    * **Skills:** NLP, Sentiment Analysis, Keyword Extraction (TF-IDF), Prototyping.
    * **Tech:** Python, Hugging Face `transformers`, Scikit-learn, Pandas.

2.  **Project 2: LLM-Powered Summarization Tool**
    * **Goal:** Generate concise summaries of text using a Large Language Model API.
    * **Skills:** LLMs, API Integration (Google Gemini), Prompt Engineering, Prototyping, Automation.
    * **Tech:** Python, `google-generativeai` library, Colab Secrets (for API Key).

3.  **Project 3: Logistic Regression Classifier**
    * **Goal:** Build and evaluate a binary classifier for the Breast Cancer Wisconsin dataset.
    * **Skills:** Machine Learning Fundamentals, Classification, Data Preprocessing (Scaling), Model Evaluation (Accuracy, Confusion Matrix, Classification Report).
    * **Tech:** Python, Scikit-learn, Pandas.

4.  **Project 4: Model Comparison**
    * **Goal:** Compare the performance of Logistic Regression, SVM, Random Forest, and Gradient Boosting on the same classification task.
    * **Skills:** Model Evaluation, Model Selection, Comparative Analysis.
    * **Tech:** Python, Scikit-learn.

5.  **Project 5a: MNIST Neural Network Classifier (TensorFlow/Keras)**
    * **Goal:** Build, train, and evaluate a simple Dense Neural Network for handwritten digit recognition.
    * **Skills:** Deep Learning, Neural Networks, TensorFlow, Keras API, Data Preprocessing for DL.
    * **Tech:** Python, TensorFlow, Keras.

6.  **Project 5b: MNIST Neural Network Classifier (PyTorch)**
    * **Goal:** Re-implement the MNIST classifier using PyTorch to demonstrate framework versatility.
    * **Skills:** Deep Learning, Neural Networks, PyTorch, Data Loaders, Custom Training Loops.
    * **Tech:** Python, PyTorch, Torchvision.

7.  **Project 7: Basic OpenCV Demo**
    * **Goal:** Demonstrate basic image processing using OpenCV.
    * **Skills:** Computer Vision Fundamentals, Image Processing (Edge Detection).
    * **Tech:** Python, OpenCV (`cv2`), Matplotlib, NumPy.

## Technologies Used

* **Python 3**
* **Google Colab** (Development Environment)
* **Core Libraries:**
    * Pandas
    * NumPy
* **Machine Learning:**
    * Scikit-learn (`LogisticRegression`, `SVC`, `RandomForestClassifier`, `GradientBoostingClassifier`, `StandardScaler`, `train_test_split`, metrics)
* **Deep Learning:**
    * TensorFlow (with Keras API)
    * PyTorch
    * Torchvision
* **NLP/LLMs:**
    * Hugging Face `transformers`
    * `google-generativeai`
* **Computer Vision:**
    * OpenCV (`opencv-python-headless`)
* **Visualization:**
    * Matplotlib
    * Seaborn

## How to Use/Run

1.  **Download:** Obtain the `.ipynb` notebook file.
2.  **Upload to Colab:** Open Google Colab and upload the notebook ("File" -> "Upload notebook").
3.  **Run Cells:** Execute the cells sequentially from top to bottom.
    * Dependencies are installed using `!pip` within the notebook.
    * Most projects are self-contained or rely on data loaded in previous steps within the notebook session.
4.  **API Key (Project 2 - LLM Summarizer):** To run the LLM Summarizer section, you will need your own Google AI API key.
    * Obtain a key from Google AI Studio.
    * In the Colab notebook, go to the "Secrets" tab on the left sidebar, add a new secret named `GOOGLE_API_KEY`, paste your key as the value, and enable "Notebook access". The code cell that configures the `google-generativeai` library will then use this key.
