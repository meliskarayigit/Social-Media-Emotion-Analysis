# Social-Media-Emotion-Analysis
Turkish Social Media Sentiment Analysis

This project performs sentiment analysis on Turkish social media comments. It includes text preprocessing (lemmatization, stopword removal), exploratory data analysis, feature engineering (TF-IDF, N-gram), traditional ML models, and a deep learning model using TensorFlow.

 Dataset File: social_media_comments.csv

Columns:
-Paylaşım (Comment text)
- Tip (Sentiment: Pozitif or Negatif)

Preprocessing Steps
-Dropped missing values
-Renamed columns to Comment and Sentiment
-Calculated Comment_Length
-Removed stopwords (custom Turkish list)
-Applied lemmatization (WordNet)
-Cleaned comments for punctuation and casing

Exploratory Data Analysis
-Comment length distribution via histogram and boxplots
-Top 15 words with frequencies and percentages
-WordCloud visualization
-Sentiment distribution bar chart
-Positive vs Negative frequent word comparison

Feature Engineering
-TF-IDF vectorization with 1-gram and 2-gram features
-Dimensionality reduction: max_features=5000, min_df=5, max_df=0.85

 Models Trained

➤ Traditional ML Models
- Logistic Regression 0.8395 Accuracy
- Naive Bayes 0.8503 Accuracy
- Random Forest 0.8566 Accuracy
- Linear SVM 0.8440 Accuracy


Confusion matrices and classification reports are printed.

Cross-validation (5-fold) performed on best model (Random Forest).

Learning Curve analysis for Random Forest included.

➤ Deep Learning (Keras)

Tokenization and padding applied to comment text

Label encoding (Binary)

Model: Embedding + GlobalAveragePooling1D + Dense layers

Trained over 100 epochs with validation split

Test Accuracy: Reported after evaluation

Visualizations
-Accuracy bar plot for model comparison
-Confusion matrix heatmap for best model
-Learning curve plot (train vs validation accuracy)
-Training vs validation accuracy for Keras model

Requirements
pip install pandas matplotlib seaborn nltk sklearn tensorflow wordcloud stanza

How to Run
-Upload the dataset: social_media_comments.csv
-Run the notebook or Python script step-by-step
-View performance metrics and plots

Notes
-Turkish stopwords and Unicode characters are handled manually
-Outliers in comment lengths are filtered before training
-Model interpretability is enhanced with word frequency analysis

Language: Turkish

Task: Social media sentiment classification using NLP techniques

Let me know if you'd like a .ipynb or .py version packaged with the README!
