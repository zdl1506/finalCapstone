# This Python program will perform sentiment analysis on a dataset of product reviews from Amazon customers.

# Import libraries
import spacy
import pandas as pd
from textblob import TextBlob

# Load spaCy model
nlp = spacy.load('en_core_web_md')

# Load the 'amazon product reviews' dataset
dataframe = pd.read_csv('amazon_product_reviews.csv', low_memory=False)

# Create a preprocess text function
def preprocess_text(text):
    doc = nlp(text)
    tokens = [token.lemma_.lower() for token in doc if not token.is_stop and not token.is_punct]
    return ' '.join(tokens)

# Apply preprocessing to the entire reviews.text column
dataframe['clean_reviews'] = dataframe['reviews.text'].dropna().apply(preprocess_text)

# Define a function for sentiment analysis
def predict_sentiment(review):
    # Use TextBlob for sentiment analysis
    polarity = TextBlob(review).sentiment.polarity
    
    # Predict sentiment based on polarity
    if polarity > 0:
        return 'Positive'
    elif polarity < 0:
        return 'Negative'
    else:
        return 'Neutral'

# Test the sentiment analysis function on a sample of Amazon product reviews
def test_sentiment_analysis():
    sample_reviews = dataframe['reviews.text'].iloc[[5, 35, 60, 64, 82]]
    
    print("\n--------------- Sample Product Reviews Sentiment Analysis ---------------")
    for i, review in enumerate(sample_reviews, start=1):
        sentiment = predict_sentiment(review)
        print(f"Review {i}: '{review}' - Sentiment: {sentiment}")

# Run the sentiment analysis by calling the test_sentiment_analysis function
test_sentiment_analysis()

# Apply sentiment analysis to each review
dataframe['sentiment'] = dataframe['clean_reviews'].apply(predict_sentiment)

# Print results
print()
print('--------------- Clean Reviews ---------------')
print(dataframe['clean_reviews'].head())

print('\n--------------- Sentiment Analysis Results ---------------')
print(dataframe[['reviews.text', 'sentiment']].head())

# Create a similarity calculation function
def calculate_similarity(text1, text2):
    similarity_result = nlp(text1).similarity(nlp(text2))
    return similarity_result

# Use the similarity calculation function to calculate the similarity of the selected reviews
review1 = dataframe['clean_reviews'][20]
review2 = dataframe['clean_reviews'][3000]
similarity_score = calculate_similarity(review1, review2)

# Print results
print('\n--------------- Similarity Review of Two Amazon Product Reviews ---------------')
print(f"\nReview One: {review1}")
print(f"Review Two: {review2}")
print(f"\nSimilarity of Reviews: {similarity_score}\n")