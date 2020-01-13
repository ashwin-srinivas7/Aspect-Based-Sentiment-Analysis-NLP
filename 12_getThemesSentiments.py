import numpy as np
import pandas as pd
import nltk
import math
import re

# for sentiment analysis
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyser = SentimentIntensityAnalyzer()

from tkinter import Tk
from tkinter.filedialog import askopenfilename

# function to get topics and Sentiments as list of tuples
def fn_getTopicAndSentiment(review):
    """
    Funtion to get topics in each review and the sentiment associated with each topic
    Input - sentence in a particular review
    Output - 
    """
    grams_topics = [] # list to store the topics associated with each trigram
    relevant_grams = [] # list to store trigrams with relevant topics
    sentiment_scores = [] # List to store the sentiment scores for each topic
    
    # tokenize the sentence into words at spaces
    review = str(review)
    token = review.split()
        
    # if there are 4 or more words in the sentence, form ngrams
    if(len(token)>4):
    
        number_of_grams = math.ceil(math.sqrt(len(token))) # dynamically setting the number of grams
        #number_of_grams = 3
        # 2 line code to generate ngrams
        ngrams = zip(*[token[i:] for i in range(number_of_grams)])
        ngram_list = [" ".join(ngram) for ngram in ngrams]
        
        # for each trigram in the list of all trigrams
        for gram in ngram_list:
            gram_tokens = gram.split() #split each ngram into individual words
            for word in gram_tokens:
                if word in list(df_dict['Word']): # if any of the words are in the dictionary
                    # look up the theme for that word
                    grams_topics.append(df_dict[df_dict['Word']==word].iloc[0]['Theme']) # add the theme to the list of themes
                    relevant_grams.append(gram) # add that trigram to the list to find the sentiment
    else: # for sentences with less than 4 words
        for word in token: 
            if word in list(df_dict['Word']): 
                    grams_topics.append(df_dict[df_dict['Word']==word].iloc[0]['Theme'])
                    relevant_grams.append(" ".join(token))
    
    #calculate the sentiment scores for each trigram
    for text in relevant_grams:
        text = text.replace("_"," ")
        score = analyser.polarity_scores(text)['compound']
        sentiment_scores.append(score)
    
    zipped_list = list(zip(relevant_grams, grams_topics, sentiment_scores))
    return zipped_list

# Function to get Sentiment Label based on the sentiment score
def fn_getSentimentLabel(score):
    score = float(score)
    if(score>=0.05):
        sentiment = "Positive"
    elif(score<=-0.05):
        sentiment = "Negative"
    else:
        sentiment = "Neutral"
    return sentiment


# ------------------- MAIN EXECUTION BEGINS HERE---------------------------------------#

print("Choose corpus file with review phrases")
# read the csv file into a dataframe
Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
filename = askopenfilename()
df_reviews = pd.read_csv(filename, encoding='utf-8')


print("Input the final dictionary csv")
Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
filename = askopenfilename()
df_dict = pd.read_csv(filename, encoding='utf-8')

print("Getting themes for each review. Please wait...")

# get themes and sentiments for each trigram in every sentence
df_reviews['trigrams_themes_sentiments'] = df_reviews['phrased_summary1'].apply(fn_getTopicAndSentiment)

# stack the list of tuples
df_tuple_list = pd.DataFrame.from_records(df_reviews['trigrams_themes_sentiments'].tolist()).stack().reset_index(level=1, drop=True).rename('trigrams_themes_sentiments')

df_reviews = df_reviews.drop('trigrams_themes_sentiments', axis=1).join(df_tuple_list).reset_index(drop=True)[['Rev_id', 'Review_Text','Final_Price_after_Discount', 'Category',
                                                                                                            'Retailer','Brand','Review_Rating_Score','Product_Name','Product_URL',
                                                                                                            'phrased_summary1','phrased_summary2','trigrams_themes_sentiments']]

# remove rows without any themes
df_reviews = df_reviews.dropna(subset=['trigrams_themes_sentiments'])

# unzip the tuple into 3 seperate columns
df_reviews[['trigrams','themes','sentiment_score']] = pd.DataFrame(df_reviews['trigrams_themes_sentiments'].tolist(),index=df_reviews.index)


# for each review-theme pair, calculate the mean of sentiment scores for a particular sentiment
df_reviews = df_reviews.groupby(['Rev_id','Review_Text','Final_Price_after_Discount','Category','Retailer','Brand',
                                  'Review_Rating_Score','Product_Name','Product_URL','themes']).mean().reset_index()


df_reviews = df_reviews[df_reviews['sentiment_score']!=0]
# get sentiment labels for each sentiment score
df_reviews['sentiment_label'] = df_reviews['sentiment_score'].apply(fn_getSentimentLabel)

# export this DF as a CSV
df_reviews.to_csv('BuzzOutput.csv')

print("Topics and sentiments have been analysed. \nFile outputted: BuzzOutput.csv ")