import numpy as np
import pandas as pd
import re
import nltk
import unicodedata
import string

from tkinter import Tk
from tkinter.filedialog import askopenfilename

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
from gensim import corpora, models

def fn_tokenizeReview(review):
    """
    Funtion to tokenize reviews
    Input - String of review text
    Output - List of tokens in the review
    """
    review = str(review)
    review_tokens = review.split()
    return review_tokens
    
"""
def fn_biGramPhrase(review, bigram_mod):
    
    Funtion returns bi-gram phrases in the review
    Input - review, bigram model
    Output - list of bi-gram phrases
    
    return bigram_mod[review]
"""
def fn_triGramPhrase(review, trigram_mod, bigram_mod):
    """
    Funtion returns tri-gram phrases in the review
    Input - review, trigram model and bigram model
    Output - List of tri-gram phrases in the review
    """
    return ' '.join(trigram_mod[bigram_mod[review]])

def fn_joinWords(list_of_words):
    """
    Funtion to join items of a list into a sentence
    """
    return ' '.join(list_of_words)

# --------- MAIN EXECUTION STARTS HERE ------------#
Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
filename = askopenfilename()
df_reviews = pd.read_csv(filename, encoding='utf-8')


print("Analyzing bigram and trigram phrases. Please wait. This may take a few minutes...")

# -------------- summary1 ------------------ #
# tokenize the reviews 
df_reviews['token_list'] =  df_reviews['summary1'].apply(fn_tokenizeReview)

# Build the bigram and trigram models
bigram = gensim.models.Phrases(df_reviews['token_list'], min_count=5, threshold=100) # higher threshold fewer phrases.
trigram = gensim.models.Phrases(bigram[df_reviews['token_list']], threshold=100)

# Faster way to get a sentence clubbed as a trigram/bigram
bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)

# get bigram and tri gram phrases in new columns of dataframe
# df_reviews['bigram_phrase'] = df_reviews['token_list'].apply(fn_biGramPhrase, bigram_mod=bigram_mod)
df_reviews['phrased_summary1'] = df_reviews['token_list'].apply(fn_triGramPhrase, trigram_mod=trigram_mod, bigram_mod=bigram_mod)


# ------------------ summary2 --------------------- #

# tokenize the reviews 
df_reviews['token_list'] =  df_reviews['summary2'].apply(fn_tokenizeReview)

# Build the bigram and trigram models
bigram = gensim.models.Phrases(df_reviews['token_list'], min_count=5, threshold=100) # higher threshold fewer phrases.
trigram = gensim.models.Phrases(bigram[df_reviews['token_list']], threshold=100)

# Faster way to get a sentence clubbed as a trigram/bigram
bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)

# get bigram and tri gram phrases in new columns of dataframe
# df_reviews['bigram_phrase'] = df_reviews['token_list'].apply(fn_biGramPhrase, bigram_mod=bigram_mod)
df_reviews['phrased_summary2'] = df_reviews['token_list'].apply(fn_triGramPhrase, trigram_mod=trigram_mod, bigram_mod=bigram_mod)


# df_reviews = df_reviews.filter(['brand','sku_desc','review_text','normalized_review','rating','phrases'])
# output dataframe to CSV
df_reviews = df_reviews.drop(['token_list','normalized_review'], axis = 1)
df_reviews.to_csv('reviewPhrases.csv')
print("Reviews have been phrased.\nFile outputted: reviewPhrases.csv")