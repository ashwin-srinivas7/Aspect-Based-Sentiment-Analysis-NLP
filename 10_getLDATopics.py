import numpy as np
import pandas as pd
from pprint import pprint

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
from gensim import corpora, models

import unicodedata
import string
import re
# spacy for lemmatization
import spacy

from tkinter import Tk
from tkinter.filedialog import askopenfilename

def fn_splitWords(review):
    """
    Funtion to clean up the LDA output 
    """
    words = review.split('+')
    # remove punctuation characters
    alphabets = [char for char in words if char not in string.punctuation]

    # join each word and then split at spaces
    words_list = "".join(alphabets).split()

    # remove numbers
    words_list = [re.sub("(\\d|\\W)+","",word) for word in words_list]

    return words_list

# --- Code to set output column names correctly ----#
col_names = [] # col_names is a list that contains columns numbered from 0-9 to store each of the 10 topics
for i in range(7):
    col_names.append(i)
# -------------------------------------------------- #

def fn_formatTopics(df, col_names):
    """
    Funtion to format the LDA output correctly
    Input: dataframe with unformatted LDA topics
    Output: dataframe with formatted LDA topics
    """
    df = df.T # take transpose of the LDA output
    df.columns = col_names # set valid column names

    list1 = [] # store list of lists where each list is the list of words in each topic
    for i in range(7):
        list1.append(df[i][1])

    # zip each corrosponding word from each topic and then convert the zipped list into a DF
    df = pd.DataFrame(list(zip(list1[0],list1[1],list1[2],list1[3],list1[4],
                                 list1[5],list1[6])),columns=col_names)
    return df

def fn_tokenizeReview(review):
    """
    Funtion to tokenize reviews
    Input - String of review text
    Output - List of tokens in the review
    """
    review = str(review)
    review_tokens = review.split()
    return review_tokens

def fn_transformReviews(df):
    """
    Funtion to transform token_list to BOW and TF-IDF Corpuses

    Input - DataFrame with tokenized reviews

    Returns - 1. dictionary (ID representation for each unique token)
              2. BOW Corpus
              3. TF-IDF Corpus
    """
    # assign each word in the corpus a unique ID
    id2word = gensim.corpora.Dictionary(df['token_list'])

    # apply filters to remove specified portion of words
    id2word.filter_extremes(no_below=15, no_above=0.6, keep_n=100000)

    # convert each review document into a BOW representation based on the above created dictionary
    bow_corpus = [id2word.doc2bow(doc) for doc in df['token_list']]

    # from BOW corpus, create a tf-idf model
    tfidf = models.TfidfModel(bow_corpus)

    # transform the entire corpus with TF-IDF scores
    tfidf_corpus = tfidf[bow_corpus]

    return id2word, bow_corpus, tfidf_corpus

# ------ UNCOMMENT THIS SECTION TO FIND THE BEST NUMBER OF TOPICS ---------------#
"""
def fn_getNumOfTopics(dictionary, corpus, texts, limit, start=3, step=3):

    Compute c_v coherence for various number of topics

    Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    texts : List of input texts
    limit : Max num of topics

    Returns:
    -------
    best score and best number of topics for the corpus


    max_score= 0.0
    best_num_topics = 0
    for num_topic in range(start, limit, step):
        lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=num_topic, 
                                           random_state=100,
                                           update_every=1,
                                           chunksize=100,
                                           passes=10,
                                           alpha='auto',
                                           per_word_topics=True)
        coherence_model_lda = CoherenceModel(model=lda_model, texts=texts, 
                                                 dictionary=dictionary, coherence='c_v')
        coherence_score = coherence_model_lda.get_coherence()
        if(coherence_score>max_score):
            max_score = coherence_score
            best_num_topics = num_topic
    return max_score, best_num_topics
        
"""

# ---------- MAIN EXECUTION STARTS HERE ----------------#

Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
filename = askopenfilename()

# load the csv into a dataframe
df_reviews = pd.read_csv(filename, encoding='utf-8')

print("Analyzing topics for phrases. Please wait. This may take a few minutes...")

# tokenize the reviews with bigram and trigram phrases
df_reviews['token_list'] = df_reviews['phrased_summary2'].apply(fn_tokenizeReview)

# transform the tokens into BOW and TF-IDF Corpuses
id2word, bow_corpus, tfidf_corpus = fn_transformReviews(df_reviews)

# ---------- UNCOMMENT THIS FOR ACTUAL EXECUTION --------------------#
"""
# get best score and best number of topics
df_reviews = df_reviews[df_reviews['bigram_phrase']!='']
df_reviews = df_reviews[df_reviews['bigram_phrase']!='']
max_score, best_num_topics = fn_getNumOfTopics(dictionary=id2word, corpus=bow_corpus, texts=df_reviews['bigram_phrase'], limit=30, start=8, step=2)

print("The corpus has",best_num_topics,"topics with c_v coherence score =",max_score)

"""
# ------------------------------LDA for phrases---------------------------------#

# model with best_num_topics
corpus = tfidf_corpus # choose which corpus - tfidf or BOW?
best_lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=7, #best_num_topics, 
                                           random_state=100,
                                           update_every=1,
                                           chunksize=100,
                                           passes=10,
                                           alpha='auto',
                                           per_word_topics=True)

# get topics and put them in a dataframe
topic_list_bigrams = best_lda_model.print_topics()
df_topics_Phrases_tfidf = pd.DataFrame(topic_list_bigrams)
df_topics_Phrases_tfidf.columns = ['TopicNumber','Words']

# clean up the LDA output
df_topics_Phrases_tfidf['Words'] = df_topics_Phrases_tfidf['Words'].apply(fn_splitWords)

# format the output correctly
df_topics_Phrases_tfidf = fn_formatTopics(df_topics_Phrases_tfidf, col_names)

# LDA for BOW model
corpus = bow_corpus # choose which corpus - tfidf or BOW?
best_lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=7, #best_num_topics, 
                                           random_state=100,
                                           update_every=1,
                                           chunksize=100,
                                           passes=10,
                                           alpha='auto',
                                           per_word_topics=True)


# get topics and put them in a dataframe
topic_list_bigrams = best_lda_model.print_topics()
df_topics_Phrases_bow = pd.DataFrame(topic_list_bigrams)
df_topics_Phrases_bow.columns = ['TopicNumber','Words']

# clean up the LDA output
df_topics_Phrases_bow['Words'] = df_topics_Phrases_bow['Words'].apply(fn_splitWords)

# format output correctly
df_topics_Phrases_bow = fn_formatTopics(df_topics_Phrases_bow, col_names)
print("Topics have been analysed from bigram and trigram phrases.")



# -------------- LDA for Unigrams -------------------------------#
print("Analyzing topics from unigrams.\n Please wait..")
df_reviews = pd.read_csv(filename, encoding = 'utf-8')

# tokenize the reviews with unigrams
df_reviews['token_list'] = df_reviews['summary2'].apply(fn_tokenizeReview)

# transform the tokens into BOW and TF-IDF Corpuses
id2word, bow_corpus, tfidf_corpus = fn_transformReviews(df_reviews)

# ---------- UNCOMMENT THIS TO FIND THE BEST NUMBER  OF TOPICS --------------------#
"""
# get best score and best number of topics
max_score, best_num_topics = fn_getNumOfTopics(dictionary=id2word, corpus=bow_corpus, texts=df_reviews['normalized_review'], limit=30, start=8, step=2)

print("The corpus has",best_num_topics,"topics with c_v coherence score =",max_score)

"""
# LDA
# model with best_num_topics
corpus = tfidf_corpus # choose which corpus - tfidf or BOW?
best_lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=7, #best_num_topics, 
                                           random_state=100,
                                           update_every=1,
                                           chunksize=100,
                                           passes=10,
                                           alpha='auto',
                                           per_word_topics=True)


# get topics and put them in a dataframe
topic_list_unigrams = best_lda_model.print_topics()
df_topics_unigrams_tfidf = pd.DataFrame(topic_list_unigrams)
df_topics_unigrams_tfidf.columns = ['TopicNumber','Words']

# clean up LDA output
df_topics_unigrams_tfidf['Words'] = df_topics_unigrams_tfidf['Words'].apply(fn_splitWords)

# format output correctly
df_topics_unigrams_tfidf = fn_formatTopics(df_topics_unigrams_tfidf, col_names)

print("Topics have been analysed from unigram words.")

# LDA with BOW corpus for unigrams
corpus = bow_corpus # choose which corpus - tfidf or BOW?
best_lda_model_bow = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=7, #best_num_topics, 
                                           random_state=100,
                                           update_every=1,
                                           chunksize=100,
                                           passes=10,
                                           alpha='auto',
                                           per_word_topics=True)

# get topics and put them in a dataframe
topic_list_unigrams = best_lda_model.print_topics()
df_topics_unigrams_bow = pd.DataFrame(topic_list_unigrams)
df_topics_unigrams_bow.columns = ['TopicNumber','Words']

df_topics_unigrams_bow['Words'] = df_topics_unigrams_bow['Words'].apply(fn_splitWords)

# format output correctly
df_topics_unigrams_bow = fn_formatTopics(df_topics_unigrams_bow, col_names)

# Create a Pandas Excel writer using XlsxWriter as the engine.
writer = pd.ExcelWriter('LDATopicAnalysis.xlsx', engine='xlsxwriter')
df_topics_Phrases_tfidf.to_excel(writer, sheet_name='Phrases-TFIDF')
df_topics_unigrams_tfidf.to_excel(writer, sheet_name='Unigrams-TFIDF')
df_topics_Phrases_bow.to_excel(writer, sheet_name='Phrases-BOW')
df_topics_unigrams_bow.to_excel(writer, sheet_name='Unigrams-BOW')
print("File outputted: LDATopicAnalysis.xlsx")

writer.save()