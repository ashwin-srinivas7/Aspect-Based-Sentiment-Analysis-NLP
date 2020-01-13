import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from tkinter import Tk
from tkinter.filedialog import askopenfilename

def fn_getTopNPhrases(df):
    n = 100 # set number of phrases/words

    # df = df[df['summary3']!='']
    # Create a Pandas Excel writer using XlsxWriter as the engine.
    writer = pd.ExcelWriter('TopNPhrases.xlsx', engine='xlsxwriter')

    # Unigrams - TFIDF
    vec = CountVectorizer().fit(df['summary2'].values.astype('U'))
    bag_of_words = vec.transform(df['summary2'].values.astype('U'))
    tfidf_transformer_object = TfidfTransformer().fit(bag_of_words)
    tfidf_feature_matrix = tfidf_transformer_object.transform(bag_of_words)
    sum_words = tfidf_feature_matrix.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)

    # convert dictionary into DataFrame
    df_unigram_tfidf = pd.DataFrame(words_freq[:n])
    df_unigram_tfidf.columns = ['Words', 'TF-IDF Value']

    # Unigram - BOW
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)

    # convert dictionary into DataFrame
    df_unigram_bow = pd.DataFrame(words_freq[:n])
    df_unigram_bow.columns = ['Word', 'Frequency']
    

    # Phrases - TFIDF
    vec = CountVectorizer().fit(df['phrased_summary2'].values.astype('U'))
    bag_of_words = vec.transform(df['phrased_summary2'].values.astype('U'))
    tfidf_transformer_object = TfidfTransformer().fit(bag_of_words)
    tfidf_feature_matrix = tfidf_transformer_object.transform(bag_of_words)
    sum_words = tfidf_feature_matrix.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)

    # convert dictionary into DataFrame
    df_phrases_tfidf = pd.DataFrame(words_freq[:n])
    df_phrases_tfidf.columns = ['Words', 'TF-IDF Value']

    # Phrases - BOW
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)

    # convert dictionary into DataFrame
    df_phrases_bow = pd.DataFrame(words_freq[:n])
    df_phrases_bow.columns = ['Word', 'Frequency']

    # Write each dataframe to a different worksheet
    df_unigram_tfidf.to_excel(writer, sheet_name='Unigrams-TFIDF')
    df_unigram_bow.to_excel(writer, sheet_name='Unigram-BOW')
    df_phrases_tfidf.to_excel(writer, sheet_name='Phrases-TFIDF')
    df_phrases_bow.to_excel(writer, sheet_name='Phrases-BOW')

    print("Top 'n' Phrases have been analyzed.\nFile Outputted: TopNPhrases.xlsx")
    writer.save()
    return

# ------------------ MAIN EXECUTION STARTS HERE ----------------------- #
Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
filename = askopenfilename()
df_reviews = pd.read_csv(filename, encoding='utf-8')

print("Getting top n phrases. Please wait...")
fn_getTopNPhrases(df_reviews)