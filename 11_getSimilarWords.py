import numpy as np
import pandas as pd

from tkinter import Tk
from tkinter.filedialog import askopenfilename


# Gensim
import gensim

def fn_tokenizeReview(review):
    """
    Funtion to tokenize reviews
    Input - String of review text
    Output - List of tokens in the review
    """
    review = str(review)
    review_tokens = review.split()
    return review_tokens

def fn_getSimilarWords(word, model):
    """
    Function to get similar words in a vector space
    Input: word, word2vec model object
    Output: List of similar 20 words in descending order of imporatances
    """
    similar_list = model.wv.most_similar(positive=word, topn=10) # Change topn to get how many ever words required
    words_list = []
    for entry in similar_list:
        words_list.append(entry[0])
    return words_list

print("Choose corpus file to train the Word2Vec model")
# read the csv file into a dataframe
Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
filename = askopenfilename()
df_phrases_corpus = pd.read_csv(filename, encoding='utf-8')

print("Input the dictionary csv")
Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
filename = askopenfilename()
df_dict = pd.read_csv(filename, encoding='utf-8')

print("Similar words are being analysed. Please wait...")
# tokenize the reviews
df_phrases_corpus['tokens'] = df_phrases_corpus['phrased_summary2'].apply(fn_tokenizeReview)
doc = df_phrases_corpus['tokens']

# train the word2vec model
word2vec_model = gensim.models.Word2Vec(doc,
                                        size=150,
                                        window=5,
                                        min_count=10,
                                        workers=10,
                                        iter=10)

df_dict['similar_list'] = df_dict['Word'].apply(fn_getSimilarWords, model=word2vec_model)

df_similar_words = pd.DataFrame(df_dict['similar_list'].tolist(), index=df_dict['Word']).stack().reset_index(level=1, drop=True).reset_index(name='similar_list')[['Word','similar_list']]

df_similar_words.to_csv('SimilarWords.csv')
print("Similar words have been analysed. File outputted: SimilarWords.csv")