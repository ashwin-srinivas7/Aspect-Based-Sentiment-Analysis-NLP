import numpy
import pandas as pd
import re
import nltk
import unicodedata
import string
import unidecode

from nltk.corpus import stopwords
import nltk.data


from tkinter import Tk
from tkinter.filedialog import askopenfilename

from contractions import CONTRACTION_MAP

# construct list of stop words
stop_words = set(stopwords.words("english"))
words_to_include = ['not','didn', "didn't", 'doesn', "doesn't",'no']
words_to_remove = ['would','should','could','the','i','u','a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p',
                    'q','r','s','t','u','v','w','x','y','z','ive']

stop_words = [word for word in stop_words if word not in words_to_include]
stop_words = set(stop_words)
stop_words = list(stop_words.union(words_to_remove))

# tokenize review into sentences
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
def fn_getSentences(review):
    """
    Funtion to tokenizr reviews into sentences
    We are doing this as we want to tag each sentence with a theme
    Input: The entire review
    Output: List of sentences in the review
    """
    return re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s',review)

# function to get length of normalized reviews    
def fn_getNumOfWords(normalized_review):
    return len(normalized_review.split())

# function to clean the review_text
def fn_cleanText(review):

    # remove accents from words
    #review_clean = unicodedata.normalize('NFKD', review_clean).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    review = unidecode.unidecode(review)
    
    # remove punctuation characters
    alphabets_no_punct = [char for char in review if char not in string.punctuation]
    words_list = "".join(alphabets_no_punct).split() # join each word and then split at spaces

    # remove digits and special characters
    words_list = [re.sub("(\\d|\\W)+"," ",word) for word in words_list]
    
    # convert words to lower case
    words_list = [word.lower() for word in words_list]

    # remove stop words and convert to lower case an
    #review_clean = ' '.join([word for word in words_list if word not in stop_words])

    review_clean = ' '.join(words_list)
    # remove extra whitespace
    review_clean = re.sub('  +', ' ', review_clean)

    review_clean = str(review_clean)
    return review_clean

# ------------ MAIN EXECUTION STARTS HERE -------------------------#
Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
filename = askopenfilename()
df_reviews = pd.read_csv(filename, encoding='utf-8')

print("Tokenizing reviews into sentences. Please wait...")
# tokenize reviews into sentences
df_reviews['sentences'] = df_reviews['Review_Text'].apply(fn_getSentences)


# stack list of sentences into different rows
df_sentence_list = pd.DataFrame.from_records(df_reviews['sentences'].tolist()).stack().reset_index(level=1, drop=True).rename('sentences')
df_reviews = df_reviews.drop('sentences', axis=1).join(df_sentence_list).reset_index(drop=True)[['Rev_id', 'Review_Text','sentences','Final_Price_after_Discount', 'Category',
                                                                                                'Retailer','Brand','Review_Rating_Score','Product_Name','Product_URL']]

print("Tokenization complete!")

print("\nReviews are being normalized. Please wait. This may take a few minutes...")
# normalize reviews ans store it in a new column
df_reviews['normalized_review'] =  df_reviews['sentences'].apply(fn_cleanText)
df_reviews['length'] = df_reviews['normalized_review'].apply(fn_getNumOfWords)

df_reviews = df_reviews[df_reviews['length']>=2]

df_reviews = df_reviews.drop(columns=['length'])
# output the dataframe into a csv
# df_reviews = df_reviews.filter(['brand','sku_desc','review_text','normalized_review','rating'])
df_reviews = df_reviews[df_reviews['normalized_review']!='']
df_reviews.to_csv('normReviews.csv')
print("Review normalization complete! \nFile outputted: normReviews.csv")