# Imports
import pandas as pd
from tkinter import Tk
from tkinter.filedialog import askopenfilename

import string


# read the csv file into a dataframe
Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
filename = askopenfilename()
df_reviews = pd.read_csv(filename, encoding='utf-8')

n_words = int(input("Enter the minimum number of words in reviews"))

# remove punctuations to get tokens and count
def fn_getNumWordsInReview(review):
    # remove punctuations
    txt_no_punct = [char for char in review if char not in string.punctuation]
    txt_no_punct = "".join(txt_no_punct).split()
    return len(txt_no_punct)

df_reviews = df_reviews.astype({"Review_Text": str})
df_reviews['num_words_in_review'] = df_reviews['Review_Text'].apply(fn_getNumWordsInReview)

# remove rows with less than 'n_words'
df_reviews = df_reviews[df_reviews['num_words_in_review']>=n_words]
print("Reviews with less than",n_words,"words removed!")

print("Number of rows after deleting: ", df_reviews.shape[0])

df_reviews.to_csv('RemoveShortReviews.csv', encoding='utf-8')
print("File saved to: shortenedReviews.csv")
