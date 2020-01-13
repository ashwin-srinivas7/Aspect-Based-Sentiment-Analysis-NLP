import numpy
import pandas as pd
import spacy
from lemmatize_withPOS import fn_lemmatizeText

from tkinter import Tk
from tkinter.filedialog import askopenfilename


allowed_postags1 = ['NOUN','ADJ','VERB','ADV','CONJ','ADP','PRON','PART','INTJ','X'] # for sentiment analysis
allowed_postags2 = ['NOUN','ADJ'] # for LDA
#allowed_postags3 = ['NOUN','VERB']

nlp = spacy.load('en')

Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
filename = askopenfilename()
df_reviews = pd.read_csv(filename, encoding='utf-8')

print("Generating review summaries. Please wait...")
# df_reviews['summary1'] = df_reviews['normalized_review'].apply(fn_lemmatizeText, allowed_postags=allowed_postags1, nlp=nlp)
df_reviews['summary1'] = df_reviews['normalized_review'].apply(fn_lemmatizeText, allowed_postags=allowed_postags1, nlp=nlp)
df_reviews['summary2'] = df_reviews['normalized_review'].apply(fn_lemmatizeText, allowed_postags=allowed_postags2, nlp=nlp)
#df_reviews['summary3'] = df_reviews['normalized_review'].apply(fn_lemmatizeText, allowed_postags=allowed_postags3, nlp=nlp)

df_reviews.to_csv('reviewSummary.csv')
print("Review Summarization complete.\nSummary1 - ['NOUN','ADJ','VERB','ADV','CONJ','ADP','PRON','PART','INTJ','X'] \nSummary2 - ['NOUN','ADJ']")