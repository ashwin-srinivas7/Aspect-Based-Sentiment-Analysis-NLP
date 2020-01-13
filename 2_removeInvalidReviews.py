import pandas as pd
from tkinter import Tk
from tkinter.filedialog import askopenfilename

# read the csv file into a dataframe
Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
filename = askopenfilename()
df_reviews = pd.read_csv(filename, encoding='utf-8')

user_input1 = 'Y'
while(user_input1=='Y' or user_input1=='y'):
    invalid_review = input("Enter the review to be deleted")
    df_reviews = df_reviews[df_reviews['Review_Text']!=invalid_review]
    print("Review deleted!")
    user_input1 = input("Do you delete more reviews? Y/N?")

# save the final dataframe to csv
df_reviews.to_csv("finalReviews.csv", encoding = 'utf-8')
print("Number of rows after deleting: ", df_reviews.shape[0])
print("File saved to: finalReviews.csv")