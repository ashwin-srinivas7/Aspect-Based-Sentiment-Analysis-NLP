import pandas as pd
from tkinter import Tk
from tkinter.filedialog import askopenfilename

# read the csv file into a dataframe
Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
filename = askopenfilename()
df_reviews = pd.read_csv(filename, encoding='utf-8')

# show the value counts for the ratings column
print("Rating values in the dataset:")
print(df_reviews['rating'].value_counts())

user_input1 = str(input("Do you want to delete any rating values? Y/N ?"))

while(user_input1=='Y' or user_input1=='y'):
    rating_remove = input("Enter the rating value to be deleted")
    df_reviews = df_reviews[df_reviews['rating']!=rating_remove]
    print(rating_remove,"removed!")
    user_input1 = input("Do you want to continue deleting values? Y/N ?")

# convert the ratings to float
df_reviews['rating']=df_reviews['rating'].astype(float)

print("Number of rows after deleting:",df_reviews.shape[0])

# save the dataframe into CSV
df_reviews = df_reviews.filter(['brand','sku_desc','review_text','rating'])
df_reviews.to_csv('cleanRatings.csv')
print("File outputted: cleanRatings.csv")