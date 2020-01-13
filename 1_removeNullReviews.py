import pandas as pd
from tkinter import Tk
from tkinter.filedialog import askopenfilename

Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
filename = askopenfilename()
df_ceilingFanData = pd.read_csv(filename, encoding='utf-8')

# filter relevant columns
df_ceilingFanData = df_ceilingFanData.filter(['Rev_ID', 'Review_Text','Final_Price_after_Discount', 'Category',
                         'Retailer','Brand','Review_Rating_Score','Product_Name','Product_URL','Review_Creation_Date'])

# print the number of rows
print("Number of rows before removing null reviews: ", df_ceilingFanData.shape[0])
# remove null reviews
def fn_removeNullReviews(df):
    # Check for null values in the review_text column
    if(df['Review_Text'].isnull().sum()>0):
        print(df['Review_Text'].isna().sum(), "null values found in \"Review_Text\". Deleting these rows...")
        df = df.dropna(subset=['Review_Text'])
        print("Deleted!")
        print(len(df),"rows present after null reviews have been deleted")
    else:
        print("No null values found in the Review_Text column")
    return df

df_ceilingFanData = fn_removeNullReviews(df_ceilingFanData)


# save dataframe to CSV and return
df_ceilingFanData.to_csv('reviewWithoutNulls.csv')
print("Data saved to: reviewWithoutNulls.csv")