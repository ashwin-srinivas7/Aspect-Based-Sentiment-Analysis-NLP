# Aspect-Based Sentiment Analysis: NLP

This project contains the code framework for extracting "topics" from large number of customer reviews, 
tag each review with the specified topics and then find the sentiment for topics in each review.

This will enable any retailer to quickly understand customer pain points mentioned in the reviews and take actions in order to provide better customer services.

#### Steps to build the dictionary


##### Steps – 

1.	Prepare the crawl file by adding Rev_ID column to uniquely identify each review

2.	Open Anaconda Navigator and launch VS Code

3.	If it’s your first time, Set workspace as folder with all python scripts. Otherwise, open existing workspace

4.	Execute scripts on VS Code Terminal in the following order: 


	1.	1_removeNullReviews.py – Removes NULL rows in Review_Text column

		Input – Prepared Crawl File

		Output - reviewWithoutNulls.csv


	2.	2_removeInvalidReviews.py – Removed rows with specified values of Review_Text (user input)

		Inputs

		-	reviewWithoutNulls.csv

		-	Review_Text of reviews that must be deleted

		Output - finalReviews.csv (File with invalid values in Review_Text removed)


	3.	3_cleanRatings.py – Removes rows with specified Rating values

		Inputs

		-	finalReviews.csv

		-	Rating values that must be deleted (User Input)

		Output - cleanRatings.csv


	4.	4_shortenReviews.py – Removes rows with Review_Text fewer than specified number of words (User Input)

		Inputs

		-	cleanRatings.csv

		-	Minimum number of words per review

		Output - shortenedReviews.csv


	5.	5_normalizeReviews.py – Performs RegEx based sentence tokenization and basic cleaning

		Inputs - shortenedReviews.csv

		Output - normReviews.csv


	6.	6_getSummary.py – Summarizes each sentence by using POS filtering

		Input - normReviews.csv

		Parameters - allowed_postags1, allowed_postags2 (list of Parts of speech to keep in every sentence)

		Output - reviewSummary.csv


	7.	7_getPhrase.py – Identifies phrases using co-occurrence 

		Input - reviewSummary.csv

		Parameters – threshold, min_count

		Output - reviewPhrases.csv


	8.	8_getTop100Phrases.py – Computes the top 100 words/phrases using Bag of Words and TF-IDF

		Input - reviewPhrases.csv

		Parameter – n (number of phrases/words)

		Output - TopNPhrases.xlsx


	9.	10_getLDATopics.py – Get list of words that identify “latent” topics

		Input - reviewPhrases.csv

		Parameter - num_topics (set number of topics). If this is changed, make changes in function fn_formatTopics accordingly

		Output - LDATopicAnalysis.xlsx



	10.	Using outputs TopNPhrases.xlsx & LDATopicAnalysis.xlsx, manually extract words from the outputs and build an initial mapping dictionary.
 
		Dictionary will contain 2 columns (Word & Theme)
	

	11.	11_getSimilarWords.py – Get list of similar words for list of words

		Input - Initial dictionary saved as a CSV file; 

		Parameters – size, window, min_count

		Output - SimilarWords.csv

	12.	Use VLOOKUP function on Excel to tag each similar word with the right theme from the initial dictionary. 
		Manually remove invalid entries by filtering through each entry in word column to get the final list of similar words

	13.	Append the final list of similar words to the initial dictionary to get the final dictionary



#### Steps to tag themes and get sentiment of every review-theme pair
	1.	Execute 12_getThemesSentiments.py – Tag themes to each review and get sentiment of every review-theme pair

		Inputs - reviewPhrases.csv & Final Dictionary CSV

		Outputs - BuzzOutput.csv



