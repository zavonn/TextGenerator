# CONTEXT SPECIFIC TEXT GENERATOR


### This project utilizes over 25000 articles scraped from over 60 news sites to build a text generator that takes a user-defined title as input and outputs a 200 word (or any arbitrary number) length 'article.'

The 'Final Cap Clean Process' Jupyter Notebook contains the first section of the process that involves aggregating a list of English language sources using the NEWSAPI package, filtering the list of sources, and then cleaning and reducing the 25,000 articles to 5,000 articles with word counts between 100 and 500 words. 

The next stage clusters the cleaned article titles using a TFIDF vectorizer, UMAP dimension reduction, and finally KMEANS to create 10 clusters. After the clusters have been created, each cluster is trimmed or bootstrapped to a uniform length of 400 articles per cluster. 

The last section of the notebook utilizes the article contents within each cluster to built a LSTM Neural Network that intakes a sequence of four words and predicts the next word. Each cluster requires a separate model, which due to the resource intensive nature of training neural networks requires the use of a SageMaker Notebook. Amazon's SageMaker platform allows for the use of high RAM, GPU accelerated computing instances to cut down the total training time of all ten models to 24hours or less. Within the SageMaker Notebook, all of the models and their related content (Vocabulary Dictionaries, etc) are saved to an S3 bucket for easy retrieval.

The next Jupyter Notebook, 'Article Generation' contains all of the processes involved in utilized the trained models to generate text based on the user inputted 'Title.' Generating content involves intaking a user defined Title, cleaning the text, and assigning the Title to a cluster. Once the user input has been assigned a cluster, the related model is loaded and used to generate text of a user defined length (200 words in the example). 

The final notebook is the most condensed and loads all of the required formulas for content generation from the TextGenerate python file. Within three lines of code we can now run the process of obtaining input, assigning a cluster, and outputting text.

In its current state some of the outputted text is nonsensical; although the vocabulary used is context specific, however, tweaking the clustering by reducing the total number of clusters to 5 and increasing the total article count, which would increase both the vocabulary and wordlist, would almost necessarily improve performance. 

This project in its current state provides an almost entirely complete blueprint that can be adjusted slightly to produce an interesting context specific text generator that is useful for creating original content.
































```python

```
