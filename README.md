## Comparison of different algorthims with Twitter Sentiment Analysis

Sentiment analysis is the interpretation and classification of **emotions** (positive, negative and neutral) within text data using text analysis techniques. Sentiment analysis tools allow businesses to identify customer sentiment toward products, brands or services in online feedback.

### Need for Sentiment analysis

* Sentiment analysis tools allow businesses to identify customer sentiment toward products, brands or services in online feedback.
* Understanding people’s emotions is essential for businesses since customers are able to express their thoughts and feelings more openly than ever before.
* Using sentiment analysis to automatically analyze 4,000+ reviews about your product could help you discover if customers are happy about your pricing plans and customer service.
* Real-Time Analysis Sentiment analysis can identify critical issues in real-time

### Rule-Based Approach

A rule-based system uses a set of human-crafted rules to help identify subjectivity, polarity, or the subject of an opinion. These rules may include various techniques developed in computational linguistics, such as stemming, tokenization, part-of-speech tagging parsing and lexicons (i.e. lists of words and expressions).

### Sentiment Analysis Process

#### Step 1: Obtaining the Dataset
##### WEB SCRAPING
Web scraping, web harvesting, or web data extraction is data scraping used for extracting data from websites.

#### Step 2: Pre-Processing of data
##### REMOVING PUNCTUATIONS
##### TOKENIZATION
In order to use textual data for predictive modeling, the text must be parsed to remove certain words – this process is called tokenization.
##### STEMMING
Stemming is the process of reducing inflected words to their word stem, base or root form—generally a written word form.
##### LEMMATIZATION
Lemmatisation in linguistics is the process of grouping together the inflected forms of a word so they can be analysed as a single item, identified by the word's lemma, or dictionary form.
##### COUNT VECTORIZATION
Scikit-learn’s CountVectorizer is used to convert a collection of text documents to a vector of term/token counts.
##### TFIDF TRANSFORMER
Tf-idf transformers aim to convert a collection of raw documents to a matrix of TF-IDF features.

#### Step 3: Fitting and training the model
##### Implemented Algorithms
###### DECISION TREES
###### RANDOM FOREST CLASSIFIER
###### K-NEAREST NEIGHBOURS
###### LOGISTIC REGRESSION
###### SUPPORT VECTOR MACHINE

#### Step 4: Model prediction and result comparison
###### DECISION TREES
###### RANDOM FOREST CLASSIFIER
###### K-NEAREST NEIGHBOURS
###### LOGISTIC REGRESSION
###### SUPPORT VECTOR MACHINE

#### Conclusion

### Limitations of Sentiment Analysis
* One of the downsides of using lexicons is that people express emotions in different ways. Some may express sarcasm and irony in the statements.
* Multilingual sentiment analysis.
* Making the model automatic. Automatic methods, contrary to rule-based systems, don't rely on manually crafted rules, but on machine learning techniques. A sentiment analysis task is usually modeled as a classification problem, whereby a classifier is fed a text and returns a category
* Can take emoticons into account to predict better.
* Apart from the positive and negative category, the model could be developed to learn to classify tweets that are satircal or sarcastic.

### References
* [MonkeyLearn - Everything there is to know about Sentiment Analysis](https://monkeylearn.com/sentiment-analysis/#:~:text=Sentiment%20analysis%20is%20the%20interpretation,or%20services%20in%20online%20feedback.)
* Wikipedia [click here to donate](https://donate.wikimedia.org/w/index.php?title=Special:LandingPage&country=IN&uselang=en&utm_medium=sidebar&utm_source=donate&utm_campaign=C13_en.wikipedia.org)
