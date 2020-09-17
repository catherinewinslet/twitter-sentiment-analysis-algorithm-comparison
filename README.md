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
Web scraping, web harvesting, or web data extraction is data scraping used for extracting data from websites. Using twitter OAuthHandler, obtain tweets from given date to current about any particular hashtag. Save the dataset to dataset directory as test_tweets.csv

#### Step 2: Pre-Processing of data
##### REMOVING PUNCTUATIONS
Using regular expression(regex), remove punctuation, hashtags and @-mentions from each tweet.</br>
```train_data['processed_tweets'] = train_data['tweet'].str.replace('[^A-Za-z0-9 ]', '')```

##### TOKENIZATION
In order to use textual data for predictive modeling, the text must be parsed to remove certain words – this process is called tokenization.</br>
```train_data['processed_tweets'] = train_data['processed_tweets'].apply(lambda x: x.split())```

##### STEMMING
Stemming is the process of reducing inflected words to their word stem, base or root form—generally a written word form.</br>
```
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english")
train_data['processed_tweets']= train_data['processed_tweets'].apply(lambda x: [stemmer.stem(i) for i in x])
```

##### LEMMATIZATION
Lemmatisation in linguistics is the process of grouping together the inflected forms of a word so they can be analysed as a single item, identified by the word's lemma, or dictionary form.</br>
```
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
stopwords = nltk.corpus.stopwords.words('english')

###replaces [running, ran, run] with run
```

##### COUNT VECTORIZATION
Scikit-learn’s CountVectorizer is used to convert a collection of text documents to a vector of term/token counts.</br>
```count_vect = CountVectorizer(stop_words='english')```

##### TFIDF TRANSFORMER
Tf-idf transformers aim to convert a collection of raw documents to a matrix of TF-IDF features.</br>
```transformer = TfidfTransformer(norm='l2',sublinear_tf=True)```

#### Step 3: Fitting and training the model
##### Implemented Algorithms
###### DECISION TREES
A decision tree is a flowchart-like structure in which each internal node represents a "test" on an attribute (e.g. whether a coin flip comes up heads or tails), each branch represents the outcome of the test, and each leaf node represents a class label (decision taken after computing all attributes).
###### RANDOM FOREST CLASSIFIER
Random forests or random decision forests are an ensemble learning method for classification, regression and other tasks that operate by constructing a multitude of decision trees at training time and outputting the class that is the mode of the classes or mean prediction of the individual trees.
###### K-NEAREST NEIGHBOURS
In pattern recognition, the k-nearest neighbors algorithm is a non-parametric method proposed by Thomas Cover used for classification and regression.
###### LOGISTIC REGRESSION
Logistic regression is a statistical model that in its basic form uses a logistic function to model a binary dependent variable, although many more complex extensions exist. In regression analysis, logistic regression (or logit regression) is estimating the parameters of a logistic model (a form of binary regression).
###### SUPPORT VECTOR MACHINE
In machine learning, support-vector machines are supervised learning models with associated learning algorithms that analyze data used for classification and regression analysis.

#### Step 4: Model prediction and result comparison
###### DECISION TREES
```
accuracy_score(y_test,predDT)
output: 0.94
```
###### RANDOM FOREST CLASSIFIER
```
accuracy_score(y_test,predRF)
output: 0.96
```
###### K-NEAREST NEIGHBOURS
```
accuracy_score(y_test,predKNN)
output: 0.93
```
###### LOGISTIC REGRESSION
```
accuracy_score(y_test,predLR)
output: 0.94
```
###### SUPPORT VECTOR MACHINE
```
accuracy_score(y_test,predSVM)
output: 0.95
```
#### Conclusion
Among all other techniques used, Random Forest Classifier has performed best with the highest accuracy. One reason why RF works well is because the algorithm can look past and handle the missing values in the tweets.

### Limitations of Sentiment Analysis
* One of the downsides of using lexicons is that people express emotions in different ways. Some may express sarcasm and irony in the statements.
* Multilingual sentiment analysis.
* Making the model automatic. Automatic methods, contrary to rule-based systems, don't rely on manually crafted rules, but on machine learning techniques. A sentiment analysis task is usually modeled as a classification problem, whereby a classifier is fed a text and returns a category
* Can take emoticons into account to predict better.
* Apart from the positive and negative category, the model could be developed to learn to classify tweets that are satircal or sarcastic.

### References
* [MonkeyLearn - Everything there is to know about Sentiment Analysis](https://monkeylearn.com/sentiment-analysis/#:~:text=Sentiment%20analysis%20is%20the%20interpretation,or%20services%20in%20online%20feedback.)
* Wikipedia [click here to donate](https://donate.wikimedia.org/w/index.php?title=Special:LandingPage&country=IN&uselang=en&utm_medium=sidebar&utm_source=donate&utm_campaign=C13_en.wikipedia.org)
