import redis
import tweepy
import csv
import json
import string
import re
import operator
import nltk
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import bigrams
import logging
from string import punctuation
import io
import codecs
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
import pandas
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn import svm
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier

#Redis pipeline
r = redis.StrictRedis(host='localhost', port=6379, db=0)
p = r.pubsub()
p.subscribe('tweet')


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(module)s - %(funcName)s: %(message)s',
                    datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger()
logger.setLevel(logging.INFO)


#'before','to','for','between','by','on','of','into','down','from','until','with','in','up','at'
stopWords=['during', 'this', 'very', 'all', 'just', 'being', 'over', 'both', 'through', 'yourselves', 'its', 'o', 'hadn', 'herself', 'll', 'had', 'should', 'only', 'won', 'ours', 'has', 'do', 'them', 'his', 'they', 'him', 'nor', 'd', 'did', 'didn', 'she', 'each', 'further', 'where', 'few', 'because', 'doing', 'some', 'hasn', 'are', 'our', 'ourselves', 'out', 'what', 'under', 'not', 'now', 'while', 're', 'does', 'above', 'mustn', 't', 'be', 'we', 'who', 'were', 'here', 'shouldn', 'hers', 'about', 'couldn', 'against', 's', 'isn', 'or', 'own', 'yourself', 'mightn', 'wasn', 'your', 'her', 'their', 'aren', 'there', 'been', 'whom', 'too', 'wouldn', 'themselves', 'weren', 'was', 'more', 'himself', 'that', 'but', 'don', 'than', 'those', 'he', 'me', 'myself', 'ma', 'these', 'will', 'below', 'ain', 'can', 'theirs', 'my', 'and', 've', 'then', 'is', 'am', 'it', 'doesn', 'an', 'as', 'itself', 'have', 'any', 'if', 'again', 'no', 'when', 'same', 'how', 'other', 'which', 'you', 'shan', 'needn', 'haven', 'after', 'most', 'such', 'why', 'a', 'off', 'i', 'm', 'yours', 'so', 'y', 'the', 'having', 'once']

#Relevant Tweets
train_data = pandas.read_csv(open('train73.csv', 'r'), quotechar='"')#,delimiter=',', encoding="utf-8-sig")#,
raw_tweets = train_data['tweets'].tolist()
train_labels = train_data['relevent'].tolist()

vectorizer = TfidfVectorizer(min_df=5,  #discard words appearing in less than 5 documents
                             max_df = 0.9,  #discard words appering in more than 90% of the documents
                             sublinear_tf = True, #True, use sublinear weighting
                             use_idf = True,  #True, enable IDF
    tokenizer = nltk.word_tokenize,
    stop_words = stopWords,
    max_features = 5000,
    binary = True,
    ngram_range = (1, 2)
)
train_vectors = vectorizer.fit_transform(raw_tweets)


#Categorizing
cat = pandas.read_csv(open('categories1.csv', 'r'), quotechar='"')
raw_tweets2 = cat['tweets'].tolist()
train_labels2= cat['categories'].tolist()

vectorizer2 = TfidfVectorizer(min_df = 5,
                             max_df = 0.9,
                             sublinear_tf = True,
                             use_idf = True,
    tokenizer = nltk.word_tokenize,
    stop_words = stopWords,
    max_features = 5000,
    binary = True,
    ngram_range = (1, 2)
)
train_vectors2 = vectorizer2.fit_transform(raw_tweets2)


#NB
classifier_nb = BernoulliNB() #(binarize=0.0)
classifier_nb.fit(train_vectors, train_labels)
#RF
classifier_rf = RandomForestClassifier()
classifier_rf.fit(train_vectors, train_labels)
#SGD
classifier_sgd = SGDClassifier()
classifier_sgd.fit(train_vectors, train_labels)
#SVM
classifier_svm = svm.LinearSVC()#(C=0.1)
classifier_svm.fit(train_vectors, train_labels)
#Decision Tree
classifier_dt = DecisionTreeClassifier()#(max_depth=5)
classifier_dt.fit(train_vectors, train_labels)

#Categories
#SVM
classifier_svm2 = svm.LinearSVC()#(C=0.1)
classifier_svm2.fit(train_vectors2, train_labels2)
#NB
classifier_nb2 = BernoulliNB() #(binarize=0.0)
classifier_nb2.fit(train_vectors2, train_labels2)
#RF
classifier_rf2 = RandomForestClassifier()
classifier_rf2.fit(train_vectors2, train_labels2)
#SGD
classifier_sgd2 = SGDClassifier()
classifier_sgd2.fit(train_vectors2, train_labels2)
#Decision Tree
classifier_dt2 = DecisionTreeClassifier()#(max_depth=5)
classifier_dt2.fit(train_vectors2, train_labels2)





while True:
    message = p.get_message()
    if message:

        tweet = json.loads(str(message['data']))
        geoo = ''
        tweetsss = ''

        texts = ''
        try:
            #if not tweet['retweeted'] and 'RT' not in tweet['text']:          #removing retweets
                tweetsss = tweet
                texts = tweet['text'] #[tweet['text']]
                #texts = texts.encode('utf-8', 'replace')
                id = tweet['id']
                geo = tweet['geo']
                geoo= geo



        except BaseException as e:
            logger.warning(e)
        #return True

        def on_error(self, status):
            logger.warning(status)
            return True


        emoticons_str = r"""
                (?:
                    [:=;] # Eyes
                    [oO\-]? # Nose (optional)
                    [D\)\]\(\]/\\OpP] # Mouth
                )"""
        regex_str = [
            emoticons_str,
            r'<[^>]+>',  # HTML tags
            r'(?:@[\w_]+)',  # @-mentions
            r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)",  # hash-tags
            r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+',  # URLs
            r'(?:(?:\d+,?)+(?:\.?\d+)?)',  # numbers
            r"(?:[a-z][a-z'\-_]+[a-z])",  # words with - and '
            r'(?:[\w_]+)',  # other words
            r'(?:\S)'  # anything else
        ]
        tokens_re = re.compile(r'(' + '|'.join(regex_str) + ')', re.VERBOSE | re.IGNORECASE)
        emoticon_re = re.compile(r'^' + emoticons_str + '$', re.VERBOSE | re.IGNORECASE)
        def tokenize(s):
            return tokens_re.findall(s)
        def preprocess(s, lowercase=False):
            tokens = tokenize(s)
            if lowercase:
                tokens = [token if emoticon_re.search(token) else token.lower() for token in tokens]
            return tokens
        def processTweet(twt):
            # process the tweets


            twt = twt.lower()             # Convert to lower case
            twt = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', ' ', twt)  # remove www.* or https?://*
            twt = re.sub(r'http([^\s]+)', r'\1', twt) #remove http
            twt = re.sub(r" hwy", " highway", twt)  # replace hwy to highway
            twt = re.sub(r" rd ", " road ", twt)  # replace rd to road
            twt = re.sub(r" st ", " street ", twt)
            twt = re.sub(r" trfc ", " traffic ", twt)  # trfc hwy to traffic
            twt = re.sub('@[^\s]+', ' ', twt)  # remove @username
            twt = re.sub(r'#([^\s]+)', r'\1', twt)  # replace #word with word
            twt = twt.strip('\'"')  # trim
            twt = ''.join(c for c in twt if c not in punctuation) # removing punctuation
            twt = re.sub(r"@", "at ", twt)  # replace @ to at
            #pattern = re.compile(r'\b(' + r'|'.join(stopwords.words('english')) + r')\b\s*')
            #twt = pattern.sub('', twt)
            #twt = ' '.join([word for word in twt.split() if word not in (stopwords.words('english'))])
            twt = re.sub(r"rt ", "", twt)
            emoji_pattern = re.compile(
                u"(\ud83d[\ude00-\ude4f])|"  # emoticons
                u"(\ud83c[\udf00-\uffff])|"  # symbols & pictographs
                u"(\ud83d[\u0000-\uddff])|"
                u"(\ud83d[\ude80-\udeff])|"  # transport & map symbols
                u"(\ud83c[\udde0-\uddff])"  # flags (iOS)
                "+", flags=re.UNICODE)
            twt = emoji_pattern.sub(r'', twt)  # no emoji
            myre = re.compile(u'('
                              u'\ud83c[\udf00-\udfff]|'
                              u'\ud83d[\udc00-\ude4f\ude80-\udeff]|'
                              u'[\u2600-\u26FF\u2700-\u27BF])+',
                              re.UNICODE)
            twt = myre.sub(r'', twt)
            twt = re.sub('[\s]+', ' ', twt)  # remove additional white spaces

            return twt

        outtweets = (processTweet(texts.encode("utf-8")))
        #outtweets = unicode(outtweets, 'utf8')


        print (outtweets)

        test_vectors = vectorizer.transform([outtweets])

        prediction_nb = classifier_nb.predict(test_vectors)
        prediction_rf = classifier_rf.predict(test_vectors)
        prediction_sgd = classifier_sgd.predict(test_vectors)
        prediction_svm = classifier_svm.predict(test_vectors)
        prediction_dt = classifier_dt.predict(test_vectors)


        #if prediction_nb==1 or prediction_svm==1 or prediction_nb==1 or prediction_rf==1 or prediction_sgd:
        test_vectors2 = vectorizer2.transform([outtweets])

        prediction_nb2 = classifier_nb2.predict(test_vectors2)
        prediction_rf2 = classifier_rf2.predict(test_vectors2)
        prediction_sgd2 = classifier_sgd2.predict(test_vectors2)
        prediction_svm2 = classifier_svm2.predict(test_vectors2)
        prediction_dt2 = classifier_dt2.predict(test_vectors2)


        #NB
        with open('NaiveBayes.txt', 'a') as f:
        #    f.write(str(id))
         #   f.write(',')
            #f.write(str(geoo))
            #f.write(',')
            f.write (str(outtweets))#(unicode(str(outtweets), "utf-8"))
            f.write(',')
            f.write(str(prediction_nb) + '\n')
        if prediction_nb == 1:
            with open('NaiveBayes-R.txt', 'a') as f:
                f.write(str(outtweets))
                f.write(',')
                f.write(str(prediction_nb) + '\n')
            with open('NaiveBayes-C.txt', 'a') as f:
                f.write(str(outtweets))
                f.write(',')
                f.write(str(prediction_nb2) + '\n')
        if geoo != None and prediction_nb == 1:
            with open('NaiveBayes-G.txt', 'a') as f:
                f.write(str(outtweets))
                f.write(',')
                f.write(str(geoo) + '\n')

        # RF
        with open('RandomForest.txt', 'a') as f:
            f.write(str(outtweets))
            f.write(',')
            f.write(str(prediction_rf) + '\n')
        if prediction_rf == 1:
            with open('RandomForest-R.txt', 'a') as f:
                f.write(str(outtweets))
                f.write(',')
                f.write(str(prediction_rf) + '\n')
            with open('RandomForest-C.txt', 'a') as f:
                f.write(str(outtweets))
                f.write(',')
                f.write(str(prediction_rf2) + '\n')
        if geoo != None and prediction_rf == 1:
            with open('RandomForest-G.txt', 'a') as f:
                f.write(str(outtweets))
                f.write(',')
                f.write(str(geoo) + '\n')

        # SGD
        with open('SGD.txt', 'a') as f:
            f.write(str(outtweets))
            f.write(',')
            f.write(str(prediction_sgd) + '\n')
        if prediction_sgd == 1:
            with open('SGD-R.txt', 'a') as f:
                f.write(str(outtweets))
                f.write(',')
                f.write(str(prediction_sgd) + '\n')
            with open('SGD-C.txt', 'a') as f:
                f.write(str(outtweets))
                f.write(',')
                f.write(str(prediction_sgd2) + '\n')
        if  geoo != None and prediction_sgd == 1:
            with open('SGD-G.txt', 'a') as f:
                f.write(str(outtweets))
                f.write(',')
                f.write(str(geoo) + '\n')

        # DT
        with open('DecisionTree.txt', 'a') as f:
            f.write(str(outtweets))
            f.write(',')
            f.write(str(prediction_dt) + '\n')
        if prediction_dt == 1:
            with open('DecisionTree-R.txt', 'a') as f:
                f.write(str(outtweets))
                f.write(',')
                f.write(str(prediction_dt) + '\n')
            with open('DecisionTree-C.txt', 'a') as f:
                f.write(str(outtweets))
                f.write(',')
                f.write(str(prediction_dt2) + '\n')
        if geoo != None and prediction_dt == 1:
            with open('DecisionTree-G.txt', 'a') as f:
                f.write(str(outtweets))
                f.write(',')
                f.write(str(geoo) + '\n')

        # SVM
        with open('SupportVector.txt', 'a') as f:
            f.write(str(outtweets))
            f.write(',')
            f.write(str(prediction_svm) + '\n')
        if prediction_svm == 1:
            with open('SupportVector-R.txt', 'a') as f:
                f.write(str(outtweets))
                f.write(',')
                f.write(str(prediction_svm) + '\n')
            with open('SupportVector-C.txt', 'a') as f:
                f.write(str(outtweets))
                f.write(',')
                f.write(str(prediction_svm2) + '\n')
        if geoo != None and prediction_svm == 1:
            with open('SupportVector-G.txt', 'a') as f:
                f.write(str(outtweets))
                f.write(',')
                f.write(str(geoo) + '\n')
