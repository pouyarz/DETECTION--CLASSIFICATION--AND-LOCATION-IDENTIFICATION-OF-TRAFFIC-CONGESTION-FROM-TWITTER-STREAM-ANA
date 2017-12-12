import pandas
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVR
from sklearn import svm
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import f1_score, precision_recall_fscore_support, roc_auc_score
import sklearn
from sklearn.tree import DecisionTreeClassifier


test_samples = [
  '',
  'horriveeeeeel',
  'a concept the book',
  'blocked many book the',
  'themes of weinstein ',
  'I frown whenever I see you in a poor state of mind',
  'no other choice after he suckerpunched traffic cop he turn himself in when this video put him on blast ',
  'traffic source 01 traffic free real time traffic and leads world wide ',
  'there is heavy collision traffic',
'information about this coming friday closure of ri 95 north exit 4 traffic will be routed up route 3 north',
]

train_data = pandas.read_csv(open('test-train.csv', 'r'), quotechar='"')#,delimiter=',', encoding="utf-8-sig")#,
raw_tweets = train_data['tweets'].tolist() #.encode("utf-8")
train_labels= train_data['relevent'].tolist()
#raw_tweets = [x.encode('utf8') for x in raw_tweets]
#raw_tweets = raw_tweets[960:1225]

stopWords=['all', 'just', 'being', 'over', 'both', 'through', 'yourselves', 'its', 'o', 'hadn', 'herself', 'll', 'had', 'should',
           'only', 'won', 'ours', 'has', 'do', 'them', 'his', 'they', 'him', 'nor', 'd', 'did', 'didn', 'she', 'each', 'further',
           'where', 'few', 'because', 'doing', 'some', 'hasn', 'are', 'our', 'ourselves', 'out', 'what', 'under', 'not', 'now',
           'while', 're', 'does', 'above', 'mustn', 't', 'be', 'we', 'who', 'were', 'here', 'shouldn', 'hers', 'about', 'couldn',
           'against', 's', 'isn', 'or', 'own', 'yourself', 'mightn', 'wasn', 'your', 'her', 'their', 'aren', 'there', 'been',
           'whom', 'too', 'wouldn', 'themselves', 'weren', 'was', 'more', 'himself', 'that', 'but', 'don', 'than', 'those', 'he',
           'me', 'myself', 'ma', 'these', 'will', 'below', 'ain', 'can', 'theirs', 'my', 'and', 've', 'then', 'is', 'am', 'it',
           'doesn', 'an', 'as', 'itself', 'have', 'any', 'if', 'again', 'no', 'when', 'same', 'how', 'other', 'which', 'you',
           'shan', 'needn', 'haven', 'after', 'most', 'such', 'why', 'a', 'off', 'i', 'm', 'yours', 'so', 'y', 'the', 'having', 'once']
vectorizer = TfidfVectorizer(min_df=5,  #discard words appearing in less than 5 documents
                             max_df = 0.9,  #discard words appering in more than 80% of the documents
                             sublinear_tf=True, #True, use sublinear weighting
                             use_idf=True,  #True, enable IDF
    tokenizer=nltk.word_tokenize,
    stop_words=stopWords,
    max_features=10000,
    binary=True,
    ngram_range=(1, 2)
)
train_vectors = vectorizer.fit_transform(raw_tweets)

#train_labels = train_labels[960:1225]

cat = pandas.read_csv(open('train73-1.csv', 'r'), quotechar='"')
raw_tweets2 = cat['tweets'].tolist()
train_labels2= cat['relevent'].tolist()





#SVM
classifier_svm = svm.LinearSVC()#(C=0.1)
classifier_svm.fit(train_vectors, train_labels)
test_vectors = vectorizer.transform(raw_tweets2)
prediction_svm = classifier_svm.predict(test_vectors)

#NB
classifier_nb = BernoulliNB()#(C=0.1)
classifier_nb.fit(train_vectors, train_labels)
test_vectors = vectorizer.transform(raw_tweets2)
prediction_nb = classifier_nb.predict(test_vectors)

#DT
classifier_dt = DecisionTreeClassifier()
classifier_dt.fit(train_vectors, train_labels)
test_vectors = vectorizer.transform(raw_tweets2)
prediction_dt = classifier_dt.predict(test_vectors)

#RF
classifier_rf = RandomForestClassifier()
classifier_rf.fit(train_vectors, train_labels)
test_vectors = vectorizer.transform(raw_tweets2)
prediction_rf = classifier_rf.predict(test_vectors)

#SGD
classifier_sgd =  SGDClassifier()
classifier_sgd.fit(train_vectors, train_labels)
test_vectors = vectorizer.transform(raw_tweets2)
prediction_sgd = classifier_sgd.predict(test_vectors)


print "SVM"
score = metrics.accuracy_score(train_labels2, prediction_svm)
print("accuracy:   %0.3f" % score)
print(sklearn.metrics.classification_report(train_labels2, prediction_svm))

print""
print "SGD"
score = metrics.accuracy_score(train_labels2, prediction_sgd)
print("accuracy:   %0.3f" % score)
print(sklearn.metrics.classification_report(train_labels2, prediction_sgd))

print""
print "RF"
score = metrics.accuracy_score(train_labels2, prediction_rf)
print("accuracy:   %0.3f" % score)
print(sklearn.metrics.classification_report(train_labels2, prediction_rf))

print""
print "NB"
score = metrics.accuracy_score(train_labels2, prediction_nb)
print("accuracy:   %0.3f" % score)
print(sklearn.metrics.classification_report(train_labels2, prediction_nb))

print""
print "DT"
score = metrics.accuracy_score(train_labels2, prediction_dt)
print("accuracy:   %0.3f" % score)
print(sklearn.metrics.classification_report(train_labels2, prediction_dt))