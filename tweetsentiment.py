# tweetsentiment.py
#
# Make a space for a domain, make positive and negative canonical
# documents, and figure out where tweets reside in this space.

import os
import luminoso2
import numpy
import re
from scikits.learn import svm

# make_luminoso_files:  convert my tweetDict to a directory full of
# files, as luminoso expects
#
# tweetDict is a dictionary from terms to lists of tweets, as created
# by tweetrate's logfile reader.  Separated out in case this should use
# a database at some point.  Luminoso expects docs all in different
# files; may be a deeper interface later so less space wasted.
#
# docDir should exist.
# filenames are subject_time, in case either needs recovery later
def make_luminoso_files(tweetDict, docDir):
    for key, tweetlist in tweetDict.iteritems():
        for (time, text) in tweetlist:
            file = open(docDir + '/' + key + '_' + time + '.txt', 'w')
            file.write(text)
            file.close()

def make_luminoso_space(docDir, studyDir):
    if not os.path.isdir(studyDir):
        model = luminoso2.make_common_sense(studyDir, 'en')
    else:
        model = luminoso2.load(studyDir)
    model.learn_from(docDir,studyDir)
    return model

posMovieText = 'Awesome fabulous loved must see great excited going'
negMovieText = 'Boring terrible fucking horrible sucked wasted'

def make_canonical_vectors(model, posText, negText):
    posVec = model.vector_from_text(posText)
    negVec = model.vector_from_text(negText)
    return (normalize(posVec), normalize(negVec))

def norm(vec):
    return numpy.sqrt(numpy.dot(vec,vec))

def normalize(vec):
    # / in numpy is elementwise divide, matlab's ./
    return vec/norm(vec)

# Fixing this so we find the angle to posVec - negVec
# ...hey, this ain't bad
def get_sentiment_score(model, posVec, negVec, text):
    vec = normalize(model.vector_from_text(text))
    sentimentVec = normalize(posVec - negVec)
    sentimentScore = numpy.dot(vec, sentimentVec)
    return sentimentScore

# topic_replace:  Replace the topic of interest with TOPIC throughout
# the tweetdict.  This is common practice in sentiment analysis.
# Should precede labeling (or creating the space, for that matter)
def topic_replace(tweetDict):
    newDict = {}
    for topic, tweetlist in tweetDict.iteritems():
        newtweetlist = []
        for tweet in tweetlist:
            newText = re.sub(topic.lower(),'TOPIC',tweet[1].lower())
            newTuple = tweet[0],newText
            newtweetlist.append(newTuple)
        newDict[topic] = newtweetlist
    return newDict


def make_svm(model, labeledTweetDict):
    vector_list = []
    answer_list = []
    classifier = svm.SVC()
    for topic, tweetlist in labeledTweetDict.iteritems():
        for tweet in tweetlist:
            vector_list.append(model.vector_from_text(tweet[1]))
            answer_list.append(sentToNumber(tweet[2]))
    data = numpy.array(vector_list)
    target = numpy.array(answer_list)
    classifier.fit(data,target)
    return classifier

def svm_classify(model, svm, tweetText):
    vector = model.vector_from_text(tweetText)
    print vector
    prediction = svm.predict(numpy.array(vector))
    return prediction

def sentToNumber(sentClass):
    if sentClass == 'pos':
        return 1.0
    if sentClass == 'neg':
        return -1.0
    return 0

def classifyTweets(tweetDict, model, posVec, negVec):
    newDict = {}  # Could make this more efficient later by modding same structure
    for topic, tweetlist in tweetDict.iteritems():
        newTweetlist = []
        for tweetTuple in tweetlist:
            tweetText = tweetTuple[1]  # Todo:  make this a structure
            classification = classifyTweet(tweetText, model, posVec, negVec)
            newTuple = (tweetTuple[0], tweetText,classification)
            newTweetlist.append(newTuple)
        newDict[topic] = newTweetlist
    return newDict

def classifyTweet(tweetText, model, posVec, negVec):
    pos, neg = get_pos_and_neg_scores(model, posVec, negVec, tweetText)
    if pos > neg:
        return 'pos'
    return 'neg'

# Count classifications:  Returns dict of topic:{pos:count, neg:count}
def countClassifications(tweetDict):
    sentimentDict = {}
    for topic, classifiedList in tweetDict.iteritems():
        posCount = 0
        negCount = 0
        for classifiedTweet in classifiedList:
            if classifiedTweet[2] == 'pos':
                posCount = posCount + 1
            elif classifiedTweet[2] == 'neg':
                negCount = negCount + 1
        thisDict = {'pos':posCount, 'neg':negCount}
        sentimentDict[topic] = thisDict
    return sentimentDict

# Get ground truth:  interactively label a set of tweets for sentiment
# 0 = neutral, p = positive, n = negative
# skip -- go to next topic
# stop -- stop
# Optional second parameter picks up at first unlabeled topic
def getGroundTruth(tweetDict, newDict = {}):
    labelShort = ''
    for topic in tweetDict:
        if topic in newDict: continue # skip past what we were adding last time
        newList = []
        for tweet in tweetDict[topic]:
            labelShort = ''
            while (not (labelShort == 'n' or labelShort == 'p' or labelShort == 'stop' or labelShort == 'skip' or labelShort == '0')):
                labelShort = raw_input(tweet[1])
            if labelShort == 'n':
                label = 'neg'
            elif labelShort == 'p':
                label = 'pos'
            elif labelShort == '0':
                label = 'neutral'
            if labelShort == 'skip' or labelShort == 'stop': break
            newTuple = tweet[0],tweet[1],label
            newList.append(newTuple)
        newDict[topic] = newList
        if labelShort == 'stop': break
    return newDict

                

