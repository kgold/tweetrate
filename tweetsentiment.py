# tweetsentiment.py
#
# Make a space for a domain, make positive and negative canonical
# documents, and figure out where tweets reside in this space.

import os
import luminoso2
import numpy
import re
import json
import math
from scikits.learn import svm
# GNB does not work -- avoid, avoid!
from scikits.learn.naive_bayes import GNB

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
            try:
                file.write(text)
            except UnicodeEncodeError:
                continue
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
def get_sentiment_vec(model, posText, negText):
    (posVec, negVec) = make_canonical_vectors(model,posText,negText)
    sentimentVec = normalize(posVec - negVec)
    return sentimentVec

def get_sentiment_score(model, sentimentVec, text):
    vec = normalize(model.vector_from_text(text))
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
            if len(tweet) < 3:
                newTuple = tweet[0],newText
            else:
                newTuple = tweet[0],newText,tweet[2]
            newtweetlist.append(newTuple)
        newDict[topic] = newtweetlist
    return newDict

# Turn a dataset into the format scikits expects
# ... now including scaling standardization
def make_dataset(model, labeledTweetDict):
    vector_list = []
    answer_list = []
    for topic, tweetlist in labeledTweetDict.iteritems():
        for tweet in tweetlist:
            vector_list.append(model.vector_from_text(tweet[1]))
            answer_list.append(sentToNumber(tweet[2]))
    data = numpy.array(vector_list)
    target = numpy.array(answer_list)
    means, devs = get_scaling_info(data)
    data = scale(data, means, devs)
    return (data, target, means, devs)

# SVM is probably failing because the data isn't scaled.  Scaling to
# mean 0, variance 1.
def get_scaling_info(datamatrix):
    columnMeans = numpy.mean(datamatrix,axis=0)
    columnDevs = numpy.std(datamatrix, axis=0)
    return columnMeans, columnDevs

# Scale the data given the column means and standard deviations
# Must be done for SVM input in train & classification
def scale(datamatrix, columnMeans, columnDevs):
    scaledMatrix = numpy.zeros((datamatrix.shape))
    if datamatrix.ndim < 2:
        for i in range(datamatrix.size):
            scaledMatrix[i] = datamatrix[i] - columnMeans[i]
            scaledMatrix[i] = scaledMatrix[i] / columnDevs[i]
    else:
        for row in range(datamatrix.shape[0]):
            scaledMatrix[row,:] = (datamatrix[row,:] - columnMeans)/columnDevs
    return scaledMatrix

# Trying Gaussian Naive Bayes so we get an interpretable number
# (a confidence) -- also has fewer params
def make_gnb(model, labeledTweetDict):
    classifier = GNB()
    data, target = make_dataset(model,labeledTweetDict)
    classifier.fit(data,target)
    return classifier

# C: governs tradeoff between overfitting and underfitting.
# C = 10 worked reasonably for the movies.
def make_svm(model, labeledTweetDict, C):
    classifier = svm.SVC(C)
    (data, target, means, devs) = make_dataset(model, labeledTweetDict)
    classifier.fit(data,target)
    return classifier, means, devs

def classify(model, classifier, tweetText, means, devs):
    vector = numpy.array(model.vector_from_text(tweetText))
    scaledVector = scale(vector, means, devs)
    prediction = classifier.predict(scaledVector)
    return prediction

def classify_all(model, classifier, tweetDict, means, devs):
    newDict = {}
    for topic, tweetlist in tweetDict.iteritems():
        newTweetlist = []
        for tweetTuple in tweetlist:
            tweetText = tweetTuple[1]
            prediction = classify(model, classifier, tweetText, means, devs)
            if prediction == sentToNumber('pos'):
                predictionText = 'pos'
            elif prediction == sentToNumber('neg'):
                predictionText = 'neg'
            else:
                predictionText = 'neutral'
            newTuple = (tweetTuple[0], tweetText, predictionText)
            newTweetlist.append(newTuple)
        newDict[topic] = newTweetlist
    return newDict

def sentToNumber(sentClass):
    if sentClass == 'pos':
        return 2
    if sentClass == 'neg':
        return 0
    return 1

def classifyTweets(tweetDict, model, sentimentVec, thresh):
    newDict = {}  # Could make this more efficient later by modding same structure
    for topic, tweetlist in tweetDict.iteritems():
        newTweetlist = []
        for tweetTuple in tweetlist:
            tweetText = tweetTuple[1]  # Todo:  make this a structure
            classification = classifyTweet(tweetText, model, sentimentVec, thresh)
            newTuple = (tweetTuple[0], tweetText,classification)
            newTweetlist.append(newTuple)
        newDict[topic] = newTweetlist
    return newDict

def classifyTweet(tweetText, model, sentimentVec, thresh):
    score = get_sentiment_score(model, sentimentVec, tweetText)
    # This method has no way to decide a tweet is irrelevant...
    # so maybe we need the SVM after all
    if score > thresh:
        return 'pos'
    elif score < -thresh:
        return 'neg'
    return 'neutral'

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
            while (not (labelShort == 'n' or labelShort == 'p' or labelShort == 'stop' or labelShort == 'skip' or labelShort == '0' or labelShort == 'a')):
                try:
                    labelShort = raw_input(tweet[1])
                except UnicodeEncodeError:
                    labelShort = '0' # if we can't print it, chuck it
            if labelShort == 'n':
                label = 'neg'
            elif labelShort == 'p':
                label = 'pos'
            elif labelShort == '0':
                label = 'neutral'
            elif labelShort == 'a':
                label = 'anticipatory' # Common in best buy, seems useful
            if labelShort == 'skip' or labelShort == 'stop': break
            newTuple = tweet[0],tweet[1],label
            newList.append(newTuple)
        newDict[topic] = newList
        if labelShort == 'stop': break
    return newDict

# Reads the Best Buy format and puts the stuff in a format our code expects
def getTweetDictFromJson(filename, topicname):
    jsonThing = json.load(open(filename,'r'))
    myTweetList = []
    for item in jsonThing:
        if 'timestamp' in item:
            # Cramming ID in here next to timestamp
            newTuple = str(item['timestamp'])+'\t'+item['name'],item['text']
        else:
            newTuple = '0\t' + item['name'], item['text']
        myTweetList.append(newTuple)    
    # Let's just throw this all in the same topic.
    # In case of best buy, will let us replace their name
    myTweetDict = {topicname:myTweetList}
    return myTweetDict

def getExamples(labeledDict, label):
    exampleList = []
    for topic, tweetlist in labeledDict.iteritems():
        for tweet in tweetlist:
            if tweet[2] == label:
                exampleList.append(tweet)
    return exampleList

# Makes tuples in format Rob expects, write to json
def makeTuples(labeledDict, filename):
    tupleList = []
    for topic, tweetlist in labeledDict.iteritems():
        for tweet in tweetlist:
            # Unpacking the timestamp\tmessageID
            parts = tweet[0].split('\t')
            newTuple = tweet[1],float(parts[0]),tweet[2],parts[1]
            tupleList.append(newTuple)
    myFile = open(filename + '.json', 'w')
    jsonString = json.dumps(tupleList, myFile)
    myFile.write(jsonString)
    myFile.close()
    return tupleList

# Look for time of best change using max likelihood
# Slow but it will do the job
def bestChangeTime(labeledTweetDict, minTime, maxTime):
    timeList = []
    sentValueList = []
    for topic,tweetlist in labeledTweetDict.iteritems():
        for tweet in tweetlist:
            parts = tweet[0].split('\t') # break off ID if it's there
            time = float(parts[0]) 
            if time > minTime and time < maxTime: # 0 was the value for "no timestamp"
                timeList.append(time)
                sentValue = sentToNumber(tweet[2])
                sentValueList.append(sentValue)
    bestLogLikelihood = -1000000000000
    bestTime = 0
    pCounts, nCounts, neuCounts = countSents(sentValueList)
    for i in range(len(sentValueList)-1):
        pCounts1 = pCounts[i+1]
        nCounts1 = nCounts[i+1]
        neuCounts1 = neuCounts[i+1]
        pCounts2 = pCounts[len(sentValueList)-1] - pCounts1
        nCounts2 = nCounts[len(sentValueList)-1] - nCounts1
        neuCounts2 = neuCounts[len(sentValueList)-1] - neuCounts1
        firstPartLikelihood = getLikelihood(pCounts1, nCounts1, neuCounts1)
        secondPartLikelihood = getLikelihood(pCounts2,nCounts2, neuCounts2)
        likelihood = firstPartLikelihood + secondPartLikelihood
        if likelihood > bestLogLikelihood:
            bestLogLikelihood = likelihood
            bestTime = timeList[i+1]
    return bestTime

def countSents(valueArray):
    pCounts  = [0]
    nCounts = [0]
    neuCounts = [0]
    for i in range(len(valueArray)):
        if valueArray[i] == sentToNumber('pos'):
            pCounts.append(pCounts[i] + 1)
            nCounts.append(nCounts[i])
            neuCounts.append(neuCounts[i])
        elif valueArray[i] == sentToNumber('neg'):
            pCounts.append(pCounts[i])
            nCounts.append(nCounts[i] + 1)
            neuCounts.append(neuCounts[i])
        else:
            pCounts.append(pCounts[i])
            nCounts.append(nCounts[i])
            neuCounts.append(neuCounts[i] + 1)
    return pCounts, nCounts, neuCounts

# include pseudocounts of 1
def getLikelihood(p, n, neu):
    pseudoTotal = p + n + neu + 3
    pProb = (p + 1.0) / pseudoTotal
    nProb = (n + 1.0) / pseudoTotal
    neuProb = (neu + 1.0) / pseudoTotal
    likelihood = math.log(pProb) * p + math.log(nProb) * n + math.log(neuProb) * neu
    return likelihood
