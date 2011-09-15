# tweetsentiment.py
#
# Make a space for a domain, make positive and negative canonical
# documents, and figure out where tweets reside in this space.

import os
import luminoso2
import numpy

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
        model = luminoso2.load(studyDir, 'en')
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

def get_pos_and_neg_scores(model, posVec, negVec, text):
    vec = normalize(model.vector_from_text(text))
    posScore = numpy.dot(vec, posVec)
    negScore = numpy.dot(vec, negVec)
    return posScore, negScore

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
