# tweetrate.py:
# Checking out the hourly tweet rate hypothesis of Asur & Huberman, "Predicting
# the Future With Social Media."
#
# Based on Rob's birdfeeder code, minus the database.

import tweetstream
import datetime
import re
import email.utils
import math
import numpy

# When using track_rates, create a map from item of interest to search
# terms, like so.
myMovieDict = {'Contagion':['Contagion','#Contagion'],'Bucky Larson':['Bucky Larson','#BuckyLarson']}

# But that's a little tedious, so we can try automating it a little.
# tooCommon is a list of words we wouldn't want to search for.
# In the far-flung future, we could automate that decision.
def initTopicDict(topics, tooCommon):
    myDict = {}
    for topic in topics:
        termList = []
        if topic not in tooCommon:
            termList.append(topic)
        termList.append('#' + topic.replace(' ',''))
        myDict[topic] = termList
    return myDict

# Here's the list of movies opening this weekend (9/8)
movies = ['Contagion','Bucky Larson','Circumstance','Creature','Littlerock','Love Crime','Puzzle','Shaolin','Warrior']

# And the ones that are probably too common as-is
tooCommon = ['Circumstance','Creature','Puzzle','Warrior']

# 9/16
movies2 = ['One Fall','Jane\'s Journey','Weird World of Blowfly','Afternoons With Margueritte', 'Restless','Straw Dogs','Drive','I Don\'t Know How She Does It']

tooCommon2 = ['Restless','Drive']

# For sentiment analysis, we need a twitter-specific corpus where the dimensions that
# fall out aren't the movies themselves.  So we'll look at tweets
# that include some elements of interest, such that the primary
# axis that falls out should be sentiment.  We'll blend with common sense, too.
# (Trying to pick words here that are least likely to be used sarcastically...)
sentiment = ['winning', 'fail', 'awesome', 'terrible', 'sweet', 'sucks', 'miserable', 'excellent', 'excited', 'eager', 'stupid', 'boring', 'offensive', 'beautiful', 'must see']

# Eh, never mind.  Makes more sense to do topic-specific search on general
# terms (did 'movie' and 'film').  Sentiment is bound to be one of the top
# axes to fall out.  (saving corpus to generalMovies.txt)
# It's likely that only a small fraction of movies are opening weekend
# movies, in this age of Netflix etc.  Likewise for any general category:
# most things are not new things.

# topicDict maps topics of interest to their search terms.
# logfilename is a file that will record the topic, timestamp, and text
# in tab-delimited format.  This could be changed to a database.
def track_hits(topicDict, username, password, logfilename):
    # We actually need to create a reverse dictionary too, for
    # looking up which key was a hit.
    searchterms = []
    reverseDict = {}
    for k, v in topicDict.iteritems():
        searchterms.extend(v)
        for term in v:
            reverseDict[term] = k
    stream = topic_stream(searchterms,username,password)
    hitsDict = {}
    spamSet = set([])
    for tweet in stream:
        if 'text' in tweet:  # tweets are dicts; this is the text entry
            tweetText = tweet['text']
            [spam, spamSet] = spam_hash(tweetText,spamSet)
            if spam:
                print 'Spam detected:' + tweetText
                continue
            print 'Relevant tweet:' + tweetText
            # Strip newlines so we can do simple tab-delimited logfiles
            tweetText = tweetText.replace('\n','')
            for term in searchterms:
                if tweetText.lower().rfind(term.lower()) > -1:
                    topic = reverseDict[term]
                    mylogfile = open(logfilename,'ab')
                    # created_at is time of tweet
                    try:
                        mylogfile.write(topic + '\t' + tweet['created_at'] + '\t' + tweetText + '\n')
                    except UnicodeEncodeError:  # damn you, tweeters
                        mylogfile.write(topic + '\t' + tweet['created_at'] + '\t' + '[text unwriteable]\n')
                    mylogfile.close()
                    if topic in hitsDict:
                        hitsDict[topic] = hitsDict[topic]+1
                    else:
                        hitsDict[topic] = 1
                    statusString = ''
                    for topic in topicDict.keys():
                        if topic in hitsDict:
                            statusString = statusString + topic + ':' + str(hitsDict[topic])
                    print statusString
                    break

def topic_stream(words, username, password):
    return tweetstream.FilterStream(username, password, track=words)

# spam_hash:  Detect spamming of the same text.  The examples I've seen
# tend to be exactly the same minus URLs.
#
# returns True if the item has been seen before, and updates the dict
# if it hasn't.
def spam_hash(string, spamSet):
    urlLess = re.sub(r'http://t.co/\w+','', string)
    if urlLess in spamSet:
        return [True, spamSet]
    spamSet = spamSet | set([urlLess])
    return [False, spamSet]

# read_logfile:  Returns dict from topic to list of (time, text).
# Time is string for now, but will probably parse to datetime later
def read_logfile(filename):
    myDict = {}
    spamSet = set([])
    myFile = open(filename,'r')
    for line in myFile.readlines():
        parts = line.split('\t')
        if len(parts) < 3: continue # Oops, tweets can have newlines -- stripping these in subsequent iterations
        topic = parts[0]
        time = parts[1]
        tweetText = parts[2]
        [spam, spamSet] = spam_hash(tweetText,spamSet)
        if spam: continue
        if topic in myDict:
            myDict[topic].append((time,tweetText))
        else:
            myDict[topic] = [(time,tweetText)]
    return myDict

# Total tweet count gives us average tweet rate if we're looking over
# the same period of time
# 
# tweetDict: a dict of the form returned by read_logfile
# returns dict from topic to count
def count_tweets(tweetDict):
    countDict = {}
    for topic, tweets in tweetDict.iteritems():
        countDict[topic] = len(tweets)
    return countDict

# Convert twitter timestamp into datetime object.
def timestamp_convert(timestring):
    myDate = email.utils.parsedate(timestring)
    return datetime.datetime(myDate[0],myDate[1],myDate[2],myDate[3],myDate[4],myDate[5],myDate[6])

# Calculate the hourly tweetrate for each topic, using our tweetDict.
# It may also make sense to calculate the hourly tweetrate by sentiment;
# break the dict down into separate dicts for pos, neg, and neutral first,
# and call this individually.
#
# The new dict is from topic to a list of (datetime, rate) tuples.
#
# This starts at exactly the time of the first tweet --
# could do a pass to remove the minutes & seconds before calling this 
# if that behavior is undesired.
def get_hourly_tweetrate_offline(tweetDict):
    rateDict = {}
    oneHour = datetime.timedelta(0,0,0,0,0,1) # hours is 6th 
    for topic, tweetlist in tweetDict.iteritems():
        print tweetlist[0][0]
        # Some tweets are missing timestamps; we will
        # just jettison these rather than assume falsely when they appeared,
        # on the assumption that the drop rate is independent
        try:
            lastHour = timestamp_convert(tweetlist[0][0])
        except:
            continue
        tweetsInLastHour = 0
        rateList = []
        for tweet in tweetlist:
            tweetDatetime = timestamp_convert(tweet[0])
            if tweetDatetime - lastHour < oneHour:
                tweetsInLastHour = tweetsInLastHour + 1
            else:
                newTuple = lastHour,tweetsInLastHour
                rateList.append(newTuple)
                lastHour = lastHour + oneHour
                # Insert 0's for hours with no tweets
                while tweetDatetime - lastHour > oneHour:
                    newTuple = lastHour, 0
                    rateList.append(newTuple)
                    lastHour = lastHour + oneHour
                tweetsInLastHour = 1
        newTuple = lastHour, tweetsInLastHour
        rateList.append(newTuple)
        rateDict[topic] = rateList
    return rateDict

# Change detection in the tweetrate.  We assume the tweetrate is
# normally distributed when nothing is happening, and look for the maximum
# likelihood breakpoint, recurring as desired.
#
# This gets called on a particular topic for a tweetDict, and wants a list of
# (datetime, rate) tuples.  Like its discrete counterpart in tweetsentiment.py,
# it returns both a time and a likelihood gain.  It pretty much ignores
# the timestamps, assuming that these were created on the hour (by
# the function above), so you could theoretically stuff some other info
# in there if you felt like it.
#
# Not optimized -- calculation of the normal parameters is very redundant.
def change_detect_offline(rateList):
    bestChangepoint = 0
    bestGain = -100000000000000
    allVals = numpy.array([rateEntry[1] for rateEntry in rateList])
    totalLoglike = computeLoglike(allVals)
    # Require at least 2 points on each side so we don't get 0-variance nonsense
    for changepoint in range(2,len(rateList)-2):
        dist1Vals = numpy.array([rateList[i][1] for i in range(changepoint)])
        dist2Vals = numpy.array([rateList[i][1] for i in range(changepoint,len(rateList))])
        loglike1 = computeLoglike(dist1Vals)
        loglike2 = computeLoglike(dist2Vals)
        gain = abs(loglike1 + loglike2 - totalLoglike) 
        if gain > bestGain:
            bestGain = gain
            bestChangepoint = changepoint
    return bestChangepoint, bestGain

# Helper fn computing likelihood of data assuming it is all drawn from same Gaussian
def computeLoglike(distVals):
    distMean = numpy.mean(distVals)
    distDev = numpy.std(distVals)
    likelihoods = [normpdf(val, distMean, distDev) for val in distVals]
    loglikes = numpy.array([math.log(like) for like in likelihoods])
    return numpy.sum(loglikes)

# I don't see this fn anywhere, so we're nabbing this from
# http://telliott99.blogspot.com/2010/02/plotting-normal-distribution-with.html
# and assuming it's numerically ok
def normpdf(x, mu, sigma):
    z = 1.0*(x-mu)/sigma
    e = math.e**(-0.5*z**2)
    C = math.sqrt(2*math.pi)*sigma
    return 1.0*e/C

# Detect all changepoints with likelihood gain above a threshold -- not necessarily
# in sorted order.  The threshold doesn't have intuitive units -- just play with it
#
# Does, however, require the rateList input in sorted order
def recursiveChangeDetect(rateList, thresh, changepointList=None):
    if changepointList is None:
        changepointList = []
    bestChange, gain = change_detect_offline(rateList)
    if gain > thresh:
        changepointList.append(rateList[bestChange][0]) # datetime
        firstRateList = [rateList[i] for i in range(bestChange)]
        secondRateList = [rateList[i] for i in range(bestChange,len(rateList))]
        # Python hands off the list rather than copying it, so we don't
        # need to assign here
        recursiveChangeDetect(firstRateList,thresh,changepointList)
        recursiveChangeDetect(secondRateList,thresh,changepointList)
    return changepointList
