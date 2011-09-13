# tweetrate.py:
# Checking out the hourly tweet rate hypothesis of Asur & Huberman, "Predicting
# the Future With Social Media."
#
# Based on Rob's birdfeeder code, minus the database.

import tweetstream
import datetime
import re

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
