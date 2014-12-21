# -*- coding: utf-8 -*-
"""
MultinomialNB hw2 is modified for class project by Talha Oz
calculate likelihood ratio of each word in the training for 3 clusters !
"""

import re
import math
import sys
from collections import Counter
from collections import defaultdict
from collections import OrderedDict
import csv
from collections import namedtuple
#import preprocessing as pre
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.stem.lancaster import LancasterStemmer
from nltk import SnowballStemmer
#import variableNames

#Stem all words with stemmer of type, return encoded as "encoding"
def stemming(words_l, type="PorterStemmer", lang="english", encoding="utf8"):
	supported_stemmers = ["PorterStemmer","SnowballStemmer","LancasterStemmer","WordNetLemmatizer"]
	if type is False or type not in supported_stemmers:
		return words_l
	else:
		l = []
		if type == "PorterStemmer":
			stemmer = PorterStemmer()
			for word in words_l:
				l.append(stemmer.stem(word).encode(encoding))
		if type == "SnowballStemmer":
			stemmer = SnowballStemmer(lang)
			for word in words_l:
				l.append(stemmer.stem(word).encode(encoding))
		if type == "LancasterStemmer":
			stemmer = LancasterStemmer()
			for word in words_l:
				l.append(stemmer.stem(word).encode(encoding))
		if type == "WordNetLemmatizer": #TODO: context
			wnl = WordNetLemmatizer()
			for word in words_l:
				l.append(wnl.lemmatize(word).encode(encoding))
		return l

def remove_stopwords(l_words, lang='english'):
	"""
	Removing stopwords. Takes list of words, outputs list of words.
	"""
	l_stopwords = stopwords.words(lang)
	content = [w for w in l_words if w.lower() not in l_stopwords]
	return content

def preprocesstweet(d):
	return stemming(remove_stopwords(word_tokenize(" ".join(re.findall(r'\w+', d,flags = re.UNICODE | re.LOCALE)).lower())))

def read_csv_data(path):
	"""
	Reads CSV from given path and Return list of dict with Mapping
	"""
	data = csv.reader(open(path))
	# Read the column names from the first line of the file
	fields = data.next()
	data_lines = []
	for row in data:
		items = dict(zip(fields, row))
		items['tweet'] = preprocess(items['tweet'])
		data_lines.append(items)
	return data_lines

def getData(train_file,test_file):
	"""
	receives CSV filenames returns Dictionaries
	"""
	train = read_csv_data(train_file)
	test  = read_csv_data(test_file)
	return train,test

def getClasses(train,type='a'):
	"""
	return train.fieldnames[4:9],train.fieldnames[9:13],train.fieldnames[13:]
	"""
	y = ['s1','s2','s3','s4','s5','w1','w2','w3','w4','k1','k2','k3','k4','k5','k6','k7','k8','k9','k10','k11','k12','k13','k14','k15']
	if (type == 's') :
		return y[:5]	#4:9 labeles of sentiment
	if (type == 'w') :
		return y[5:9]	#9:13 labels of when
	if (type == 'k') :
		return y[9:]	#13: labels of kind 

	return ['s1','s2','s3','s4','s5','w1','w2','w3','w4','k1','k2','k3','k4','k5','k6','k7','k8','k9','k10','k11','k12','k13','k14','k15']

def EXTRACTVOCABULARY(traindoc):
	"""
	words are split by space, no preprocessing done yet
	"""
	a = Counter()
	for doc in traindoc:
		a.update(doc['tweet'])
	return a

def COUNTDOCSINCLASS(C, D):
	"""
	class not counted if it's weight is zero, 1 otherwise.
	"""
	a = Counter(C) #C is a list of class labels
	for doc in D:
		for lbl in C:
			a.update({lbl:int(math.ceil(float(doc[lbl])))})
	return a

def CONCATENATETEXTOFALLDOCSINCLASS(D,C):
    a = defaultdict(Counter)
    for doc in D:
    	for lbl in C:
    		if(float(doc[lbl])>0):
    			a[lbl].update(doc['tweet'])
    return a

def invertedindex(classText):
    """
    gets document by term matrix and returns term by class matrix
    similar to COUNTTOKENSOFTERM of TRAINMULTINOMIALNB(C,D)
    """
    superindex=defaultdict(Counter)
    for lbl,words in classText.iteritems():
        for word in words:
            superindex[word].update({lbl:words[word]})
    return superindex

def getCondProb(termbyclass,classText,C):
    """
    likelihood of a word to appear in each label
    """
    condprob = defaultdict(lambda: defaultdict(float))
    total={} #total number of words in that label
    for lbl,vals in classText.iteritems():
        total[lbl] = sum(vals.values())
    
    for word,cnt in termbyclass.iteritems():
        #add zero occurrences
        zeroes = [zeroc for zeroc in C if zeroc not in cnt.keys()]
        for k in zeroes:
            condprob[word][k]=1.0/(len(classText[k])+total[k])
        for k,v in cnt.iteritems():
            condprob[word][k]=float(v+1)/(len(classText[k])+total[k])
    return condprob

def TRAINMULTINOMIALNB(C,D):
	"""
	TRAINMULTINOMIALNB(C, D)
	V ← EXTRACTVOCABULARY(D)
	N ← COUNTDOCS(D)
	foreach c∈C
	    do Nc ← COUNTDOCSINCLASS(D, c)
	    prior[c] ← Nc/N
	    textc ← CONCATENATETEXTOFALLDOCSINCLASS(D, c)
	    for each t∈V
	        do Tct ← COUNTTOKENSOFTERM(textc, t)
	        for eacht∈V
	            do condprob[t][c] ←
	return V, prior, condprob
	"""
	prior={}
	V = EXTRACTVOCABULARY(D)
	classCount = COUNTDOCSINCLASS(C,D)
	classText = CONCATENATETEXTOFALLDOCSINCLASS(D,C)
	termbyclass = invertedindex(classText)
	condprob = getCondProb(termbyclass,classText,C)
	for c in C:
		prior[c] = classCount[c] / float(len(D))
	return V,prior,condprob

def APPLYMULTINOMIALNB(C, V, prior, condprob, d):
    """
    APPLYMULTINOMIALNB(C, V, prior, condprob, d)
    W ← EXTRACTTOKENSFROMDOC(V, d)
    foreach c∈C
    do score[c] ← log prior[c]
    for eacht∈W
    do score[c] += log cond prob[t][c]
    return arg maxc∈C score[c]
    """
    W = [word for word in d if word in V ] # W = EXTRACTTOKENSFROMDOC(V, d)
    score = {}
    for c in C:
        score[c] = math.log(prior[c])
        for t in W:
        	#print t,c
        	if condprob[t][c]:
        		score[c] += math.log(condprob[t][c]) 
    #print score
    return (max(C[:5],key = (lambda k: score[k])),
    max(C[5:9],key = (lambda k: score[k])),
    max(C[9:],key = (lambda k: score[k])))

def getTruesNB(C, V, prior, condprob, TEST):
	truepos = Counter()
	preds = OrderedDict()
	for d in TEST:
		ps,pw,pk = APPLYMULTINOMIALNB(C, V, prior, condprob, d['tweet'])
		#if int(d['id'])<100:
			#print d['id'],ps,pw,pk,[lbl+':'+d[lbl] for lbl in C if float(d[lbl])>0]
		preds.update({d['id']:(pw,ps,pk)})
	return preds

def submission(C,preds):
	fieldnames = ['id','s1','s2','s3','s4','s5','w1','w2','w3','w4','k1','k2','k3','k4','k5','k6','k7','k8','k9','k10','k11','k12','k13','k14','k15']
	findex = {}
	for i,lbl in enumerate(fieldnames):
		findex.update({lbl:i})
	f = open("submission.csv", "w")
	csvwriter = csv.writer(f)
	csvwriter.writerow(fieldnames)
	for k,vals in preds.iteritems():
		row = [0] * len(fieldnames)
		print k,
		row[0] = k
		for v in vals:
			row[findex[v]] = 1
		csvwriter.writerow(row)


if __name__ == '__main__':
	D, test = getData('train.csv','test.csv')
	C = getClasses(D,type='s')
	V,prior,condprob = TRAINMULTINOMIALNB(C,D)
	getTruesNB(C, V, prior, condprob, test) #TODO: CV instead of D !!!


