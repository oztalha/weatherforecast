"""
author Talha Oz
this is the main code implemented for cs780 class project
"""
# -*- coding: utf-8 -*-

import pandas as p
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from sklearn import cross_validation
import preprocessing as pre
from sklearn.naive_bayes import MultinomialNB
from sklearn.decomposition import PCA
from sklearn.multiclass import OneVsRestClassifier
from sklearn import linear_model
import matplotlib.pyplot as plt
from variableNames import *
import scipy.sparse
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import SelectKBest, chi2
"""
from sklearn.svm import SVC
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import Perceptron
from sklearn import svm
from sklearn.preprocessing import LabelBinarizer
import matplotlib.pyplot as plt
from scipy import sparse
from sklearn import decomposition
from sklearn.ensemble import ExtraTreesRegressor
import sklearn.decomposition as deco
import argparse
from sklearn.svm import SVC
%autoreload 2
"""


def plotClasses(y):
	"""
	each class is counted by its weight, not # of nonzero occurrences
	"""
	fig = plt.figure()
	ax = plt.subplot(1,1,1)
	x1 = range(y.shape[1])
	y1 = [sum(y[:,a]) for a in range(y.shape[1])]
	width = 0.8
	labels = "s1,s2,s3,s4,s5,w1,w2,w3,w4,k1,k2,k3,k4,k5,k6,k7,k8,k9,k10,k11,k12,k13,k14,k15".split(',')
	if y.shape[1] == 5:
		labels = labels[:5]
	elif y.shape[1] == 4:
		labels = labels[5:9]
	else:
		labels = labels[9:]

	plt.xticks(np.arange(y.shape[1])+width/2,labels)
	legendkeys = tuple([k for k,v in legend.items() if k in labels])
	legendvalues= tuple([v for k,v in legend.items() if k in labels])
	[ax.bar(X,Y,width=width,label=k+' '+v) for X,Y,k,v in zip(x1,y1,legendkeys,legendvalues)]
	# Shink current axis by 20%
	box = ax.get_position()
	ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
	# Put a legend to the right of the current axis
	ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
	#ax.legend(('1','2'),('1a','2a'))#legendkeys,legendvalues
	plt.show()


def vectorize(y,train,t2,model,kbest=0,is_tfidf=True,is_sparse=True,max_features=None,is_cv=False,perlabel=False,min_df=5,is_nominal=False,is_normal=False,is_LSA=False,scale=False):
	
	if is_cv:
		X_train, X_test, y_train, y_true = cross_validation.train_test_split(train['tweet'], y, test_size=.20, random_state = 0)
		# X_train1, X_test1, y_train, y_true = cross_validation.train_test_split(train['tweet'], y, test_size=.20, random_state = 0)
		# X_train2, X_test2, y_train, y_true = cross_validation.train_test_split(train['state'], y, test_size=.20, random_state = 0)
		# X_train3, X_test3, y_train, y_true = cross_validation.train_test_split(train['location'], y, test_size=.20, random_state = 0)
		# X_train = np.hstack((X_train1,X_train2,X_train3))
		# X_test = np.hstack((X_test1,X_test2,X_test3))
	else:
		X_train = train['tweet']
		X_test = t2['tweet']
		y_train = y
	
	# if (y_train.shape[1] > 6):
	# 	model = linear_model.Ridge (alpha = 3.0, normalize = False)

	# if is_PCA:
	# 	modelPCA = PCA(n_components='mle')
	# 	model.fit(X_train)
	if is_tfidf:
		#tfidf = TfidfVectorizer(max_features=max_features, strip_accents='unicode', analyzer='word', smooth_idf=True,sublinear_tf=True,max_df=0.5,min_df=min_df,ngram_range=(1,2),use_idf=True)
		tfidf = TfidfVectorizer(max_features=max_features,strip_accents='unicode', analyzer='word', smooth_idf=True,sublinear_tf=True,max_df=0.5,min_df=5,ngram_range=(1,2),use_idf=True)
		#tfidf.fit(np.hstack((X_train,X_test))) #fitting on the whole data resulted in a worse mse score
		tfidf.fit(X_train)
		X_train = tfidf.transform(X_train)
		X_test = tfidf.transform(X_test)
		if is_LSA:
			LSA = TruncatedSVD(n_components=10000, algorithm='randomized', n_iter=5, random_state=0, tol=0.0)
			X_train = LSA.fit_transform(X_train)
			X_test = LSA.transform(X_test)
	else:
		vectorizer = CountVectorizer( binary = True )
		X_train = vectorizer.fit_transform(X_train)
		X_test = vectorizer.transform(X_test)

	if is_nominal:
		if (y_train.shape[1] < 16):
			y_rest = y_train.copy()
			X_train_list = []
			y_weight_list = []
			y_train_list = []
			for i in range(y_rest.shape[1]):
				X_train_list.append(X_train) # repeat X to matchup
				y_weight_list.append(np.apply_along_axis(lambda a: a.max(), 1, y_rest)) # get the maximum in y_rest
				y_train_list.append(np.apply_along_axis(lambda a: a.argmax(), 1, y_rest).astype(int)) # get the position of the maximum in y_rest
				y_rest = np.apply_along_axis(lambda a: [0 if i == a.argmax() else x for i,x in enumerate(a)], 1, y_rest) #set maximum to zero
				# y_weight = np.concatenate((y_weight, np.apply_along_axis(lambda a: a.max(), 1, y_rest)))
				# y_train = np.concatenate((y_train, np.apply_along_axis(lambda a: a.argmax(), 1, y_rest).astype(int)))
				# y_train = np.apply_along_axis(lambda a: [np.floor(x) if x != a.max() else 1 for x in a], 1, y_train).astype(bool)
			not_kind = True
			X_train = scipy.sparse.vstack(X_train_list)
			y_train = np.concatenate(y_train_list)
			y_weight = np.concatenate(y_weight_list)
		else:
			not_kind = False
			#y_train = np.apply_along_axis(lambda a: [np.floor(x) if i != a.argmax() else 1 for i,x in enumerate(a)], 1, y_train).astype(bool)
		#y_train = np.ceil(y_train).astype(bool)
		#y_weight = y_train.copy()

	if perlabel:
		test_prediction=[]
		for i in range(y_train.shape[1]):
			if is_nominal:
				model.fit(X_train,y_train[:,i]) #sample_weight=y_weight[:,i]
				pred = model.predict_proba(X_test)
				# pred = model.predict_log_proba(X_test) # for log in SGDRegressor
				print pred.shape
				test_prediction.append(pred)
			else:
				model.fit(X_train,y_train[:,i])
				test_prediction.append(model.predict(X_test))
		pred = np.array(test_prediction).T

	if kbest:
		ch2 = SelectKBest(chi2,kbest,k=1000)
		#yb = y_train
		yb = np.around(y_train).astype(bool)
		X_train = ch2.fit_transform(X_train, yb)
		X_test  = ch2.transform(X_test)
	
	if not is_sparse:
		X_train = X_train.toarray()
		X_test = X_test.toarray()
		#nmf = decomposition.NMF(n_components=y_train.shape[1]).fit(tfidf)
		#cca = CCA(n_components=100)
		#X_train = cca.fit_transform(X_train)
		#X_test = cca.transform(X_test)

	if not perlabel:
		if is_nominal and not_kind:
			model.fit(X_train, y_train,sample_weight=y_weight)
			pred = model.predict_proba(X_test)
			#model.fit(X_train.toarray(), y_train.toarray(),sample_weight=y_weight)
			#pred = model.predict_proba(X_test.toarray())
			# model.fit(scipy.sparse.csr_matrix(X_train), scipy.sparse.csr_matrix(y_train),sample_weight=y_weight) # added tocsr() !!!
			# pred = model.predict_proba(scipy.sparse.csr_matrix(X_test))
			#model.fit(scipy.sparse.csr_matrix(X_train), y_train,sample_weight=y_weight) #perceptron
			#pred = model.predict_proba(scipy.sparse.csr_matrix(X_test))
		else:
			model.fit(X_train, y_train)
			pred = model.predict(X_test)

	if scale:
		if (y_train.shape[1] < 6):
			pred = np.apply_along_axis(lambda a: a/(np.max(a)-np.min(a)),1,pred)

	if is_normal and (y_train.shape[1] < 6):
		#pred[pred < 0.1] = 0.0
		#pred[pred > 0.9] = 1.0
		row_sums = pred.sum(axis=1)
		pred = pred / row_sums[:, np.newaxis]

	pred = np.around(pred,3)
	pred = pred.clip(0,1)

	if is_cv:
		return pred,y_true
	else:
		return pred

def cv_loop(train, t2, model, is_sparse=True,kbest=0,is_class=False,is_tfidf=True,max_features=20000,perlabel=False,min_df=5,is_nominal=False,is_normal=False,is_LSA=False,scale=False):
	y = np.array(train.ix[:,4:])
	ys = y[:,:5]#4:9 labeles of sentiment
	yw = y[:,5:9]#9:13 labels of when
	yk = y[:,9:]#13: labels of kind
	if is_class:
		ys,yw,yk = [np.around(y).astype(bool) for y in (ys,yw,yk)]

	if perlabel:
		pred,ytrue = vectorize(y,train,t2,model,is_tfidf = is_tfidf,kbest=kbest,is_sparse=is_sparse,max_features=max_features,is_cv=True,perlabel=perlabel,is_nominal=is_nominal,is_normal=is_normal,min_df=min_df,scale=scale)
	else:
		#(preds,ys_true),(predw,yw_true) = [vectorize(y,train,t2,model,is_tfidf = is_tfidf,kbest=kbest,is_sparse=is_sparse,max_features=max_features,is_cv=True,perlabel=perlabel,min_df=min_df,is_nominal=is_nominal,is_normal=is_normal) for y in (ys,yw)]
		#pred = np.hstack((preds,predw))
		#ytrue = np.hstack((ys_true,yw_true))
		(preds,ys_true),(predw,yw_true),(predk,yk_true) = [vectorize(y,train,t2,model,is_tfidf = is_tfidf,kbest=kbest,is_sparse=is_sparse,max_features=max_features,is_cv=True,perlabel=perlabel,min_df=min_df,is_nominal=is_nominal,is_normal=is_normal,is_LSA=is_LSA,scale=scale) for y in (ys,yw,yk)]
		pred = np.hstack((preds,predw,predk))
		ytrue = np.hstack((ys_true,yw_true,yk_true))

	#pred[pred < 0.01] = 0.0
	#pred[pred > 0.99] = 1.0
	mse = np.sqrt(np.sum(np.array(pred-ytrue)**2)/(ytrue.shape[0]*float(ytrue.shape[1])))
	print 'Train error: {0}'.format(mse)
	return pred,ytrue
	
def submission(predictions,filename='prediction.csv'): 
	col = '%i,' + '%.2f,'*23 + '%.2f'
	header = "id,s1,s2,s3,s4,s5,w1,w2,w3,w4,k1,k2,k3,k4,k5,k6,k7,k8,k9,k10,k11,k12,k13,k14,k15"
	np.savetxt(filename, predictions,col, header=header,delimiter=',') # need to remove first two characters in the output file before submitting!

def crossvalidate(clf,X,y,cv=3):
	scores=[]
	for i in range(int(y.shape[1])):
		clf.fit(X,y[:,i])
		scores.append(cross_val_score(clf, X, y[:,i]))
		print scores[-1],
	return scores

def predictMe(clf,X,y,test,ids):
	test_prediction=[]
	for i in range(int(y.shape[1])):
		clf.fit(X,y[:,i])
		test_prediction.append(clf.predict(test))
	testpred = np.array(test_prediction)
	prediction = np.array(np.hstack([np.matrix(ids).T, testpred.T]))
	return prediction

def predictThis(clf,train,t2,kbest=0,max_features=20000,is_tfidf=True,is_sparse=True,is_nominal=False,is_LSA=False,min_df=5):
	y = np.array(train.ix[:,4:])
	ys = y[:,:5]#4:9 labeles of sentiment
	yw = y[:,5:9]#9:13 labels of when
	yk = y[:,9:]#13: labels of kind

	if is_tfidf:
		#create a tf-idf class with the given features (stop_words='english' is removed since this is done in preprocessing)
		tfidf = TfidfVectorizer(max_features=max_features, strip_accents='unicode', analyzer='word', smooth_idf=True,sublinear_tf=True,max_df=0.5, min_df=min_df, ngram_range=(1,2))
		sent, when, kind = [vectorize(y,train,t2,clf,is_tfidf = is_tfidf,kbest=kbest,is_sparse=is_sparse,max_features=max_features,is_nominal=is_nominal,is_LSA=is_LSA) for y in (ys,yw,yk)]

	testpred = np.hstack((sent, when, kind))
	testpred = np.around(testpred.clip(0,1),3)
	prediction = np.array(np.hstack([np.matrix(t2['id']).T, testpred]))
	return prediction


# to learn about indexing in pandas: http://pandas.pydata.org/pandas-docs/stable/indexing.html#advanced-indexing-with-hierarchical-index
def predictKind(train_file,test_file):
	train_file='train.csv'
	test_file='test.csv'
	#read files into pandas
	train = p.read_csv(train_file)
	t2 = p.read_csv(test_file)
	for row in train.index:
		train['tweet'][row]=' '.join([train['tweet'][row],train['state'][row],str(train['location'][row])])

	#preprocessing for kind prediction: emoticons and stop words can be ignored
	# for row in train.index:
	# 	train['tweet'][row] = pre.preprocess_pipeline(' '.join([train['tweet'][row],train['state'][row],str(train['location'][row])]), return_as_str=True, do_remove_stopwords=True,do_emoticons=True)
	# for row in t2.index:
	# 	t2['tweet'][row] = pre.preprocess_pipeline(' '.join([t2['tweet'][row],str(t2['state'][row]),str(t2['location'][row])]), return_as_str=True, do_remove_stopwords=True,do_emoticons=True)

	clf = linear_model.Ridge (alpha = 3.0, normalize = True)
	#pred,ytrue = cv_loop(train, t2, clf)
	#row_sums = pred.sum(axis=1)
	#pred_norm = pred / row_sums[:, numpy.newaxis]
	#mse = np.sqrt(np.sum(np.array(pred_norm-ytrue)**2)/(pred_norm.shape[0]*24.0))
	#print 'Normalized train error: {0}'.format(mse) #Normalized train error: 0.366281924654

	prediction = predictThis(clf,train,t2)
	submission(prediction,'prediction.csv')
	#metadata =sparse.csr_matrix([ metadata ]).T
	#X = sparse.hstack([X, metadata]).tocsr()
	#metadata = (metadata - mean(metadata))/(max(metadata) - min(metadata))

if __name__ == "__main__":
	# parse commandline arguments
	parser = argparse.ArgumentParser()
	parser.add_argument("train_file", help="a filename name must be specified")
	parser.add_argument("test_file", help="a filename name must be specified")
	args = parser.parse_args()
	prediction = predictKind(args.train_file, args.test_file)
	print 'elh'

"""
# a nice document classification example: http://scikit-learn.org/stable/auto_examples/document_classification_20newsgroups.html
# we like ensemble: http://scikit-learn.org/stable/modules/ensemble.html

You use the vocabulary parameter to specify what features should be used. For example, if you want only emoticons to be extracted, you can do the following:

emoticons = {":)":0, ":P":1, ":(":2}
vect = TfidfVectorizer(vocabulary=emoticons)
matrix = vect.fit_transform(traindata)
This will return a <Nx3 sparse matrix of type '<class 'numpy.int64'>' with M stored elements in Compressed Sparse Row format>]. Notice there are only 3 columns, one for each feature.

If you want the vocabulary to include the emoticons as well as the N most common features, you could calculate the most frequent features first, then merge them with the emoticons and re-vectorize like so:

# calculate the most frequent features first
vect = TfidfVectorizer(vocabulary=emoticons)
matrix = vect.fit_transform(traindata, max_features=10)
top_features = vect.vocabulary_
n = len(top_features)

# insert the emoticons into the vocabulary of common features
emoticons = {":)":0, ":P":1, ":(":2)}
for feature, index in emoticons.items():
    top_features[feature] = n + index

# re-vectorize using both sets of features
# at this point len(top_features) == 13
vect = TfidfVectorizer(vocabulary=top_features)
matrix = vect.fit_transform(traindata)
"""