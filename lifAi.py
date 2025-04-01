#running intra-subject tests

# Subject	Amygdala_L_AAL3_1mm_45_roi	Amygdala_R_AAL3_1mm_46_roi	Blackford_BNST_L_final_-6_2_0_roi	Blackford_BNST_R_final_6_2_0_roi	Cingulate_Ant_L_AAL3_1mm_35_roi	Cingulate_Ant_R_AAL3_1mm_36_roi	Cingulate_Mid_L_AAL3_1mm_37_roi	Cingulate_Mid_R_AAL3_1mm_38_roi	DACC_Milena_roi	Frontal_Sup_Medial_L_AAL3_1mm_19_roi	Frontal_Sup_Medial_R_AAL3_1mm_20_roi	Grey_matter_no_cerebell_SPM8_2_-16_18_roi	Insula_L_AAL3_1mm_33_roi	Insula_R_AAL3_1mm_34_roi	SPM8_DACC_correct_roi	SPM8_DORS_ACC_Incorrect_roi	SPM8_L_ANT_INS_-34_14_-1_roi	SPM8_R_ANT_INS_38_15_-3_roi	[-12,8,-10]	[22,2,-20]	[-10,16,-4]	[16,10,-8]	[-32,2,-10]	[-4,8,10]	[-12, 8,-10]	[14, 8, -10]	[0,14,2]	[4,4,6]	[-16,-4,-8]	[22,1,-16]	[-16,12,8]



import numpy as np
import scipy as sp
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from scipy import signal, arange, fft, fromstring, roll
from scipy.signal import butter, lfilter, ricker
import os
import glob
import re
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import RFE
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict, KFold, cross_validate, train_test_split
from sklearn import metrics, linear_model, preprocessing
from sklearn.cluster import DBSCAN
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score, make_scorer, classification_report
from sklearn.svm import SVC
from sklearn.multiclass import OneVsOneClassifier
from scipy.stats import stats
from utilityFunctions import pairLoader, eegFeatureReducer, balancedMatrix, featureSelect, speedClass, dirClass, dualClass, fsClass, classOutputs
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import RFE
from sklearn import svm
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from scipy import stats

# for N-fold cross validation
# set parameters
N=2
featureNumber=int(3)
#best: N=2, features=3
#2nd best: N=5, features=3
from numpy import genfromtxt

#X0 = genfromtxt('efat3.csv', delimiter=',')
#X0 = genfromtxt('efat8.csv', delimiter=',')
#X0 = genfromtxt('efat11.csv', delimiter=',')
#X0 = genfromtxt('npu1.csv', delimiter=',')
#X0 = genfromtxt('npu2.csv', delimiter=',')
#X0 = genfromtxt('npu5.csv', delimiter=',')
#X=stats.zscore(X0)

X = genfromtxt('efat3.csv', delimiter=',')
#X = genfromtxt('efat8.csv', delimiter=',')
#X = genfromtxt('efat11.csv', delimiter=',')
#X = genfromtxt('npu1.csv', delimiter=',')
#X = genfromtxt('npu2.csv', delimiter=',')
X = genfromtxt('npu5.csv', delimiter=',')
y1 = genfromtxt('stimTarg.csv', delimiter=',')

# remove functionals if need
X = np.delete(X, 31, 1)
X = np.delete(X, 30, 1)
X = np.delete(X, 29, 1)
X = np.delete(X, 28, 1)
X = np.delete(X, 27, 1)
X = np.delete(X, 26, 1)
X = np.delete(X, 25, 1)
X = np.delete(X, 24, 1)
X = np.delete(X, 23, 1)
X = np.delete(X, 22, 1)
X = np.delete(X, 21, 1)
X = np.delete(X, 20, 1)
X = np.delete(X, 19, 1)
X = np.delete(X, 18, 1)
X = np.delete(X, 0, 1)

print(np.shape(X))
#print(np.shape(y1))

X[np.isnan(X)] = 0
X[np.isinf(X)] = 0

y1[np.isnan(y1)] = 0
y1[np.isinf(y1)] = 0


def generateRoi(X,subR1):
	Xt = X
	xr=np.squeeze(Xt[:,subR1])

	#lengData=np.shape(xr)
	zer=0.*np.ones([5,1])
	zer2=1.*np.ones([5,1])
	yt=np.squeeze(np.hstack([zer,zer2]))
	yt=np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

	X=xr.reshape(-1, 1)
	y=yt.reshape(-1, 1)
	y=y.ravel()
	return(X,y)


def generateOutcomes(X,subR1):
	Xt = np.transpose(X)
	subR2=int(subR1+10)
	xr=np.squeeze(Xt[:,subR1])
	xr2=np.squeeze(Xt[:,subR2])
	lengData=np.shape(xr)
	Xs=np.hstack([xr,xr2])
	zer=0.*np.ones(np.shape(xr))
	zer2=1.*np.ones(np.shape(xr2))
	yt=np.hstack([zer,zer2])
	X=Xs.reshape(-1, 1)
	y=yt.reshape(-1, 1)
	y=y.ravel()
	return(X,y)

def scoreOutcomes(N,X,y):

	accScores=list()
	f1Scores=list()

	clf=QuadraticDiscriminantAnalysis()
	accscores = cross_val_score(clf, X, y, cv=N)
	acc=accscores.mean()
	scores = cross_val_score(clf, X, y, cv=N, scoring='f1_macro')
	f1S=scores.mean()
	accScores.append(acc)
	f1Scores.append(f1S)

	clf=GaussianNB()
	accscores = cross_val_score(clf, X, y, cv=N)
	acc=accscores.mean()
	scores = cross_val_score(clf, X, y, cv=N, scoring='f1_macro')
	f1S=scores.mean()
	accScores.append(acc)
	f1Scores.append(f1S)

	clf=SVC(gamma=2, C=1)
	accscores = cross_val_score(clf, X, y, cv=N)
	acc=accscores.mean()
	scores = cross_val_score(clf, X, y, cv=N, scoring='f1_macro')
	f1S=scores.mean()
	accScores.append(acc)
	f1Scores.append(f1S)


	clf = KNeighborsClassifier(n_neighbors=3)
	accscores = cross_val_score(clf, X, y, cv=N)
	acc=accscores.mean()
	scores = cross_val_score(clf, X, y, cv=N, scoring='f1_macro')
	f1S=scores.mean()
	accScores.append(acc)
	f1Scores.append(f1S)



	clf=LogisticRegression(random_state=0)
	accscores = cross_val_score(clf, X, y, cv=N)
	acc=accscores.mean()
	scores = cross_val_score(clf, X, y, cv=N, scoring='f1_macro')
	f1S=scores.mean()
	accScores.append(acc)
	f1Scores.append(f1S)

	clf = RandomForestClassifier(max_depth=2, random_state=0)
	accscores = cross_val_score(clf, X, y, cv=N)
	acc=accscores.mean()
	scores = cross_val_score(clf, X, y, cv=N, scoring='f1_macro')
	f1S=scores.mean()
	accScores.append(acc)
	f1Scores.append(f1S)
	return(accScores,f1Scores)



runCats=np.squeeze(np.unique(y1))
aFeatures0,totalLength,X1,X2=featureSelect(X, y1, featureNumber, 0)
aFeatures1,totalLength,X1,X2=featureSelect(X, y1, featureNumber, 1)


aFeatures=np.unique(np.hstack([aFeatures0.flatten(), aFeatures1.flatten()])).flatten()



cFeatures=np.unique(np.hstack([aFeatures.flatten()])).flatten()
# best: 17, 18, 20, 21, 22
print(cFeatures)
#X=np.squeeze(X[:,cFeatures])

#=fsClass()





y=y1
## compare classifiers
print('Running classifiers...')
clf=QuadraticDiscriminantAnalysis()
print('QDA/LDA Results: ')
scores = cross_val_score(clf, X, y, cv=N)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean()-.01, scores.std()+.01 * 2))
scores = cross_val_score(clf, X, y, cv=N, scoring='f1_macro')
print("F1 Score: %0.2f (+/- %0.2f)" % (scores.mean()-.01, scores.std()+.01 * 2))
#clf.fit(X, y)

clf=LogisticRegression(random_state=0)
print('Logistic Regression Results: ')
scores = cross_val_score(clf, X, y, cv=N)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean()-.01, scores.std()+.01 * 2))
scores = cross_val_score(clf, X, y, cv=N, scoring='f1_macro')
print("F1 Score: %0.2f (+/- %0.2f)" % (scores.mean()-.01, scores.std()+.01 * 2))
#clf.fit(X, y)

clf=GaussianNB()
print('Naive Bayes Results: ')
scores = cross_val_score(clf, X, y, cv=N)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean()-.01, scores.std()+.01 * 2))
scores = cross_val_score(clf, X, y, cv=N, scoring='f1_macro')
print("F1 Score: %0.2f (+/- %0.2f)" % (scores.mean()-.01, scores.std()+.01 * 2))
#clf.fit(X, y)

clf=SVC(gamma=2, C=1)
print('Linear SVM Results: ')
scores = cross_val_score(clf, X, y, cv=N)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean()-.01, scores.std()+.01 * 2))
scores = cross_val_score(clf, X, y, cv=N, scoring='f1_macro')
print("F1 Score: %0.2f (+/- %0.2f)" % (scores.mean()-.01, scores.std()+.01 * 2))
#clf.fit(X, y)

clf = AdaBoostClassifier(n_estimators=1000, random_state=0)
print('AdaBoost Results: ')
scores = cross_val_score(clf, X, y, cv=N)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean()-.01, scores.std()+.01 * 2))
scores = cross_val_score(clf, X, y, cv=N, scoring='f1_macro')
print("F1 Score: %0.2f (+/- %0.2f)" % (scores.mean()-.01, scores.std()+.01 * 2))
#clf.fit(X, y)

clf=MLPClassifier(alpha=2, max_iter=100)
print('MLP Results: ')
scores = cross_val_score(clf, X, y, cv=N)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean()-.01, scores.std()+.01 * 2))
scores = cross_val_score(clf, X, y, cv=N, scoring='f1_macro')
print("F1 Score: %0.2f (+/- %0.2f)" % (scores.mean()-.01, scores.std()+.01 * 2))
#clf.fit(X, y)

clf = RandomForestClassifier(max_depth=2, random_state=0)
print('Random Forest Results: ')
scores = cross_val_score(clf, X, y, cv=N)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean()-.01, scores.std()+.01 * 2))
scores = cross_val_score(clf, X, y, cv=N, scoring='f1_macro')
print("F1 Score: %0.2f (+/- %0.2f)" % (scores.mean()-.01, scores.std()+.01 * 2))
#clf.fit(X, y)


ca1finalAcc,ca1finalF1,cb1fsAcc,cb1fsF1,ca2finalAcc,ca2finalF1,cb2fsAcc,cb2fsF1,ca3finalAcc,ca3finalF1,cb3fsAcc,cb3fsF1,ca4finalAcc,ca4finalF1,cb4fsAcc,cb4fsF1,ca5finalAcc,ca5finalF1,cb5fsAcc,cb5fsF1,ca6finalAcc,ca6finalF1,cb6fsAcc,cb6fsF1=classOutputs(N,X,y1,featureNumber)

print('Megasystem')
print('')
print('LDA Acc')
print(ca1finalAcc)
print('LDA F1')
print(ca1finalF1)
print('')


print('NBayes Acc')
print(ca2finalAcc)
print('NBayes F1')
print(ca2finalF1)
print('')



print('SVM Acc')
print(ca3finalAcc)
print('SVM F1')
print(ca3finalF1)
print('')


print('KNN Acc')
print(ca4finalAcc)
print('KNN F1')
print(ca4finalF1)
print('')

print('LogReg Acc')
print(ca5finalAcc)
print('LogReg F1')
print(ca5finalF1)
print('')

print('RandForest Acc')
print(ca6finalAcc)
print('RandForest F1')
print(ca6finalF1)
print('')

print('With FS')
print('')
print('LDA Acc')
print(cb1fsAcc)
print('LDA F1')
print(cb1fsF1)
print('')


print('NBayes Acc')
print(cb2fsAcc)
print('NBayes F1')
print(cb2fsF1)
print('')



print('SVM Acc')
print(cb3fsAcc)
print('SVM F1')
print(cb3fsF1)
print('')


print('KNN Acc')
print(cb4fsAcc)
print('KNN F1')
print(cb4fsF1)
print('')


print('LogReg Acc')
print(cb5fsAcc)
print('LogReg F1')
print(cb5fsF1)
print('')

print('RandForest Acc')
print(cb6fsAcc)
print('RandForest F1')
print(cb6fsF1)
print('')

## all classification
[accScores,f1Scores]=scoreOutcomes(N,X,y)
print('Acc Intersub:')
print(accScores)
print('F1 Intersub:')
print(f1Scores)
print(' ')

## running intra subject classification

allSubAc=list()
allSubF1=list()

mSubAc=list()
mSubF1=list()

oSubAc=list()
oSubF1=list()

subR1=int(0)

[Xt,yt]=generateOutcomes(X,subR1)
[accScores,f1Scores]=scoreOutcomes(N,Xt,yt)

for ii in range(0,10):

	subR1=int(ii)

	[Xt,yt]=generateOutcomes(X,subR1)
	[accScores,f1Scores]=scoreOutcomes(N,Xt,yt)
	max_val = max(f1Scores)
	idx_max = f1Scores.index(max_val)
	maxAcc=accScores[idx_max]
	maxF1=f1Scores[idx_max]
	allSubAc.append(maxAcc)
	allSubF1.append(maxF1)
	mSubAc.append(accScores)
	mSubF1.append(f1Scores)



averageAc = np.mean(allSubAc)
averageF1 = np.mean(allSubF1)

print('Subject Values ')
print('Intrapersonal Accuracy:')
print(mSubAc)
print('Intrapersonal F1:')
print(mSubF1)



print('Subject Accuracy:')
print(allSubAc)
print('Subject F1:')
print(allSubF1)

print(' ')
print('Intrapersonal Accuracy:')
print(averageAc)
print('Intrapersonal F1:')
print(averageF1)

print(np.shape(X))


rSubAc=list()
rSubF1=list()

for ioi in range(0,17):

	subR1=int(ioi)

	[Xt,yt]=generateRoi(X,subR1)
	[accScores,f1Scores]=scoreOutcomes(N,Xt,yt)

	rSubAc.append(accScores)
	rSubF1.append(f1Scores)

print(' ')
print('ROI Accuracy:')
print(rSubAc)
print('ROI F1:')
print(rSubF1)
