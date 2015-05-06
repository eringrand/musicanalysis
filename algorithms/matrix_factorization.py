import numpy as np
#import sqlite3 as sqlite
import pandas as pd
#import pandas.io.sql as psql
import matplotlib.pyplot as plt
import random

import matplotlib
matplotlib.style.use('ggplot')

def apk(actual, predicted, k=500):
    """
    Computes the average precision at k.

    This function computes the average prescision at k between two lists of
    items.

    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements

    Returns
    -------
    score : double
            The average precision at k over the input lists

    """
    if len(predicted)>k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i,p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    if not actual:
        return 1.0

    return score / min(len(actual), k)

def MF(Mtrain, Mtest, var, omega, omega_test, omegau, omegau_test, d=20, iter=15):
    
    N1 = np.shape(Mtrain)[0]
    N2 = np.shape(Mtrain)[1]

    users1 = {}
    songs1 = {}
    for i in range(N1):
        users1[i] = np.where(Mtrain[i] != 0)[0]

    for j in range(N2):
        songs1[j] = np.where(Mtrain[:,j] != 0)[0]

    I = np.identity(d)
    lamb = 10
    t1 = lamb*var*I

    mean = np.zeros([d])
    v = np.empty([d,N2])
    u = np.empty([N1,d])

    for j in xrange(0, N2):
        v[:,j] = np.random.multivariate_normal(mean,(1/float(lamb))*I)

    RMSE = []
    loglik=[]

    def predict(u, v):
        product = np.dot(u, v)
        if product < 0.:
            pred = 0.0
        elif product > 1.:
            pred = 1.0
        else:
            pred = product
        return pred

    # Performing matrix factorization
    for c in xrange(0,iter):
        for i in range(N1):
            inner = np.dot(v[:,users1[i]], v[:,users1[i]].T)
            outer = np.dot(Mtrain[i][users1[i]], v[:,users1[i]].T)
            u[i] = np.dot((np.linalg.inv(t1 + inner)), outer.T).T

        for j in range(N2):
            inner = np.dot(u[songs1[j]].T, u[songs1[j]])
            outer = np.dot(Mtrain[songs1[j],j], u[songs1[j]])
            v[:,j] = np.dot(np.linalg.inv(t1 + inner), outer.T)

        sum3 = 0        
        for (i,j) in omega_test:
            prediction = predict(u[i], v[:,j])
            actual = Mtest[i][j]
            sum3 = sum3 + (prediction - actual)**2
        temp = (sum3/float(len(omega_test)))**0.5
        RMSE.append(temp)

        sum4 = 0
        for (i, j) in omega:
            sum4 = sum4 + 0.5/var*np.power(Mtrain[i][j] - np.dot(u[i], v[:,j]),2)
        sum4 = -sum4
        sum5 = 0
        for i in xrange(0, N1):
            sum5 = sum5 + 0.5*lamb*np.sum(u[i]**2)
        sum5 = -sum5
        sum6 = 0
        for j in xrange(0, N2):
            sum6 = sum6 + 0.5*lamb*np.sum(v[:,j]**2)
        sum6 = -sum6
        loglik.append(sum4+sum5+sum6)

    fig = plt.figure()         
    # RMSE 
    plt.subplot(1,2,1)      
    plt.plot(RMSE)
    plt.xlabel('Iteration')
    plt.ylabel('RMSE')

    # Log Joint Likelihood
    plt.subplot(1,2,2) 
    plt.plot(loglik)
    plt.xlabel('Iteration')
    plt.ylabel('Log Joint Likelihood')

    predict_m = np.dot(u,v)
    
    apk_sum = 0
    counter = 0

    recommend = {}
    rec = {}
    for i in list(set(omegau_test).intersection(omegau)):
        recommend[i] = np.argsort(predict_m[i])[::-1]
        rec[i] = [x for x in recommend[i] if x not in omegau[i]][0:500]
        apk_sum += apk(omegau_test[i], rec[i])
        counter += 1

    mapval = apk_sum/counter

    return mapval

# load observation data
f = open("kaggle_visible_evaluation_triplets.txt", 'rb')
eval = pd.read_csv(f,sep='\t',header = None, names = ['user_id','sid','plays'])

# count number of songs per user and subset
userhist = eval.groupby('user_id').count()
userhist = pd.DataFrame(userhist).reset_index()
usersub = userhist[userhist['plays']>27]

# count number of users per song and subset
songhist = eval.groupby('sid').count()
songhist = pd.DataFrame(songhist).reset_index()
songsub = songhist[songhist['plays']>22]

# subset the whole dataset
sub = eval[eval['sid'].isin(songsub['sid'])]
sub = sub[sub['user_id'].isin(usersub['user_id'])]

# sampling/splitting the dataset
sample = random.sample(sub.index, int(sub.shape[0]*0.2))
trainsub = sub.copy()
trainsub.ix[trainsub.index.isin(sample),'plays'] = 0
testsub = sub.copy()
testsub.ix[~trainsub.index.isin(sample),'plays'] = 0

# generating a matrix out of dataframe
trainpivot = trainsub.pivot(index='user_id',columns='sid', values='plays')
# creating the mapping of user index to user id
user_index = pd.DataFrame(trainpivot.index).reset_index()
user_index.columns = [['user_index','user_id']]
trainsub = pd.merge(trainsub, user_index, on='user_id')
testsub = pd.merge(testsub, user_index, on='user_id')
# creating the mapping of song index to song id
song_index = pd.DataFrame(trainpivot.columns).reset_index()
song_index.columns = [['song_index','sid']]
trainsub = pd.merge(trainsub, song_index, on='sid')
testsub = pd.merge(testsub, song_index, on='sid')

# Generating the default M_train and M_test matrices
M_train = trainpivot.as_matrix()
M_train = np.nan_to_num(M_train)

testpivot = testsub.pivot(index='user_id',columns='sid', values='plays')
M_test = testpivot.as_matrix()
M_test = np.nan_to_num(M_test)

# creating omegau_test
testplays = testsub[testsub.plays>0]
test_usergroup = testsub[testsub.plays>0].groupby('user_index')
omegau_test = {}
for i in list(set(testplays.user_index.values)):
    omegau_test[i] = list(test_usergroup.get_group(i)['song_index'])
# creating omegau_train
trainplays = trainsub[trainsub.plays>0]
train_usergroup = trainplays.groupby('user_index')
omegau_train = {}
for i in list(set(trainplays.user_index.values)):
    omegau_train[i] = list(train_usergroup.get_group(i)['song_index'])

# creating tuple lists
omega = [tuple(x) for x in trainsub[['user_index','song_index']].values]
omega_test = [tuple(x) for x in testsub[['user_index','song_index']].values]

# Processing M_train, row normalization, column normalization, and tfidf
rowsum = np.sum(M_train,axis=1).reshape((-1,1))
rowsum[rowsum == 0] = 1E-16
M_train_rownorm = M_train/rowsum

colsum = np.sum(M_train,axis=0).reshape((1,-1))
colsum[colsum == 0] = 1E-16
M_train_colnorm = M_train/colsum

idftrain = trainsub.copy()
idftrain.plays = 1
idf_train = idftrain.pivot(index='user_id',columns='sid', values='plays').as_matrix()
idf_train = np.nan_to_num(idf_train)
colsum = np.sum(idf_train,axis=0)
idf = np.log10(idf_train.shape[0]/colsum)
M_train_tfidf = M_train_rownorm*idf

M_train_binary = M_train.copy()
M_train_binary[M_train_binary != 0] = 1

#for var in [0.01, 0.001, 0.0001]:
#    for d in [20, 50, 100]:
#        for iter in [20, 40, 60]:
#            print var, d, iter
#            print MF(M_train,M_test,var, omega, omega_test, omegau_train, omegau_test,d,iter)
#            print MF(M_train_rownorm, M_test, var, omega_test, omegau_train, omegau_test,d,iter)
#            print MF(M_train_colnorm, M_test, var, omega_test, omegau_train, omegau_test,d,iter)
#            print MF(M_train_binary, M_test, var, omega_test, omegau_train, omegau_test,d,iter)
#            print MF(M_train_tfidf, M_test, var, omega_test, omegau_train, omegau_test,d,iter)       
