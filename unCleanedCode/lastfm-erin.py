
# coding: utf-8

# In[1]:

import numpy as np
from collections import defaultdict
from scipy import sparse
import matplotlib.pyplot as plt
from operator import itemgetter


f = open('kaggle_visible_evaluation_triplets.txt', 'r')
data = f.read().split('\n')


# In[2]:

users = set()
songs = set()
numplays = {}
numsongplayed = defaultdict(list)
songdic = defaultdict(list)
userdic = defaultdict(list)

for line in data:
    split = line.strip().split('\t')
    if len(split) != 3:
        print len(split), split
    else:
        userhash = split[0]
        songhash = split[1]
        plays = float(split[2])
        users.add(userhash)
        songs.add(songhash)
        numplays[songhash, userhash] = plays
        if songhash not in userdic[userhash]:
            userdic[userhash].append(plays)
        if userhash not in songdic[songhash]:
            songdic[songhash].append(userhash)
        #if songhash not in numsongplayed:
        numsongplayed[songhash].append(plays)


# In[3]:

# subset by number of songs a user has rated
p1 = { key: value for key, value in userdic.items() if sum(value) > 29 }
# subset by number of plays
p2 = { key: value for key, value in numsongplayed.items() if sum(value) > 34 }

# list of songs
subsongs = p2.keys()

# list of users who have rated those songs
subusers = set()
for s in subsongs:
    for u in songdic[s]:
        if u in p1:
            subusers.add(u)
        
p3 = { key: value for key, value in numplays.items() if key[0] in p2 and key[1] in p1 }

print len(subusers), len(users)
print len(subsongs), len(songs)
print 100. - (float(len(p3))/len(data) * 100)
#print (float(len(subsongs)) * len(subusers)) / ( len(songs) * len(users)) * 100


# In[4]:

for key, value in p3.items():
    song = key[0]
    user = key[1]
    playcounts = p2[song]
    maxs = np.max(playcounts)
    rng = maxs - 0.0
    high = 1.0
    low = 0.0 
    normplay = high - (((high - low) * (maxs - value)) / rng)
    p3[song, user] = normplay


# In[5]:

allkeys = p3.keys()

np.random.shuffle(allkeys)

trainingN = int(len(allkeys) * 0.8)
trainingKEYS, testKEYS = allkeys[:trainingN], allkeys[trainingN:]

print len(trainingKEYS), len(testKEYS)


# In[16]:

n = len(subusers)
m = len(subsongs)

users = subusers
songs = subsongs

user_dict = {}
i = 0
for user in subusers:
    user_dict[user] = i
    i += 1
    
song_dict = {}
i = 0
for song in subsongs:
    song_dict[song] = i
    i += 1
    
Mtrain = np.zeros((n,m))
Mtest = np.zeros((n,m))

omega = []
omega_test = []
varcalc = []
omegau = {}
omegav = {}

for key in trainingKEYS:
    plays = p3[key]
    song = key[0]
    user = key[1]
    userindex = user_dict[user]
    songindex = song_dict[song]
    omega.append((userindex,songindex))
    varcalc.append(float(plays))
    Mtrain[userindex, songindex] = float(plays)
    i = userindex
    j = songindex
    if i in omegau:
        omegau[i].append(j)
    else:
        omegau[i] = [j]
    if j in omegav:
        omegav[j].append(i)
    else:
        omegav[j] = [i]

for key in testKEYS:
    plays = p3[key]
    song = key[0]
    user = key[1]
    userindex = user_dict[user]
    songindex = song_dict[song]
    omega_test.append((userindex,songindex))
    Mtest[userindex, songindex] = float(plays)


# In[12]:

var = np.var(varcalc)

#print np.sum(Mtrain, axis=0) 
#print np.sum(Mtest, axis=0) 
#print n, m
print var
#print np.shape(Mtrain)
#print np.shape(Mtest)


# In[17]:

N1 = np.shape(Mtrain)[0]
N2 = np.shape(Mtrain)[1]

d = 20
I = np.identity(d)
lamb = 10
t1 = lamb*var*I

mean = np.zeros([d])
v = np.empty([N2,d])
u = np.empty([N1,d])

for j in xrange(0, N2):
    v[j] = np.random.multivariate_normal(mean,(1/float(lamb))*I)

def predict(u, v):
    product = np.dot(u, v)
    if product < 0:
        pred = 0.0
    elif product > 1:
        pred = 1.0
    else:
        pred = product
    return pred

RMSE = []
loglik = [] 


# In[21]:

for iter in xrange(0,10):
    #print iter
    for i in xrange(0, N1):
        sum1 = np.zeros([d,d])
        sum2 = np.zeros(d)
        if i in omegau:
            for j in omegau[i]:
                sum1 = sum1 + np.outer(v[j],np.transpose(v[j]))
                sum2 = sum2 + np.dot(Mtrain[i][j],v[j])
            u[i] = np.dot(np.linalg.inv(t1 + sum1), sum2)
               
    for j in xrange(0, N2):
        sum1 = np.zeros([d,d])
        sum2 = np.zeros(d)
        if j in omegav:
            for i in omegav[j]:
                sum1 = sum1 + np.outer(u[i],np.transpose(u[i]))
                sum2 = sum2 + np.dot(Mtrain[i][j],u[i])
            v[j] = np.dot(np.linalg.inv(t1 + sum1), sum2)
    
    sum3 = 0        
    for (i,j) in omega_test:
        prediction = predict(u[i], v[j])
        actual = Mtest[i][j]
        sum3 = sum3 + (prediction - actual)**2
    temp = (sum3/float(len(omega_test)))**0.5
    RMSE.append(temp)
    
    sum4 = 0
    for (i, j) in omega:
        sum4 = sum4 + 0.5/var*np.power(Mtrain[i][j] - np.dot(u[i], v[j]),2)
    sum4 = -sum4
    sum5 = 0
    for i in xrange(0, N1):
        sum5 = sum5 + 0.5*lamb*np.sum(u[i]**2)
    sum5 = -sum5
    sum6 = 0
    for j in xrange(0, N2):
        sum6 = sum6 + 0.5*lamb*np.sum(v[j]**2)
    sum6 = -sum6
    loglik.append(sum4+sum5+sum6)


# In[24]:

#get_ipython().magic(u'matplotlib inline')

# Part 1: RMSE and likelihood
fig = plt.figure()         
# RMSE 
plt.subplot(1,2,1)      
plt.plot(RMSE)
plt.xlabel('Iteration')
plt.ylabel('RMSE')


# In[25]:

# Log Joint Likelihood
plt.subplot(1,2,2) 
plt.plot(loglik)
plt.xlabel('Iteration')
plt.ylabel('Log Joint Likelihood')