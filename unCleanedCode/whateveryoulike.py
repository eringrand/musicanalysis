
# coding: utf-8

# In[1]:

import numpy as np
from collections import defaultdict
from scipy import sparse
import matplotlib.pyplot as plt
from operator import itemgetter
from sklearn.cluster import KMeans

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
    playcounts = p1[user]
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


# In[6]:

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
    i = userindex
    j = songindex
    if i in omegau_test:
        omegau_test[i].append(j)
    else:
        omegau_test[i] = [j]


# In[7]:

var = np.var(varcalc)
print var


# In[8]:

N1 = np.shape(Mtrain)[0]
N2 = np.shape(Mtrain)[1]

users1 = {}
songs1 = {}
for i in range(N1):
    users1[i] = np.where(Mtrain[i] != 0)[0]

for j in range(N2):
    songs1[j] = np.where(Mtrain[:,j] != 0)[0]


# In[9]:

d = 20
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


# In[15]:

def predict(u, v):
    product = np.dot(u, v)
    if product < 0:
        pred = 0.0
    #elif product > 1:
    #    pred = 1.0
    else:
        pred = product
    return pred

iter = 15


# In[11]:

# Performing matrix factorization
for c in range(iter):
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

predict_m = np.dot(u,v)
apk_sum = 0

for i in range(len(u)):
    apk_sum += apk(omegau_test[i],np.argsort(predict_m[i])[::1][:500])

map = apk_sum/len(u)

# In[13]:

get_ipython().magic(u'matplotlib inline')

# Part 1: RMSE and likelihood
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


# In[18]:

#count = np.nonzero(M)
#count = len(count[0])
#print count
#print len(users)*len(songs)
#print float(count)/(len(users)*len(songs))

#from sklearn.decomposition import ProjectedGradientNMF
#model = ProjectedGradientNMF(n_components=10, sparseness='data', tol=0.0005)
#M = M.asformat('csc')
#model.fit(M) 
#print model.components_
#print model.reconstruction_err_


# In[142]:

def kmeans(data, obs, clusters):    
    lik = []
    c_old = np.zeros([obs]) - 1
    mu = np.empty([clusters, np.shape(data)[1]])
    temp = np.random.random_integers(0, np.shape(data)[0]-1, clusters)
    for k in xrange(0, clusters):
        mu[k] = data[temp[k]]
    #mu = np.random.uniform(-10, 10, (clusters, 2))
    c = np.empty([obs])
    
    n = np.zeros([clusters])

    
    while not np.array_equiv(c_old,c):
        c_old = np.copy(c)
        
        for i in xrange(0,obs):
            temp = np.sum(np.power(data[i] - mu[0],2))
            c[i] = 0
            for k in xrange(1,clusters):
                 if np.sum(np.power(data[i] - mu[k],2)) < temp:
                     temp = np.sum(np.power(data[i] - mu[k],2))
                     c[i] = k
                     
        for k in xrange(0, clusters):
            n[k] = (c==k).sum()
            temp = np.zeros([np.shape(data)[1]])
            for i in xrange(0, obs):
                if c[i] == k:
                    temp = temp + data[i]
            mu[k] = temp/n[k]
        
        temp = 0        
        for i in xrange(0, obs):
            for k in xrange(0, clusters):
                if c[i] == k:
                    temp = temp + np.sum(np.power(data[i] - mu[k],2)) 
        lik.append(temp)
        
    #if len(lik) < 20:
    #lik = lik + [lik[len(lik)-1]] * (20 - len(lik))        
        
    return c, mu, lik


# In[23]:

#k = 10
#cluster, centroid, lik = kmeans(u, N1, k)
#plt.plot(lik)
#plt.xlabel('Iteration')
#plt.ylabel('Likelihood')
#plt.title('K='+ str(k))


# In[44]:

# Cluster by songs, check for top songs
for i in xrange(0,1):
    kmeans_model = KMeans(n_clusters=10).fit_transform(v.T).T
    
    slist = []
    for i in xrange(0,10):
        sorted_array = np.argsort(kmeans_model[i])[::-1][0:10]
        for k in xrange(0,10):
            s = [key for key, value in song_dict.items() if value == sorted_array[k]][0]
            print i, s
            slist.append(s) 
    #print len(slist)
    print len(set(slist))


# In[43]:

# clustering on the users
for i in xrange(0,1):
    centroid = KMeans(n_clusters=10).fit(u).cluster_centers_
    #print np.shape(kmeans_model)
    
    slist = []
    for i in xrange(0,10):
        score = {}
        for j in xrange(0, N2):
            score[j] = np.dot(centroid[i], v[:,j])
        sorted_score = sorted(score.items(), key = itemgetter(1), reverse=True)
        print sorted_score[0]
        for k in xrange(0,10):
            s = [key for key, value in song_dict.items() if value == sorted_score[k][0]][0]
            #print i, s
            slist.append(s)
    #print '\n'  
    #print len(slist)
    print len(set(slist))


# In[32]:

np.savetxt("centroids", centroid)
np.savetxt("VintoKmeans", v.T)
np.savetxt("UintoKmeans", u)


# In[42]:




# In[ ]:



[x for x in recommend[user1] if x[0] not in user_to_song[user1]][0:500]
