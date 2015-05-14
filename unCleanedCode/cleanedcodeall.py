
# coding: utf-8

# In[32]:

def norm(data_dict, norm_dict, typeofnorm, high=1.0, low=0.0):
# Normilization of User Playcount
    if typeofnorm == "user":
        for key, value in data_dict.items():
            song = key[0]
            user = key[1]
            playcounts = norm_dict[user]
            maxs = np.max(playcounts)
            rng = maxs - low
            normplay = high - (((high - low) * (maxs - value)) / rng)
            data_dict[song, user] = normplay
    if typeofnorm == "song":
        for key, value in data_dict.items():
            song = key[0]
            user = key[1]
            playcounts = norm_dict[song]
            maxs = np.max(playcounts)
            rng = maxs - low
            normplay = high - (((high - low) * (maxs - value)) / rng)
            data_dict[song, user] = normplay
    if typeofnorm == "none":
        return data_dict
    if typeofnorm == "binary":
        for key, value in data_dict.items():
            song = key[0]
            user = key[1]
            data_dict[song, user] = 1.0
    return data_dict


# In[33]:

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


# In[34]:

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

    for i in omegau_test:
        apk_sum += apk(omegau_test[i], np.argsort(predict_m[i])[::-1][:500])

    map = apk_sum/len(omegau_test)

    return map


# In[35]:

def rawkmeans(Mtrain, omegau, ncl=10):
    kmeans_setup = KMeans(n_clusters=ncl)
    kmeans_model = kmeans_setup.fit_transform(Mtrain)

    ulist = []
    slist = {}
    flatsonglist = {}
    notinOMEGAU = set()

    for i in xrange(0,ncl):
        cl = i
        sorted_array = np.argsort(kmeans_model.T[i])[0:10]
        sort = kmeans_model.T[i, sorted_array]
        for k in xrange(0,10):
            uindex = sorted_array[k]
            if cl in slist:
                if uindex in omegau:
                    slist[cl].append(omegau[uindex])
                else:
                    #print "This index is not in OMEGA_U Uindex=", uindex
                    notinOMEGAU.add(uindex)
            else:
                if uindex in omegau:
                    slist[cl] = [(omegau[uindex])]
                else:
                    #print "This index is not in OMEGA_U Uindex=", uindex 
                    notinOMEGAU.add(uindex)

    for cl in xrange(0,10):
        if cl in slist:
            flatsonglist[cl] = [item for sublist in slist[cl] for item in sublist]

    # assign one user to one cluster
    cl_user_dic = defaultdict(list)
    user_cl_dic = {}    

    #for i in xrange(0, len(kmeans_model)):
    #    cl_assignment = np.argmin(kmeans_model[i]) 
    #    cl_user_dic[cl_assignment].append(i)

    #for key, value in cl_user_dic.items():
    #    print key, len(value)

    for i in xrange(0, len(kmeans_model)):
        cl_assignment = np.argmin(kmeans_model[i]) 
        user_cl_dic[i] = cl_assignment

    # for each song how many users have played it
    cls_songs = {}

    for cl, sindexLIST in flatsonglist.items():
        nplaycnt = []
        for sindex in sindexLIST:
            songh = [key for key, value in song_dict.items() if value == sindex][0]
            tupole = (sindex, len(p2[songh]))
        nplaycnt.append(tupole) 
        sortsongs_t = sorted(nplaycnt, key=lambda x: x[1], reverse=True)
        sortsonsgs = [i[0] for i in sortsongs_t]
        cls_songs[cl] = sortsonsgs

    # list of users and the 500 songs
    for uindex in user_cl_dic:
        # which cluster is the user in
        cl = user_cl_dic[uindex]
        # what songs are in that cluser
        songlist = cls_songs[cl]
        omegau[uindex] = songlist
    apk_sum = 0

    for i in omegau_test:
        apk_sum += apk(omegau_test[i], omegau[i])

    map = apk_sum/len(omegau_test)
    return map


# In[36]:

def fromthetop(data, typeofnorm, alg, d=20, iter=15, ncl=10):
    users = set()
    songs = set()
    numplays = {}
    numsongplayed = defaultdict(list)
    songdic = defaultdict(list)
    userdic = defaultdict(list)

    for line in data:
        split = line.strip().split('\t')
        if len(split) == 3:
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

    # subset by number of songs a user has rated
    p1 = { key: value for key, value in userdic.items() if len(value) > 27 }
    # subset songs by number of plays
    p2 = { key: value for key, value in numsongplayed.items() if len(value) > 22 } 

    # list of songs
    subsongs = p2.keys()

    # list of users who have rated those songs
    subusers = set()
    for s in subsongs:
        for u in songdic[s]:
            if u in p1:
                subusers.add(u)

    p3 = { key: value for key, value in numplays.items() if key[0] in p2 and key[1] in p1 }

    #print len(subusers), len(users)
    #print len(subsongs), len(songs)
    #print (float(len(p3))/len(data) * 100)

    if typeofnorm == "user":
        p3 = norm(p3, p1, "user")
        
    if typeofnorm == "song":
        p3 = norm(p3, p2, "song")
        
    if typeofnorm == "binary":
        p3 = norm(p3, p2, "binary")
        
    if typeofnorm == "none":
        p3 = p3

    allkeys = p3.keys()
    np.random.shuffle(allkeys)
    trainingN = int(len(allkeys) * 0.8)
    trainingKEYS, testKEYS = allkeys[:trainingN], allkeys[trainingN:]

    n = len(subusers)
    m = len(subsongs)

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
    omegau_test = {}

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
            
        if typeofnorm == "binary":
            var = 0.08
        else: 
            var = np.var(varcalc)

    if alg == "MF":
        print "Running MF Algorithm with d ="+str(d)+" with "+str(iter)+" iteration."
        mapval = MF(Mtrain, Mtest, var, omega, omega_test, omegau, omegau_test) 
            
    if alg == "Kmeans":
        mapval = rawkmeans(Mtrain, omegau, ncl)
        
    return mapval


# In[6]:

get_ipython().magic(u'matplotlib inline')
import numpy as np
from collections import defaultdict
from scipy import sparse
import matplotlib.pyplot as plt
from operator import itemgetter
from sklearn.cluster import KMeans

f = open('kaggle_visible_evaluation_triplets.txt', 'r')
data = f.read().split('\n')


# In[37]:

map_user_MF = fromthetop(data, "user", "MF")


# In[38]:

map_song_MF = fromthetop(data, "song", "MF")


# In[39]:

map_binary_MF = fromthetop(data, "binary", "MF")


# In[40]:

print map_user_MF
print map_song_MF
print map_binary_MF


# In[ ]:

# NMF using J's code
# TFIDF



# In[ ]:



