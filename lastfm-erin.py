
# coding: utf-8



import numpy as np
from collections import defaultdict
from scipy import sparse

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


# In[7]:

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


# In[8]:

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




allkeys = p3.keys()

np.random.shuffle(allkeys)

trainingN = int(len(allkeys) * 0.8)
trainingKEYS, testKEYS = allkeys[:trainingN], allkeys[trainingN:]

print len(trainingKEYS), len(testKEYS)



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

for key in trainingKEYS:
    plays = p3[key]
    song = key[0]
    user = key[1]
    userindex = user_dict[user]
    songindex = song_dict[song]
    Mtrain[userindex, songindex] = float(plays)

for key in testKEYS:
    plays = p3[key]
    song = key[0]
    user = key[1]
    userindex = user_dict[user]
    songindex = song_dict[song]
    Mtest[userindex, songindex] = float(plays)



print np.sum(Mtrain, axis=0) 
print np.sum(Mtest, axis=0) 

print n, m
print np.shape(Mtrain)
print np.shape(Mtest)


