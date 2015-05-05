from operator import itemgetter
import numpy as np
from collections import defaultdict



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
p1 = { key: value for key, value in userdic.items() if sum(value) > 98 }
# subset by number of plays
p2 = { key: value for key, value in numsongplayed.items() if sum(value) > 52 }

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
        
omegau_test = {}
omegav_test = {}

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
# list of songs listened to by a given user
#user_to_song = {}

# List of users who listened to given song
#song_to_user = {}


def item_based_rec(user_to_song, song_to_user, N2):
    
    # Item-based similarity
    
    sim_songs = np.zeros([N2,N2])
    top_songs = {}
    
    for song1 in song_to_user:
        top_songs[song1] = []
        for song2 in song_to_user:
            user1 = song_to_user[song1]
            user2 = song_to_user[song2]
            sim_songs[song1][song2] = (len(set(user1).intersection(user2))) / (len(user1)**0.5 * len(user2)**0.5)
            top_songs[song1].append((song2, sim_songs[song1][song2]))
        top_songs[song1] = sorted(top_songs[song1], key = itemgetter(1), reverse = True)[0:500]
    
    recommend = {}       
    for user in user_to_song:
        songs = user_to_song[user]
        recommend[user] = []
        for song in songs:
            recommend[user] = recommend[user] + top_songs[song]
        recommend[user] = sorted(recommend[user], key = itemgetter(1), reverse = True)[0:500]
        recommend[user] = [x[0] for x in recommend[user]]
    
    return recommend


# In[8]:


item_based = item_based_rec(omegau, omegav, np.shape(Mtrain)[1])