from operator import itemgetter
import numpy as np
from collections import defaultdict
import sqlite3 as sqlite
import pandas as pd
import pandas.io.sql as psql
import random






f = open("kaggle_visible_evaluation_triplets.txt", 'rb')
eval = pd.read_csv(f,sep='\t',header = None, names = ['user_id','sid','plays'])

userhist = eval.groupby('user_id').count()
userhist = pd.DataFrame(userhist).reset_index()
usersub = userhist[userhist['plays']>27]

songhist = eval.groupby('sid').count()
songhist = pd.DataFrame(songhist).reset_index()
songsub = songhist[songhist['plays']>22]

sub = eval[eval['sid'].isin(songsub['sid'])]
sub = sub[sub['user_id'].isin(usersub['user_id'])]

len(set(sub['user_id']))
len(set(sub['sid']))

sub.shape[0]/float(eval.shape[0])

sub.shape[0]/float(len(set(sub['user_id']))*len(set(sub['sid'])))

#sub_max = pd.DataFrame(sub.groupby('sid').max()).reset_index()
#merged = pd.merge(sub,sub_max,on="sid")
#merged['plays_x'] = merged['plays_x']/merged['plays_y']
#sub_norm = merged[['user_id_x', 'sid', 'plays_x']]
#sub_norm.columns = ['user_id', 'sid', 'plays']

sample = random.sample(sub.index, int(sub.shape[0]*0.2))

trainsub = sub.copy()
trainsub.ix[trainsub.index.isin(sample),'plays'] = 0

testsub = sub.copy()
testsub.ix[~trainsub.index.isin(sample),'plays'] = 0

trainpivot = trainsub.pivot(index='user_id',columns='sid', values='plays')

user_index = pd.DataFrame(trainpivot.index).reset_index()
user_index.columns = [['user_index','user_id']]
trainsub = pd.merge(trainsub, user_index, on='user_id')
testsub = pd.merge(testsub, user_index, on='user_id')
song_index = pd.DataFrame(trainpivot.columns).reset_index()
song_index.columns = [['song_index','sid']]
trainsub = pd.merge(trainsub, song_index, on='sid')
testsub = pd.merge(testsub, song_index, on='sid')

M_train = trainpivot.as_matrix()
M_train = np.nan_to_num(M_train)

rowsum = np.sum(M_train,axis=1).reshape((-1,1))
rowsum[rowsum == 0] = 1E-16
M_train_norm = M_train/rowsum


idftrain = trainsub.copy()
idftrain.plays = 1
idf_train = idftrain.pivot(index='user_id',columns='sid', values='plays').as_matrix()
idf_train = np.nan_to_num(idf_train)
colsum = np.sum(idf_train,axis=0)
idf = np.log10(idf_train.shape[0]/colsum)
M_train_tfidf = M_train_norm*idf

omega = [tuple(x) for x in trainsub[['user_index','song_index']].values]
omega_test = [tuple(x) for x in testsub[['user_index','song_index']].values]

testplay = testsub[testsub.plays>0]
test_usergroup = testplay.groupby('user_index')
omegau_test = {}

for i in list(set(testplay.user_index.values)):
    omegau_test[i] = list(test_usergroup.get_group(i)['song_index'])
    
trainplay = trainsub[trainsub.plays>0]    
train_usergroup = trainplay.groupby('user_index')
omegau_train = {}

for i in list(set(trainplay.user_index.values)):
    omegau_train[i] = list(train_usergroup.get_group(i)['song_index'])

trainplay = trainsub[trainsub.plays>0] 
train_songgroup = trainplay.groupby('song_index')
omegav_train = {}

for i in list(set(trainplay.song_index.values)):
    omegav_train[i] = list(train_songgroup.get_group(i)['user_index'])   


testpivot = testsub.pivot(index='user_id',columns='sid', values='plays')
M_test = testpivot.as_matrix()
M_test = np.nan_to_num(M_test)




















con = sqlite.connect("lastfm_similars.db")
with con:
    sql = "SELECT * FROM similars_dest"
    lastfm_dest = psql.read_sql(sql, con)
    sql = "SELECT * FROM similars_src"
    lastfm_src = psql.read_sql(sql, con)
con.close()



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
p1 = { key: value for key, value in userdic.items() if len(value) > 27 }
# subset by number of plays
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
    
    alpha = 0.6
    gamma = 5
    
    for song1 in song_to_user:
        top_songs[song1] = []
        for song2 in song_to_user:
            user1 = song_to_user[song1]
            user2 = song_to_user[song2]
            sim_songs[song1][song2] = float(len(set(user1).intersection(user2))) / (len(user1)**alpha * len(user2)**(1-alpha))
            sim_songs[song1][song2] = sim_songs[song1][song2]**gamma
            top_songs[song1].append((song2, sim_songs[song1][song2]))
        top_songs[song1] = sorted(top_songs[song1], key = itemgetter(1), reverse = True)[0:500]
    
    recommend = {}       
    for user in user_to_song:
        songs = user_to_song[user]
        recommend[user] = []
        for song in songs:
            recommend[user] = recommend[user] + top_songs[song]
        testDict = defaultdict(float)
        for key, val in recommend[user]:
            testDict[key] += val
        recommend[user] = testDict.items()
        recommend[user] = sorted(recommend[user], key = itemgetter(1), reverse = True)
        recommend[user] = [x for x in recommend[user] if x[0] not in user_to_song[user]][0:500]                
        recommend[user] = [x[0] for x in recommend[user]]
    
    return recommend
    
   
    

def user_based_rec(user_to_song, song_to_user, N1):
    
    sim_users = np.zeros([N1,N1])
    
    alpha = 0.6
    gamma = 5
    
    for user1 in user_to_song:
        for user2 in user_to_song:
            song1 = user_to_song[user1]
            song2 = user_to_song[user2]
            sim_users[user1][user2] = float(len(set(song1).intersection(song2))) / (len(song1)**alpha * len(song2)**(1-alpha))
            sim_users[user1][user2] = sim_users[user1][user2]**gamma            
            
    recommend = {}
    for user1 in user_to_song:
        recommend[user1] = []
        for song in song_to_user:
            users = song_to_user[song]
            temp = 0
            for user2 in users:
                temp = temp + sim_users[user1][user2]
            recommend[user1].append((song, temp)) 
        recommend[user1] = sorted(recommend[user1], key = itemgetter(1), reverse = True)
        recommend[user1] = [x for x in recommend[user1] if x[0] not in user_to_song[user1]][0:500]        
        recommend[user1] = [x[0] for x in recommend[user1]]
            
    return recommend
    
    
def item_based_msd_sim(user_to_song, song_to_user):
    # Item-based similarity
    
    top_songs = {}
    
    for song1 in song_to_user:
        #get top songs and scores in list of tuples format
        top_songs[song1] = sorted(top_songs[song1], key = itemgetter(1), reverse = True)
    
    recommend = {}       
    for user in user_to_song:
        songs = user_to_song[user]
        recommend[user] = []
        for song in songs:
            recommend[user] = recommend[user] + top_songs[song]
        testDict = defaultdict(float)
        for key, val in recommend[user]:
            testDict[key] += val
        recommend[user] = testDict.items()
        recommend[user] = sorted(recommend[user], key = itemgetter(1), reverse = True)[0:500]
        recommend[user] = [x[0] for x in recommend[user]]
    
    return recommend

# In[8]:


item_based = item_based_rec(omegau, omegav, np.shape(Mtrain)[1])

user_based = user_based_rec(omegau, omegav, np.shape(Mtrain)[0])

item_based_msd = item_based_msd_sim(omegau, omegav)



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


# items based
apk_sum = 0

counter = 0
for i in omegau_test:
    if i in item_based:
        apk_sum += apk(omegau_test[i],item_based[i])
        counter += 1

map = apk_sum/counter
print counter
print map



#user based
apk_sum = 0

counter = 0
for i in omegau_test:
    if i in user_based:
        apk_sum += apk(omegau_test[i],user_based[i])
        counter += 1

map = apk_sum/counter
print counter
print map


# items based
apk_sum = 0

counter = 0
for i in omegau_test:
    if i in item_based:
        apk_sum += apk(omegau_test[i],item_based_msd[i])
        counter += 1

map = apk_sum/counter
print counter
print map
















item_based = item_based_rec(omegau_train, omegav_train, np.shape(M_train)[1])

user_based = user_based_rec(omegau_train, omegav_train, np.shape(M_train)[0])



# items based
apk_sum = 0

counter = 0
for i in omegau_test:
    if i in item_based:
        apk_sum += apk(omegau_test[i],item_based[i])
        counter += 1

map = apk_sum/counter
print counter
print map



#user based
apk_sum = 0

counter = 0
for i in omegau_test:
    if i in user_based:
        apk_sum += apk(omegau_test[i],user_based[i])
        counter += 1

map = apk_sum/counter
print counter
print map
