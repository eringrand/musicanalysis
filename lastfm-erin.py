import numpy as np
from collections import defaultdict
from scipy import sparse


f = open('kaggle_visible_evaluation_triplets.txt', 'r')
users = set()
artists = set()
numplays = {}
artistdic = defaultdict(list)
userdic = defaultdict(list)

data = f.read().split('\n')


for line in data:
    split = line.strip().split('\t')
    if len(split) != 3:
        print len(split), split
    else:
        userhash = split[0]
        artisthash = split[1]
        plays = split[2]
        users.add(userhash)
        artists.add(artisthash)
        numplays[artisthash, userhash] = plays
        #if artisthash not in userdic[userhash]:
        userdic[userhash].append(artisthash)
        #if userhash not in artistdic[artisthash]:
        artistdic[artisthash].append(userhash)
        

n = len(users)
m = len(artists)

users = list(users)
artists = list(artists)

user_dict = {}
i = 0
for user in users:
    user_dict[user] = i
    i += 1
    
artist_dict = {}
i = 0

for artist in artists:
    artist_dict[artist] = i
    i += 1
    
M = sparse.lil_matrix((n, m))

for user, artistlist in userdic.items():
    useridx = user_dict[user]
    for a in artistlist:
        artistidx = artist_dict[a]
        plays = numplays[a, user]
        M[useridx, artistidx] = plays
    
count = M.nonzero()
count = len(count[0])
print count
print len(users)*len(artists)
print float(count)/(len(users)*len(artists))