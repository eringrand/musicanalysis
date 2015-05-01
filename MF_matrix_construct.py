import numpy as np
#import pandas as pd

f = open('lastfm-dataset-360K/usersha1-artmbid-artname-plays.tsv', 'r')
data = []
i=0
for line in f:
    i = i + 1
    data.append(line)
    if i == 1000000:
        break
    

users = set()
artists = set()
artistname = {}

for i in range(len(data)-1):
    split = data[i].split('\t')
    userhash = split[0]
    artisthash = split[1]
    users.add(userhash)
    artists.add(artisthash)
        
users = list(users) #user hashes
artists = list(artists) #artist hashes

user_dict = {}
i = 0
for user in users:
    user_dict[user] = i
    i = i + 1
    
artist_dict = {}
i = 0
for artist in artists:
    artist_dict[artist] = i
    i = i + 1

M = np.zeros([len(users),len(artists)])
for i in range(len(data)-1):
    split = data[i].split('\t')
    userhash = split[0]
    artisthash = split[1]
    useridx = user_dict[userhash]
    artistidx = artist_dict[artisthash]
    M[useridx][artistidx] = split[3]

count = np.count_nonzero(M)
print count
print len(users)*len(artists)  
print float(count)/(len(users)*len(artists))
    
    
'''    
for i in range(len(data)-1):    
    if artisthash in artistname.keys():
        artistname[artisthash].append(split[2])
    else:
        artistname[artisthash] = []
        artistname[artisthash].append(split[2])
    plays = split[3]
'''


