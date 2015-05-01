# for the original last.fm dataset
f = open('usersha1-artmbid-artname-plays.tsv', 'r')
data = f.read().split('\n')

users = set()
artists = set()
artistname = {}

for i in range(len(data)):
    split = data[i].split('\t')
    userhash = split[0]
    artisthash = split[1]
    users.add(userhash)
    artists.add(artisthash)

    artistname = split[2]
    plays = split[3]

users = list(users)
artists = list(artists)

# for the kaggle last.fm dataset

import numpy as np

f = open('kaggle_visible_evaluation_triplets.txt', 'r')
users = set()
artists = set()

for line in f:
    split = line.split('\t')
    userhash = split[0]
    artisthash = split[1]
    users.add(userhash)
    artists.add(artisthash)

n = len(users)
m = len(artists)

matrix = np.zeros((n,m))