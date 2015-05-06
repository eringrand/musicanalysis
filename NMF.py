import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize


# X should be songs by users (i.e. N2 x N1)
X = M.T

rank = 25

W = np.random.uniform(size=[N2,rank])
H = np.random.uniform(size=[rank,N1])


divobj = []
for t in xrange(0,200):
    print t
    temp1 = normalize(np.transpose(W), norm='l1', axis=1)
    purple = X / (np.dot(W,H) + 10**(-16))
    H = H * np.dot(temp1,purple)
    temp2 = normalize(np.transpose(H), norm='l1', axis=0) 
    purple = X / (np.dot(W,H) + 10**(-16))
    W = W * np.dot(purple,temp2)
    temp = np.sum((X * np.log(1/(np.dot(W,H) + 10**(-16)))) + (np.dot(W,H)))
    divobj.append(temp)
  

# Plot objective function
plt.plot(divobj)
plt.xlabel('Iteration')
plt.ylabel('Objective function')
plt.title('NMF with Divergence Penalty')


# Top 10 words
W_norm = normalize(W, norm='l1', axis=0)


for col in range(rank):
    temp = np.argsort(W_norm[:,col])[::-1][0:10]
    print 'Column of W: ' + str(col)
    for j in temp:
        #index of song is j
        print index_to_song_dict[j] + ' ' + str(W_norm[:,col][j])
    print '\n'
    
    
    
    
# Loop over all users the below code 
rec_user = {}
for user in users:
    H_column = H[:,user]
    col = np.argmax(H_column)
    W_column = W[:,col]
    rec_user[user] = np.argsort(W_column)[::-1][0:500]
    
    
    
    
rec_user = {}
for user in users:
    song_ranks = np.dot(W, H[:,user])
    rec_user[user] = np.argsort(song_ranks)[::-1][0:500]
