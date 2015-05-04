import numpy as np
import matplotlib.pyplot as plt
from operator import itemgetter

# Create Mtrain and Mtest matrices here

N1 = np.shape(Mtrain)[0]
N2 = np.shape(Mtrain)[1]
  

omegau = {}
omegav = {}

for (i,j) in omega:
    if i in omegau.keys():
        omegau[i].append(j)
    else:
        omegau[i] = [j]
    if j in omegav.keys():
        omegav[j].append(i)
    else:
        omegav[j] = [i]

# Calculate var          
#var = ???????
d = 20
I = np.identity(d)
lamb = 10
t1 = lamb*var*I

mean = np.zeros([d])
v = np.empty([N2,d])
u = np.empty([N1,d])

for j in xrange(0, N2):
    v[j] = np.random.multivariate_normal(mean,(1/float(lamb))*I)
v = np.asmatrix(v)
u = np.asmatrix(u)


RMSE = {}
L={}
iter = 100
users = {}
movies = {}
for i in range(N1):
    users[i] = np.where(Mtrain[i] != 0)[0]

for j in range(movienum):
    movies[j] = np.where(Mtrain[:,j] != 0)[0]

# Performing matrix factorization
for c in range(iter):
    for i in range(usernum):
        inner = (v[:,users[i]] * v[:,users[i]].T)
        outer = np.asmatrix(Mtrain[i][users[i]]) * v[:,users[i]].T
        u[i] = ((t1 + inner).I * outer.T).T

    for j in range(movienum):
        inner = (u[movies[j]].T * u[movies[j]])
        outer = np.asmatrix(Mtrain[movies[j],j]) * u[movies[j]]
        v[:,j] = ((t1 + inner).I * outer.T)

    predict = (u * v).round()
    predict[predict>5.0] = 5.0
    predict[predict<1.0] = 1.0

    RMSE[c] = (np.mean((np.asarray(Mtest[Mtest!=0] - predict[Mtest!=0]))**2))**.5

    predict = (u * v)
    first = np.sum((np.asarray(Mtrain[Mtrain!=0] - predict[Mtrain!=0]))**2)/(2*var)
    second = (np.linalg.norm(u)**2) * lamb / 2
    third = (np.linalg.norm(v)**2) * lamb / 2
    L[c] = - first - second - third



# Part 1: RMSE and likelihood
fig = plt.figure()         
# RMSE 
plt.subplot(1,2,1)      
plt.plot(RMSE)
plt.xlabel('Iteration')
plt.ylabel('RMSE')

# Log Joint Likelihood
plt.subplot(1,2,2) 
plt.plot(L)
plt.xlabel('Iteration')
plt.ylabel('Log Joint Likelihood')




# K-means clustering 

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


k = 10
cluster, centroid, lik = kmeans(u, N1, k)
plt.plot(lik)
plt.xlabel('Iteration')
plt.ylabel('Likelihood')
plt.title('K='+ str(k))


for i in xrange(0,10):
    print 'Centroid: ', i
    score = {}
    for j in xrange(0, N2):
        score[j] = np.dot(centroid[i], v[j])
    sorted_score = sorted(score.items(), key = itemgetter(1), reverse=True)
    for i in xrange(0,10):
        #print movies[sorted_score[i][0]]
        #print songs in cluster center
    print '\n'