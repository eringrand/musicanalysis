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

def predict(u, v):
    product = np.dot(u, v)
    if product < 0:
        pred = 0.0
    elif product > 1:
        pred = 1.0
    else:
        pred = product
    return pred

RMSE = []
loglik = [] 
           
for iter in xrange(0,100):
    print iter
    for i in xrange(0, N1):
        sum1 = np.zeros([d,d])
        sum2 = np.zeros(d)
        if i in omegau.keys():
            for j in omegau[i]:
                sum1 = sum1 + np.outer(v[j],np.transpose(v[j]))
                sum2 = sum2 + np.dot(Mtrain[i][j],v[j])
            u[i] = np.dot(np.linalg.inv(t1 + sum1), sum2)
               
    for j in xrange(0, N2):
        sum1 = np.zeros([d,d])
        sum2 = np.zeros(d)
        if j in omegav.keys():
            for i in omegav[j]:
                sum1 = sum1 + np.outer(u[i],np.transpose(u[i]))
                sum2 = sum2 + np.dot(Mtrain[i][j],u[i])
            v[j] = np.dot(np.linalg.inv(t1 + sum1), sum2)
    
    sum3 = 0        
    for (i,j) in omega_test:
        prediction = predict(u[i], v[j])
        actual = Mtest[i][j]
        sum3 = sum3 + (prediction - actual)**2
    temp = (sum3/float(len(omega_test)))**0.5
    RMSE.append(temp)
    
    sum4 = 0
    for (i, j) in omega:
        sum4 = sum4 + 0.5/var*np.power(Mtrain[i][j] - np.dot(u[i], v[j]),2)
    sum4 = -sum4
    sum5 = 0
    for i in xrange(0, N1):
        sum5 = sum5 + 0.5*lamb*np.sum(u[i]**2)
    sum5 = -sum5
    sum6 = 0
    for j in xrange(0, N2):
        sum6 = sum6 + 0.5*lamb*np.sum(v[j]**2)
    sum6 = -sum6
    loglik.append(sum4+sum5+sum6)

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