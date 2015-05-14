N1 = np.shape(Mtrain)[0]
N2 = np.shape(Mtrain)[1]

d = 20
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
iter = 1
users1 = {}
songs1 = {}
for i in range(N1):
    users1[i] = np.where(Mtrain[i] != 0)[0]

for j in range(N2):
    songs1[j] = np.where(Mtrain[:,j] != 0)[0]


def predict(u, v):
    product = np.dot(u, v)
    if product < 0:
        pred = 0.0
    elif product > 1:
        pred = 1.0
    else:
        pred = product
    return pred


    # Performing matrix factorization
for c in range(iter):
    for i in range(N1):
        inner = np.dot(v[:,users1[i]], v[:,users1[i]].T)
        outer = np.dot(Mtrain[i][users1[i]], v[:,users1[i]].T)
        u[i] = np.dot((t1 + inner).I * outer.T).T

    for j in range(N2):
        inner = np.dot(u[songs1[j]].T * u[songs1[j]])
        outer = np.dot(Mtrain[songs1[j],j] * u[songs1[j]])
        v[:,j] = np.dot((t1 + inner).I, outer.T)

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
    
    
    
    
    
    
    from sklearn.cluster import KMeans
    kmeans_model = KMeans(n_clusters=10).fit_predict(u)
