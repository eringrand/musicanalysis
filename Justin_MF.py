import numpy as np

RMSE = {}
L={}
iter = 100
users = {}
movies = {}
for i in range(usernum):
    users[i] = np.where(matrix[i] != 0)[0]

for j in range(movienum):
    movies[j] = np.where(matrix[:,j] != 0)[0]

# Performing matrix factorization
for c in range(iter):
    for i in range(usernum):
        inner = (v[:,users[i]] * v[:,users[i]].T)
        outer = np.asmatrix(matrix[i][users[i]]) * v[:,users[i]].T
        u[i] = ((rr + inner).I * outer.T).T

    for j in range(movienum):
        inner = (u[movies[j]].T * u[movies[j]])
        outer = np.asmatrix(matrix[movies[j],j]) * u[movies[j]]
        v[:,j] = ((rr + inner).I * outer.T)

    predict = (u * v).round()
    predict[predict>5.0] = 5.0
    predict[predict<1.0] = 1.0

    RMSE[c] = (np.mean((np.asarray(testmatrix[testmatrix!=0] - predict[testmatrix!=0]))**2))**.5

    predict = (u * v)
    first = np.sum((np.asarray(matrix[matrix!=0] - predict[matrix!=0]))**2)/(2*variance)
    second = (np.linalg.norm(u)**2) * lmb / 2
    third = (np.linalg.norm(v)**2) * lmb / 2
    L[c] = - first - second - third

    print c, RMSE[c], L[c]
