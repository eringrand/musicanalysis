import numpy as np
import pandas as pd
import matplotlib as plt
import matplotlib
import random

from matrix_factorization import MF
from matrix_factorization import MF2

from popularity_baseline import popularity_baseline

from artist_based import artist_based

from UserAndSongBasedRec import user_based_rec
from UserAndSongBasedRec import item_based_rec

matplotlib.style.use('ggplot')

### Data Loading Section
# load observation data
f = open("kaggle_visible_evaluation_triplets.txt", 'rb')
eval = pd.read_csv(f,sep='\t',header = None, names = ['user_id','sid','plays'])

# count number of songs per user and subset
userhist = eval.groupby('user_id').count()
userhist = pd.DataFrame(userhist).reset_index()
usersub = userhist[userhist['plays']>27]

# count number of users per song and subset
songhist = eval.groupby('sid').count()
songhist = pd.DataFrame(songhist).reset_index()
songsub = songhist[songhist['plays']>22]

# subset the whole dataset
sub = eval[eval['sid'].isin(songsub['sid'])]
sub = sub[sub['user_id'].isin(usersub['user_id'])]

# sampling/splitting the dataset
sample = random.sample(sub.index, int(sub.shape[0]*0.2))
trainsub = sub.copy()
trainsub.ix[trainsub.index.isin(sample),'plays'] = 0
testsub = sub.copy()
testsub.ix[~trainsub.index.isin(sample),'plays'] = 0

# generating a matrix out of dataframe
trainpivot = trainsub.pivot(index='user_id',columns='sid', values='plays')
# creating the mapping of user index to user id
user_index = pd.DataFrame(trainpivot.index).reset_index()
user_index.columns = [['user_index','user_id']]
trainsub = pd.merge(trainsub, user_index, on='user_id')
testsub = pd.merge(testsub, user_index, on='user_id')
# creating the mapping of song index to song id
song_index = pd.DataFrame(trainpivot.columns).reset_index()
song_index.columns = [['song_index','sid']]
trainsub = pd.merge(trainsub, song_index, on='sid')
testsub = pd.merge(testsub, song_index, on='sid')

# Generating the default M_train and M_test matrices
M_train = trainpivot.as_matrix()
M_train = np.nan_to_num(M_train)

testpivot = testsub.pivot(index='user_id',columns='sid', values='plays')
M_test = testpivot.as_matrix()
M_test = np.nan_to_num(M_test)

# creating omegau_test
testplays = testsub[testsub.plays>0]
test_usergroup = testsub[testsub.plays>0].groupby('user_index')
omegau_test = {}
for i in list(set(testplays.user_index.values)):
    omegau_test[i] = list(test_usergroup.get_group(i)['song_index'])
# creating omegau_train
trainplays = trainsub[trainsub.plays>0]
train_usergroup = trainplays.groupby('user_index')
omegau_train = {}
for i in list(set(trainplays.user_index.values)):
    omegau_train[i] = list(train_usergroup.get_group(i)['song_index'])
# creating omegav_train
train_songgroup = trainplays.groupby('song_index')
omegav_train = {}
for i in list(set(trainplays.song_index.values)):
    omegav_train[i] = list(train_songgroup.get_group(i)['user_index'])   
    
# creating tuple lists
omega = [tuple(x) for x in trainsub[trainsub.plays>0][['user_index','song_index']].values]
omega_test = [tuple(x) for x in testsub[trainsub.plays>0][['user_index','song_index']].values]

### Data Exploratory and Plotting Section



### Popularity Baseline Section
popbase = popularity_baseline(trainsub,omegau_train,omegau_test)
print "The popularity baseline based on counts is " + str(popbase[0]) + ' and based on plays is ' + str(popbase[1])

### Artist-based Popularity Baseline Section


### PMF Section
# Processing M_train, row normalization, column normalization, and tfidf
rowsum = np.sum(M_train,axis=1).reshape((-1,1))
rowsum[rowsum == 0] = 1E-16
M_train_rownorm = M_train/rowsum

colsum = np.sum(M_train,axis=0).reshape((1,-1))
colsum[colsum == 0] = 1E-16
M_train_colnorm = M_train/colsum

idftrain = trainsub.copy()
idftrain.plays = 1
idf_train = idftrain.pivot(index='user_id',columns='sid', values='plays').as_matrix()
idf_train = np.nan_to_num(idf_train)
colsum = np.sum(idf_train,axis=0)
idf = np.log10(idf_train.shape[0]/colsum)
M_train_tfidf = M_train_rownorm*idf

M_train_binary = M_train.copy()
M_train_binary[M_train_binary != 0] = 1

# Bunch of plotting stuff
# First and Second PMF plot
MAP, L =  MF(M_train,M_test,1.0, omega, omega_test, omegau_train, omegau_test,80,100)

plt.plot([0] + range(4,100,5), MAP)
plt.xlabel('Iteration')
plt.ylabel('MAP')
plt.title('MAP across iterations for default PMF')
plt.show()

plt.plot(L)
plt.xlabel('Iteration')
plt.ylabel('Log Joint Likelihood')
plt.title('Log Joint Likelihood across iterations for default PMF')
plt.show()

# Third PMF plot
plotdata = []

iter = 30
for var in [100, 10, 1, 0.01, 0.001]:
    for d in [10, 20, 40, 80]:
        plotdata.append([0,'Default',var,d,MF2(M_train,M_test,var, omegau_train, omegau_test, d, iter)])
        plotdata.append([1,'Row-norm',var,d,MF2(M_train,M_test,var, omegau_train, omegau_test, d, iter)])
        plotdata.append([2,'Col-norm',var,d,MF2(M_train,M_test,var, omegau_train, omegau_test, d, iter)])
        plotdata.append([3,'Binary',var,d,MF2(M_train,M_test,var, omegau_train, omegau_test, d, iter)])
        plotdata.append([4,'TF-IDF',var,d,MF2(M_train,M_test,var, omegau_train, omegau_test, d, iter)])
            
plotdata = pd.DataFrame(plotdata,columns=['method','methodname','variance','d','MAP'])

plotdata['logvar'] = np.log10(plotdata['variance'])

groups = plotdata.sort(['d','method']).groupby('d')

n=1
for name, group in groups:
    groups2 = group.groupby('method')
    plt.subplot(2,2,n)
    n += 1
    for name2, group2 in groups2:
        plt.plot(group2.logvar, group2.MAP, label=group2.iloc[0].methodname)
        plt.title('rank d = ' + str(name))
        plt.xlabel('Log10-Variance')
        plt.ylabel('MAP')
        plt.ylim([0,0.02])
        
plt.legend(bbox_to_anchor=[-0.1, 2.3], loc='center', ncol=5)
plt.suptitle("PMF over different hyper-parameters and scaling methods", fontsize = 20)
plt.show()


### Item and user-based section

#Call item-based recommendation function
item_based_map = item_based_rec(omegau_train, omegav_train, np.shape(M_train)[1], omegau_test)
print item_based_map

#Call user-based recommendation fucntion
user_based_map = user_based_rec(omegau_train, omegav_train, np.shape(M_train)[0], omegau_test)
print user_based_map


### K-Means section


### NMF section