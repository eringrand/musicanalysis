import numpy as np
import sqlite3 as sqlite
import pandas as pd
import pandas.io.sql as psql
import random
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

matplotlib.style.use('ggplot')

def to_percent(y, position):
    # Ignore the passed in position. This has the effect of scaling the default
    # tick locations.
    s = str(100 * y)

    # The percent symbol needs escaping in latex
    if matplotlib.rcParams['text.usetex'] == True:
        return s + r'$\%$'
    else:
        return s + '%'


# 34 user plays, 29 song plays
# 98 user plays, 52 song plays
# 27 user songcounts, 22 song usercounts

# http://www.kaggle.com/c/msdchallenge/data
f = open("kaggle_visible_evaluation_triplets.txt", 'rb')
eval = pd.read_csv(f,sep='\t',header = None, names = ['user_id','sid','plays'])

#userhist = eval.groupby('user_id').sum()
#userhist = pd.DataFrame(userhist).reset_index()
#usersub = userhist[userhist['plays']>98]

#songhist = eval.groupby('sid').sum()
#songhist = pd.DataFrame(songhist).reset_index()
#songsub = songhist[songhist['plays']>52]

#sub = eval[eval['sid'].isin(songsub['sid'])]
#sub = sub[sub['user_id'].isin(usersub['user_id'])]

#len(set(sub['user_id']))
#len(set(sub['sid']))

#sub.shape[0]/float(eval.shape[0])
#sub.shape[0]/float(len(set(sub['user_id']))*len(set(sub['sid'])))

# for number of songs instead of plays

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

##
sub_max = pd.DataFrame(sub.groupby('sid').max()).reset_index()
merged = pd.merge(sub,sub_max,on="sid")
merged['plays_x'] = merged['plays_x']/merged['plays_y']
sub_norm = merged[['user_id_x', 'sid', 'plays_x']]
sub_norm.columns = ['user_id', 'sid', 'plays']

sample = random.sample(sub_norm.index, int(sub_norm.shape[0]*0.2))
trainsub = sub_norm.copy()
trainsub.ix[trainsub.index.isin(sample),'plays'] = 0

testsub = sub_norm.copy()
testsub.ix[~testsub.index.isin(sample),'plays'] = 0

#trainpivot = trainsub.pivot(index='user_id',columns='sid', values='plays')
#user_index = trainpivot.index
#song_index = trainpivot.columns
#M_train = trainpivot.as_matrix()
#M_train = np.nan_to_num(M_train)

#testpivot = testsub.pivot(index='user_id',columns='sid', values='plays')
#M_test = testpivot.as_matrix()
#M_test = np.nan_to_num(M_test)

# http://labrosa.ee.columbia.edu/millionsong/sites/default/files/AdditionalFiles/unique_tracks.txt
#f = open("unique_tracks.txt", 'rb')
#unique_tracks = pd.read_csv(f,sep='<SEP>', header = None, names = ['tid', 'sid', 'artist_name', 'song_title'])

# http://labrosa.ee.columbia.edu/millionsong/sites/default/files/AdditionalFiles/unique_artists.txt
#f = open("unique_artists.txt", 'rb')
#unique_artists = pd.read_csv(f,sep='<SEP>',header = None, names = ['artist_id', 'artist_mbid', 'tid', 'artist_name'])

# http://labrosa.ee.columbia.edu/millionsong/sites/default/files/AdditionalFiles/tracks_per_year.txt
#f = open("tracks_per_year.txt", 'rb')
#tracks_per_year = pd.read_csv(f,sep='<SEP>', header = None, names =['year','tid', 'artist_name', 'song_title'])

# http://labrosa.ee.columbia.edu/millionsong/sites/default/files/AdditionalFiles/artist_location.txt
#f = open("unique_location.txt", 'rb')
#artist_location = pd.read_csv(f,sep='<SEP>', header = None, names = ['artist_id', 'latitude', 'longitude', 'artist_name', 'location'])

# http://www.ee.columbia.edu/~thierry/artist_similarity.db
#con = sqlite.connect("artist_similarity.db")
#with con:
#    sql = "SELECT * FROM similarity"
#    artist_sim = psql.read_sql(sql, con)
#con.close()

# http://www.ee.columbia.edu/~thierry/artist_term.db
#con = sqlite.connect("artist_term.db")
#with con:
#    sql = "SELECT * FROM artist_mbtag"
#    artist_mbtag = psql.read_sql(sql, con)
#    sql = "SELECT * FROM artist_term"
#    artist_term = psql.read_sql(sql, con)
#con.close()

# http://labrosa.ee.columbia.edu/millionsong/sites/default/files/AdditionalFiles/track_metadata.db
#con = sqlite.connect("track_metadata.db")
#with con:
#    sql = "SELECT * FROM songs"
#    track_metadata = psql.read_sql(sql, con)
#con.close()

# http://labrosa.ee.columbia.edu/millionsong/sites/default/files/lastfm/lastfm_tags.db
# tag values indicate popularity of the tags on last.fm as a whole
#con = sqlite.connect("lastfm_tags.db")
#with con:
#    sql = "SELECT tags.tag, tids.tid, tid_tag.val FROM tid_tag, tids, tags WHERE tags.ROWID=tid_tag.tag AND tid_tag.tid=tids.ROWID"
#    lastfm_tags = psql.read_sql(sql, con)
#con.close()

# http://labrosa.ee.columbia.edu/millionsong/sites/default/files/lastfm/lastfm_similars.db
# Similarity data from lastfm, the dest contains songs that consider the tid as similar while the src contains songs where tid considers as similar
#con = sqlite.connect("lastfm_similars.db")
#with con:
#    sql = "SELECT * FROM similars_dest"
#    lastfm_dest = psql.read_sql(sql, con)
#    sql = "SELECT * FROM similars_src"
#    lastfm_src = psql.read_sql(sql, con)
#con.close()

# histogram of number of song counts for users
usercount = eval.groupby('user_id').count()
usercount = pd.DataFrame(usercount).reset_index()
uservalues = usercount['plays'].values
#uservalues = np.log10(uservalues)
plt.hist(uservalues,50)
plt.title("Number of Users Who've played a Given Number of Songs")
plt.xlabel("Number of Songs Played")
plt.ylabel("Count of Users")
plt.show()


# histogram of number of user counts for songs
songcount = eval.groupby('sid').count()
songcount = pd.DataFrame(songcount).reset_index()
songvalues = songcount['plays'].values
songvalues = np.log10(songvalues)
plt.hist(songvalues,50)
plt.title("Number of Songs that have a Given Number of Listeners")
plt.xlabel("Number of Listeners (log scale)")
plt.ylabel("Count of Songs")
plt.show()

#histogram of number of plays for users
#userhist = eval.groupby('user_id').sum()
#userhist = pd.DataFrame(userhist).reset_index()
#uservalues = userhist['plays'].values
#uservalues = np.log10(uservalues)
#plt.hist(uservalues,50)
#plt.title("Histogram of number of plays for each user in log10")
#plt.xlabel("Number of plays in log10")
#plt.ylabel("Count")
#plt.show()

#cumulative sum for plays vs users
userplaycsum = userhist.sort('plays')
userplaycsum['plays'] = userplaycsum['plays'].cumsum()
userplaycsum['plays'] = userplaycsum['plays']/userplaycsum['plays'].max()
userplaycsum = userplaycsum.reset_index(drop=True).reset_index()
userplaycsum.plot('index', 'plays')
plt.legend().set_visible(False)
plt.title("Cumulative Sum of Number of Songs for each User")
plt.ylabel("Perecentage of Songs")
plt.xlabel("Number of Users")
#plt.vlines(60000,0,1,linestyles='dotted')
formatter = FuncFormatter(to_percent)
plt.gca().yaxis.set_major_formatter(formatter)
plt.legend().set_visible(False)
plt.show()


#cumulative sum for plays vs songs
songplaycsum = songhist.sort('plays')
songplaycsum['plays'] = songplaycsum['plays'].cumsum()
songplaycsum['plays'] = songplaycsum['plays']/songplaycsum['plays'].max()
songplaycsum = songplaycsum.reset_index(drop=True).reset_index()
songplaycsum.plot('index', 'plays')
plt.legend().set_visible(False)
plt.title("Cumulative Sum of Number of Users for each Song")
plt.ylabel("Perecentage of Users")
plt.xlabel("Number of Songs")
#plt.vlines(60000,0,1,linestyles='dotted')
formatter = FuncFormatter(to_percent)
plt.gca().yaxis.set_major_formatter(formatter)
plt.legend().set_visible(False)
plt.show()


# histogram of number of plays for songs
#songhist = eval.groupby('sid').sum()
#songhist = pd.DataFrame(songhist).reset_index()
#songvalues = songhist['plays'].values
#songvalues = np.log10(songvalues)
#plt.hist(songvalues,50)
#plt.title("Histogram of number of plays for each songs in log10")
#plt.xlabel("Number of plays in log10")
#plt.ylabel("Count")
#plt.show()

# cumulative sum of plays for artists
#songcsum = songhist.groupby('plays').count().cumsum()
#songcsum = pd.DataFrame(songcsum).reset_index()
#songcsum.plot('plays','sid')
#plt.show()

# cumulative sum for plays for artists in log
#songcsumlog = songcsum
#songcsumlog['plays'] = np.log10(songcsumlog['plays'])
#songcsumlog['sid'] = songcsumlog['sid']/songcsum['sid'].max()
#songcsumlog.plot('plays','sid')
#plt.show()
