import numpy as np
import sqlite3 as sqlite
import pandas as pd
import pandas.io.sql as psql.
import matplotlib.pyplot as plt


# http://www.kaggle.com/c/msdchallenge/data
eval = pd.read_csv("kaggle_visible_evaluation_triplets.txt",sep='\t',header = None, names = ['user_id','song_id','plays'])
#eval.pivot(index='user_id',columns='song_id', values='plays')

# http://labrosa.ee.columbia.edu/millionsong/sites/default/files/AdditionalFiles/unique_tracks.txt
unique_tracks = pd.read_csv("unique_tracks.txt",sep='<SEP>', header = None, names = ['track_id', 'song_id', 'arist_name', 'song_title'])

# http://labrosa.ee.columbia.edu/millionsong/sites/default/files/AdditionalFiles/tracks_per_year.txt
tracks_per_year = pd.read_csv("tracks_per_year.txt",sep='<SEP>', header = None, names =['year','track_id', 'song_title'])

# http://labrosa.ee.columbia.edu/millionsong/sites/default/files/AdditionalFiles/artist_location.txt
artist_location = pd.read_csv("artist_location.txt",sep='<SEP>', header = None, names = ['artist_id', 'latitude', 'longitude', 'artist_name', 'location'])

# http://www.ee.columbia.edu/~thierry/artist_similarity.db
con = sqlite.connect("artist_similarity.db")
with con:
    sql = "SELECT * FROM similarity"
    artist_sim = psql.read_sql(sql, con)
con.close()

# http://www.ee.columbia.edu/~thierry/artist_term.db
con = sqlite.connect("artist_term.db")
with con:
    sql = "SELECT * FROM artist_mbtag"
    artist_mbtag = psql.read_sql(sql, con)
    sql = "SELECT * FROM artist_term"
    artist_term = psql.read_sql(sql, con)
con.close()

# http://labrosa.ee.columbia.edu/millionsong/sites/default/files/AdditionalFiles/track_metadata.db
con = sqlite.connect("track_metadata.db")
with con:
    sql = "SELECT * FROM songs"
    track_metadata = psql.read_sql(sql, con)
con.close()

import matplotlib
matplotlib.style.use('ggplot')

import matplotlib.pyplot as plt

# histogram of number of plays for users
userhist = eval.groupby('user_id').sum()
userhist = pd.DataFrame(userhist).reset_index()
uservalues = userhist['plays'].values
plt.hist(uservalues,50)
plt.show()

# cumulative sum of plays for users
usercsum = userhist.groupby('plays').count().cumsum()
usercsum = pd.DataFrame(usercsum).reset_index()
usercsum.plot('plays','user_id')
plt.show()

# cumulative sum for plays for users in log
usercsumlog = usercsum
usercsumlog['plays'] = np.log10(usercsumlog['plays'])
usercsumlog['user_id'] = usercsumlog['user_id']/usercsum['user_id'].max()
usercsumlog.plot('plays','user_id')
plt.show()

# cumulative sum for plays vs users
userplaycsum = userhist.sort('plays')
userplaycsum['plays'] = userplaycsum['plays'].cumsum()
userplaycsum['plays'] = userplaycsum['plays']/userplaycsum['plays'].max()
userplaycsum.plot('user_id', 'plays')
plt.show()

# histogram of number of plays for songs
songhist = eval.groupby('song_id').sum()
songhist = pd.DataFrame(songhist).reset_index()
songvalues = songhist['plays'].values
plt.hist(songvalues,50)
plt.show()

# cumulative sum of plays for artists
songcsum = songhist.groupby('plays').count().cumsum()
songcsum = pd.DataFrame(songcsum).reset_index()
songcsum.plot('plays','song_id')
plt.show()

# cumulative sum for plays for artists in log
songcsumlog = songcsum
songcsumlog['plays'] = np.log10(songcsumlog['plays'])
songcsumlog['song_id'] = songcsumlog['song_id']/songcsum['song_id'].max()
songcsumlog.plot('plays','song_id')
plt.show()

# cumulative sum for plays vs songs
songplaycsum = songhist.sort('plays')
songplaycsum['plays'] = songplaycsum['plays'].cumsum()
songplaycsum['plays'] = songplaycsum['plays']/songplaycsum['plays'].max()
songplaycsum.plot('song_id', 'plays')
plt.show()