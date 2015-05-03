import numpy as np
import sqlite3 as sqlite
import pandas as pd
import pandas.io.sql as psql

# http://www.kaggle.com/c/msdchallenge/data
eval = pd.read_csv("kaggle_visible_evaluation_triplets.txt",sep='\t',header = None, names = ['user_id','artist_id','plays'])

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

