import sqlite3

conn = sqlite3.connect('lastfm_tags.db')



# Get list of all tags for a particular track id
tid = 'TRNIEVD128F147645F'
print 'We get all tags (with value) for track: %s' % tid
sql = "SELECT tags.tag, tid_tag.val FROM tid_tag, tids, tags WHERE tags.ROWID=tid_tag.tag AND tid_tag.tid=tids.ROWID and tids.tid='%s'" % tid
res = conn.execute(sql)
data_tags = res.fetchall()
print data_tags


conn.close()



# Create dictionaries for tracks to artists and titles
f = open('unique_tracks.txt', 'r')
data = []
for line in f:
    data.append(line)
f.close()

track_to_artist = {}
track_to_title = {}
song_to_track = {}

for i in range(len(data)):
    split = data[i].split('<SEP>')
    track_to_artist[split[0]] = split[2]
    track_to_title[split[0]] = split[3]
    song_to_track[split[1]] = split[0]



# Map songs id to track id
f = open('kaggle_visible_evaluation_triplets.txt', 'r')
data = []
for line in f:
    data.append(line)
f.close()
    

users = set()
songs = set()

for i in range(len(data)):
    split = data[i].split('\t')
    userhash = split[0]
    songhash = split[1]
    users.add(userhash)
    songs.add(songhash)
        
users = list(users) #user hashes
songs = list(songs) #artist hashes

i=0
for song in songs:
    if song not in song_to_track.keys():
        i = i + 1









