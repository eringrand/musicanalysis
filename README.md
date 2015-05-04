# musicanalysis
Working with lastfm data to cluster like artists together based on genre and user plays. 

# data
http://www.dtic.upf.edu/~ocelma/MusicRecommendationDataset/

usersha1-profile.tsv
userhash
gender
age
country
join date

usersha1-artmbid-artname-plays.tsv
userhash
artist_musicbrainz_id
artistname
plays

359349 unique user hashes
160168 unique artist hashes
17559530 rows

```
>>> 17559530/float(359347*(186642+107373))
0.00016619937111663383
```
# kaggle data
kaggle_visible_evaluation.txt

110000 unique user hashes
163206 unique song hashes
1450934 rows
```
>>> 1450934/float(n*m)
8.082000104719857e-05
```
