# Ahegao

This repository is for pix2pix style transfer between normal facial expression and ahegao.

The training set of aligned pairs is automatically created by mining ahegao videos from tiktok.

### Getting videos
The [python api client](https://github.com/tolgatasci/musically-tiktok-api-python.git) is used for download from tiktok.
Login identifiers must be reverse-engineered from tiktok, by capturing packets. 
Specifically: `device_id`, `iid`, `openudid`.
If registered to tiktok without password, password needs to be set for successful registration via API.

Listing of videos by tags works when we have hashtag ID by method `list_hashtag`. 

For textual hashtag, we can obtain its ID by searching hashtags using method `search_hashtag`.

Hashtag IDs are autoincrement, thus we can crawl all hashtags over ids. 
 
The ahegao frame in video is detected by face emotion detector.
