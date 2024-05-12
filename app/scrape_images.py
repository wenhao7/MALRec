import urllib.request
import pandas as pd
import os
import PIL
import numpy as np

df = pd.read_csv('data/seasonal_anime.csv', delimiter='|')
df = df.drop_duplicates('MAL_Id', keep='last')

# Sample images from Top 500 popular titles 
df = df.sort_values('Members', ascending = False)[:500]
image_links = df.Image.sample(50)

# Send requests and save images
#for img in image_links:
#    urllib.request.urlretrieve(img, 'data/imgs/'+img.split('/')[-1])

image_files = os.listdir('data/imgs/')

def make_collage(image_files):
    collage = PIL.Image.new("RGB", (2000,1500), color=(255,255,255))
    h, w = 300, 200
    i, j = 0, 0
    for file in image_files:
        img = PIL.Image.open('data/imgs/'+file)
        img = img.resize((w, h))
        collage.paste(img, (j, i))
        j += w
        if j >= 2000:
            j //= 2000
            i += h
            if i >= 1500:
                return collage
    return collage

collage = make_collage(image_files)
collage.save('data/imgs/collage.png')        
        