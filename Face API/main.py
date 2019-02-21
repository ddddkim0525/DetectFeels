import requests
import scipy.misc
import scipy.ndimage
import csv
import numpy as np
import time


def import_csv(filename,data,flag = True):
    with open(filename) as datafile:
        csvReader = csv.reader(datafile)
        for row in csvReader:
            if(flag):
                picture = row[1].split()
                data.append([int(picture[i]) for i in range(1,len(picture))])
            else:
                data.append(int(row[0]))

subscription_key = "f837536443964071969e3ed942a4d577"
assert subscription_key

face_api_url = 'https://canadacentral.api.cognitive.microsoft.com/face/v1.0/detect'
image_url = 'https://how-old.net/Images/faces2/main007.jpg'

headers = {
    'Content-Type' : 'application/octet-stream',
    'Ocp-Apim-Subscription-Key': subscription_key }
    
params = {
    'returnFaceId': 'true',
    'returnFaceLandmarks': 'false',
    'returnFaceAttributes':'emotion',
}

data_set = []
label_set = []

filename = "train.csv"

#import_csv(filename, data_set)
#import_csv(filename,label_set)

#Upsampling images so file size is big enough for face api
'''
for i, data in enumerate(data_set):
    arr = np.array(data)
    arr.resize(48,48)
    arr = scipy.ndimage.zoom(arr,2,order=0)
    scipy.misc.imsave( 'test' + str(i) + '.jpg', arr )
'''
emotion_enum = ['anger','disgust','fear','happiness','sadness','surprise','neutral','contempt']
e2i = dict( (e,i) for i,e in enumerate(emotion_enum))

prediction = []
for i in range(547,1100):
    data = open('data/test'+str(i)+'.jpg','rb')
    response = requests.post(face_api_url, data=data, params=params, headers=headers)
    face = response.json()
    try:
        if face:
            emotions = face[0]['faceAttributes']['emotion']
            emotion = max(emotions,key=emotions.get)
            prediction.append(e2i[emotion])
        else:
            prediction.append(-1)
    except:
        prediction.append(-1)
    time.sleep(3.1)
    print(prediction[-1], end=' ')
np.savetxt('result.txt',prediction)
