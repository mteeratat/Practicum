import time
import cv2 as cv
import smtplib as sm
import os
from practicum import find_mcu_boards, McuBoard, PeriBoard
from requests import get, post
from line_notify import LineNotify


def notifymessage(message):
    payload = {"message": message}
    sendnotify(payload)

def notifypic(message, url):
    payload = {"message": message,
               "imageFile": open(url,'rb')}
    sendnotify(payload)

def sendnotify(payload, file = None):
    url = 'https://notify-api.line.me/api/notify'
    token = '2dlsMzR3c0HjNMYtZVKyt1Wou1dX02RLzs6sJRyW6iD'
    headers = {"content-type": "application/x-www-form-urlencoded",
               "Authorization": f"Bearer {token}"}
    #payload = {"message": message}
    r = post(url, headers=headers, data=payload, files=file)
    print(r.text)

def sendpic(txt, path, token):
    notify = LineNotify(token)
    notify.send(txt + ' checked in', path)  # send picture

#notifymessage("bung")

haar_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
people = ['You_know_who', 'Taro', 'prayuth', 'M']
#features = np.load('features.npy', allow_pickle=True)
#labels = np.load('labels.npy')
img = 0
path = '/home/pi/practicum/project/usb-example/python/pic/'
face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_trained.yml')
capture = cv.VideoCapture(0)
switch = 0
lst = [0] * len(people)
unknown = 0
finish = 0
nump = 1
token = '2dlsMzR3c0HjNMYtZVKyt1Wou1dX02RLzs6sJRyW6iD'

devices = find_mcu_boards()
mcu = McuBoard(devices[0])
peri = PeriBoard(mcu)
peri.get_switch()
peri.set_led(0,0)
peri.set_led(1,0)
peri.set_led(2,0)

while True:
    #capture = cv.VideoCapture("192.168.2.46:8080")
    blank, img = capture.read()
    img+=1
    img = cv.resize(img, (300,200))
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    #cv.imshow('Person', gray)

    # Detect the face in the image
    faces_rect = haar_cascade.detectMultiScale(gray, 1.1, 4)
 
    for (x,y,w,h) in faces_rect:
        faces_roi = gray[y:y+h,x:x+h]
        label, confidence = face_recognizer.predict(faces_roi)
        print(f'label = {people[label]} with a confidence of {confidence} lst[label]={lst[label]}')
        print(f'unknown={unknown}')

        if (unknown >=50):
            #red
            if(finish == 0):
                peri.set_led(0,1)
                peri.set_led(1,0)
                peri.set_led(2,0)
                cv.imwrite(os.path.join(path,'photo' + str(nump) + '.jpeg'),img)
                #notifymessage("Unknown")
                #notifypic("Unknown",path+'photo' + str(nump) + '.jpeg')
                sendpic("Unknown", path+"photo1.jpeg", token)                
                nump += 1
            lst = [0] * len(people)
            unknown = 0
            finish = 1
        if(lst[label]>=50):
            if(finish == 0):    
                peri.set_led(0,0)
                peri.set_led(1,0)
                peri.set_led(2,1)
                cv.imwrite(os.path.join(path,'photo' + str(nump) + '.jpeg'),img)
                #notifymessage(people[label]+' checked in')
                #notifypic(people[label] + ' checked in', path+'photo' + str(nump) + '.jpeg')
                sendpic(people[label], path+"photo1.jpeg", token)
                nump += 1
            #f=open("int.txt","w")
            #integer=1
            #f.write(str(integer))
            #f.truncate()
            unknown = 0
            lst = [0] * len(people)
            finish = 1
        if(lst[label]>=0 or unknown >= 0):
            #yellow
            if(finish == 0):
                peri.set_led(0,0)
                peri.set_led(1,1)
                peri.set_led(2,0)
        if (confidence >= 60 and confidence <= 100):
            lst[label] += 1
            cv.putText(img, str(people[label]), (x, y - 4), cv.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 0), thickness=2)
            cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)
            cv.imshow('Detected Face', img)
        elif(confidence < 60 or confidence > 100):
            unknown+=1
            cv.putText(img, "Unknown", (x,y-4), cv.FONT_HERSHEY_COMPLEX, 0.8, (0,0,255), thickness=2)
            cv.rectangle(img, (x,y), (x+w,y+h), (000,0,255), thickness=2)
            cv.imshow('Detected Face', img)
          
    if(cv.waitKey(1) & 0xFF == ord('d')):
        peri.set_led(0,0)
        peri.set_led(1,0)
        peri.set_led(2,0)
        break
