import pyaudio
import wave
import cv2
import os
import pickle
import time
import numpy as np
from PIL import Image
from scipy.io.wavfile import read
from sklearn.mixture.gmm import GMM
import warnings

warnings.filterwarnings("ignore")

from audio_to_mfcc import *


def train_user():
    print('start capturing video.....')
    time.sleep(3)
    cam = cv2.VideoCapture(0)
    cam.set(3, 640)  # set video width
    cam.set(4, 480)  # set video height

    face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    # For each person, enter one numeric face id'''
    face_id = input('enter user id=')
    name = input('enter user-name=')
    path = '/Users/parthpatel/Desktop/Face-Voice-Recognition/dataset/'+face_id
    os.mkdir(path)
    print("[INFO] Initializing face capture. Look at the camera and wait ...")
    # Initialize individual sampling face count
    count = 0
    while (True):
        ret, img = cam.read()
        # img = cv2.flip(img, -1) # flip video image vertically
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            count += 1

            # Save the captured image into the datasets folder
            cv2.imwrite("/Users/parthpatel/Desktop/Face-Voice-Recognition/dataset/"+face_id+"/User." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y + h, x:x + w])

            cv2.imshow('image', img)

        k = cv2.waitKey(100) & 0xff  # Press 'ESC' for exiting video
        if k == 27:
            break
        elif count >= 30:  # Take 30 face sample and stop video
            break
    # release camera
    print("[INFO] Exiting Program .......")
    cam.release()
    cv2.destroyAllWindows()

    # Path for face image database
    path = '/Users/parthpatel/Desktop/Face-Voice-Recognition/dataset/'+face_id

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    # function to get the images and label data
    def getImagesAndLabels(path):
        imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
        faceSamples = []
        ids = []
        for imagePath in imagePaths:
            if imagePath == path + '.DS_Store':
                continue
            PIL_img = Image.open(imagePath).convert('L')  # convert it to grayscale
            img_numpy = np.array(PIL_img, 'uint8')
            id = int(os.path.split(imagePath)[-1].split(".")[1])
            faces = detector.detectMultiScale(img_numpy)
            for (x, y, w, h) in faces:
                faceSamples.append(img_numpy[y:y + h, x:x + w])
                ids.append(id)
        return faceSamples, ids

    print("\n [INFO] Training faces. It will take a few seconds. Wait ...")
    faces, ids = getImagesAndLabels(path)
    recognizer.train(faces, np.array(ids))
    train_path = '/Users/parthpatel/Desktop/Face-Voice-Recognition/trainer/'+face_id
    os.mkdir(train_path)
    # Save the model into trainer/trainer.yml
    recognizer.write('/Users/parthpatel/Desktop/Face-Voice-Recognition/trainer/'+face_id+'/trainer.yml')  # recognizer.save() worked on Mac, but not on Pi

    # Print the numer of faces trained and end program
    print("\n [INFO] {0} faces trained. Exiting Program".format(len(np.unique(ids))))

    print('--------------------------------------------------------------------------------------------')
    time.sleep(5)
    print('--------------------------------------------------------------------------------------------')
    # We'll run the script for different people and store
    # the data into multiple files
    # Voice authentication
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 2
    RATE = 44100
    RECORD_SECONDS = 3
    WAVE_OUTPUT_FILENAME = "./voices/" + face_id

    os.mkdir(WAVE_OUTPUT_FILENAME)
    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT,
                     channels=CHANNELS,
                     rate=RATE,
                     input=True,
                     frames_per_buffer=CHUNK)

    print("* recording")

    print("...........")
    frames = []
    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
         audio_data = stream.read(CHUNK)
         frames.append(audio_data)

    print("* done recording")

    stream.stop_stream()
    stream.close()
    p.terminate()

    # # saving wav file of speaker
    waveFile = wave.open(WAVE_OUTPUT_FILENAME + '/' + name + '_sample_' + face_id + '.wav', 'wb')
    waveFile.setnchannels(CHANNELS)
    waveFile.setsampwidth(p.get_sample_size(FORMAT))
    waveFile.setframerate(RATE)
    waveFile.writeframes(b''.join(frames))
    waveFile.close()
    print("Done")

    # # path to training data
    source = WAVE_OUTPUT_FILENAME
    # # path to save trained model
    dest = "./gmm_model/"
    files = [os.path.join(source, f) for f in os.listdir(source) if f.endswith('.wav')]
    print(files[0])
    features = np.array([])
    sr, audio = read(files[0])
    # convert audio to mfcc
    vector = extract_features(audio,sr)
    features = vector
    # for more accurate weights
    features = np.vstack((features, vector))
    features = np.vstack((features, vector))
    print(features.shape)
    #Gaussian-mixture-model to save gmm-model
    gmm = GMM(n_components=16, n_iter=200, covariance_type='diag', n_init=3)
    gmm.fit(features)
    # # picklefile = f.split("\\")[-2].split(".wav")[0]+".gmm"
    # # model saving..
    pickle.dump(gmm, open(dest + name + '_' + face_id + '.gmm', 'wb'))
    print(name + ' ' + 'added......')
    features = np.asarray(())


if __name__ == '__main__':
    train_user()


