import pyaudio
import wave
import cv2
import os
import numpy as np
import pickle
import time
from scipy.io.wavfile import read
from audio_to_mfcc import *

import warnings

warnings.filterwarnings("ignore")


def recognize():
    #voice authentication
    FORMAT = pyaudio.paInt16
    CHANNELS = 2
    RATE = 44100
    CHUNK = 1024
    RECORD_SECONDS = 3
    FILENAME = "./voice_compare.wav"
    try:
        while True:
            audio = pyaudio.PyAudio()

            # start Recording
            stream = audio.open(format=FORMAT, channels=CHANNELS,
                                rate=RATE, input=True,
                                frames_per_buffer=CHUNK)

            print("recording...")
            frames = []

            for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
                data = stream.read(CHUNK)
                frames.append(data)
            print("finished recording")

            # stop Recording
            stream.stop_stream()
            stream.close()
            audio.terminate()

            # saving wav file
            waveFile = wave.open(FILENAME, 'wb')
            waveFile.setnchannels(CHANNELS)
            waveFile.setsampwidth(audio.get_sample_size(FORMAT))
            waveFile.setframerate(RATE)
            waveFile.writeframes(b''.join(frames))
            waveFile.close()

            modelpath = "./gmm_model/"

            gmm_files = [os.path.join(modelpath, fname) for fname in
                         os.listdir(modelpath) if fname.endswith('.gmm')]

            models = [pickle.load(open(fname, 'rb')) for fname in gmm_files]

            speakers = [fname.split("/")[-1].split(".gmm")[0] for fname
                        in gmm_files]

            if len(models) == 0:
                print("No Users in the database!!")
                break

            # read test file
            sr, audio = read(FILENAME)
            # extract mfcc features
            vector = extract_features(audio, sr)
            print('---->', vector.shape)
            print('$$$$$', type(vector))
            log_likelihood = np.zeros(len(models))

            # checking with each model one by one
            for i in range(len(models)):
                print(models[i])
                gmm = models[i]
                scores = np.array(gmm.score(vector))
                log_likelihood[i] = scores.sum()
            winner = np.argmax(log_likelihood)
            identity = speakers[winner]
            print('--->',type(identity))
            # if voice not recognized than terminate the process
            if identity == 'unknown':
                print("Not Recognized! Try again...")
                time.sleep(1.5)
                os.system('cls' if os.name == 'nt' else 'clear')
                continue
            else:
                print("Recognized as - ", identity)
            id = identity.split("_")
            time.sleep(4)
            # face recognition
            print("Keep Your face infront of the camera")

            recognizer = cv2.face.LBPHFaceRecognizer_create()
            recognizer.read('/Users/parthpatel/Desktop/Face-Voice-Recognition/trainer/'+id[1]+'/trainer.yml')
            cascadePath = "haarcascade_frontalface_default.xml"
            faceCascade = cv2.CascadeClassifier(cascadePath);

            font = cv2.FONT_HERSHEY_SIMPLEX

            # Initialize and start realtime video capture
            cam = cv2.VideoCapture(0)
            cam.set(3, 640)  # set video widht
            cam.set(4, 480)  # set video height

            # Define min window size to be recognized as a face
            minW = 0.1 * cam.get(3)
            minH = 0.1 * cam.get(4)

            while True:
                ret, img = cam.read()
                # img = cv2.flip(img, -1) # Flip vertically
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                faces = faceCascade.detectMultiScale(
                    gray,
                    scaleFactor=1.2,
                    minNeighbors=5,
                    minSize=(int(minW), int(minH)),
                )
                if len(faces) ==1:
                    for (x, y, w, h) in faces:
                        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        id, confidence = recognizer.predict(gray[y:y + h, x:x + w])

                        # Check if confidence is less them 100 ==> "0" is perfect match
                        if (confidence < 50):
                            name = identity
                            confidence = "  {0}%".format(round(100 - confidence))
                            print(name,",You are welcomed!")
                            exit(0)
                        else:
                            name = "unknown"
                            confidence = "  {0}%".format(round(100 - confidence))

                    cv2.imshow('camera', img)

                    k = cv2.waitKey(10) & 0xff  # Press 'ESC' for exiting video
                    if k == 27:
                        break

            # release camera
            print("\n [INFO] Exiting Program and cleanup stuff")
            cam.release()
            cv2.destroyAllWindows()

            if len(faces) == 0:
                print('There was no face found in the frame. Try again...')
                continue

            else:
                print("More than one faces found. Try again...")
                continue
    except KeyboardInterrupt:
        print("Stopped")
        pass

if __name__ == '__main__':
    recognize()

