#Voice Biometrics Authentication and Face Recognition

Voice Biometrics Authentication using GMM and Face Recognition Using haarcascade

How to Run :

Install dependencies by running pip3 install -r requirement.txt

1.Run in terminal in following way :

To record new user :

  python3 voice_face_record.py
To Recognize user :

  python3 voice_face_recognize.py

Voice Authentication

For Voice recognition, GMM (Gaussian Mixture Model) is used to train on extracted MFCC features from audio wav file.


Face Recognition

Face Recognition system using haarcascade CascadeClassifier. The model is implemented using  OpenCV has been done for realtime face detection and recognition.
The model uses face encodings for identifying users.

The program uses a python dictionary for mapping for users to their corresponding face encodings.


How it works? Step-by-Step guide


MFCC features and Extract delta of the feature vector

def calculate_delta(array):
    rows,cols = array.shape
    deltas = np.zeros((rows,20))
    N = 2
    for i in range(rows):
        index = []
        j = 1
        while j <= N:
            if i-j < 0:
                first = 0
            else:
                first = i-j
            if i+j > rows -1:
                second = rows -1
            else:
                second = i+j
            index.append((second,first))
            j+=1
        deltas[i] = ( array[index[0][0]]-array[index[0][1]] + (2 * (array[index[1][0]]-array[index[1][1]])) ) / 10
    return deltas

#convert audio to mfcc features
def extract_features(audio,rate):    
    mfcc_feat = mfcc.mfcc(audio,rate, 0.025, 0.01,20,appendEnergy = True, nfft=1103)
    mfcc_feat = preprocessing.scale(mfcc_feat)
    delta = calculate_delta(mfcc_feat)

    #combining both mfcc features and delta
    combined = np.hstack((mfcc_feat,delta))
    return combined

Converting audio into MFCC features and scaling it to reduce complexity of the model. Than Extract the delta of the given feature vector matrix and combine both mfcc and extracted delta to provide it to the gmm model as input.


The User have to speak his/her name one time at a time as the system asks the user to speak the name for three times. It saves three voice samples of the user as a wav file.

The function extract_features(audio, sr) extracts 40 dimensional MFCC and delta MFCC features as a vector and concatenates all the three voice samples as features and passes it to the GMM model and saves user's voice model as .gmm file.

Voice Authentication

def recognize():
   # Voice Authentication
   FORMAT = pyaudio.paInt16
   CHANNELS = 2
   RATE = 44100
   CHUNK = 1024
   RECORD_SECONDS = 3
   FILENAME = "./test.wav"

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

   modelpath = "./gmm_models/"

   gmm_files = [os.path.join(modelpath,fname) for fname in
               os.listdir(modelpath) if fname.endswith('.gmm')]

   models    = [pickle.load(open(fname,'rb')) for fname in gmm_files]

   speakers   = [fname.split("/")[-1].split(".gmm")[0] for fname
               in gmm_files]

   #read test file
   sr,audio = read(FILENAME)

   # extract mfcc features
   vector = extract_features(audio,sr)
   log_likelihood = np.zeros(len(models))

   #checking with each model one by one
   for i in range(len(models)):
       gmm = models[i]         
       scores = np.array(gmm.score(vector))
       log_likelihood[i] = scores.sum()

   winner = np.argmax(log_likelihood)
   identity = speakers[winner]

   # if voice not recognized than terminate the process
   if identity == 'unknown':
           print("Not Recognized! Try again...")
           return

   print( "Recognized as - ", identity)

This part of the function recognizes voice of the user as the user have to speak his/her name as the system asks.


Load all the pre-trained gmm models and passes the new extracted MFCC vector into the gmm.score(vector) function checking with each model one-by-one and sums the scores to calculate log_likelihood of each model. Takes the argmax value from the log_likelihood which provides the prediction of the user with highest prob distribution.

If the user's voice matches than it will go onto the face recogniton part otherwise the function will terminate by showing an appropriate message.

Face Recognition

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

if __name__ == '__main__':
   recognize()

If the User's voice matches than the function will execute the face recognition part.

First it will load the cascade classifier to detect the face from the frame and than loads embeddings.pickle file which holds facial embeddings of authorized users.


Controlling the face recognition accuracy: The confidence value controls  with which the face is recognized, you can control it by changing the value which is here 0.5.the closer the confidence value to zero is perfect match.

references: Face:https://www.hackster.io/mjrobot/real-time-face-recognition-an-end-to-end-project-a10826
            voice:https://github.com/abhijeet3922/PyGender-Voice
