import numpy as np
import joblib 
import librosa
import sounddevice as sd 
import subprocess
import config 


model = joblib.load('my_model.pkl')

DURATION = config.DURATION
SAMPLE_RATE = config.SAMPLE_RATE 
PATH = config.PATH

def get_features(mfccs): 
    mean = np.mean(mfccs, axis=1)
    std = np.std(mfccs, axis=1)
    min_val = np.min(mfccs, axis=1)
    max_val = np.max(mfccs, axis=1)

    return np.concatenate([mean, std, min_val, max_val])


try:
    print('Идет распознавание целевого звука')
    while True:
        recording = sd.rec(DURATION * SAMPLE_RATE,
                   samplerate=SAMPLE_RATE, 
                   channels=1,
                   dtype='float32')
        
        sd.wait() 

        audio = recording.flatten()  

        mfccs = librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, n_mfcc=13)
        features = get_features(mfccs).reshape(1, -1)

        prediction = model.predict(features)[0]

        if prediction == 'TARGET':
            subprocess.run([PATH])

except KeyboardInterrupt:
    pass