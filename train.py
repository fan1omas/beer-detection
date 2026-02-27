import numpy as np
import librosa 
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

X = []
y = [] 

def get_features(mfccs): 
    mean = np.mean(mfccs, axis=1)
    std = np.std(mfccs, axis=1)
    min_val = np.min(mfccs, axis=1)
    max_val = np.max(mfccs, axis=1)

    return np.concatenate([mean, std, min_val, max_val])

for folder_name, label in (('target', 'TARGET'), ('noise', 'NOISE')):
    for sound in os.listdir(f'data/{folder_name}'):
        filepath = os.path.join('data', folder_name, sound) 

        audio, sr = librosa.load(filepath)
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)

        X.append(get_features(mfccs))
        y.append(label)

X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(
    X,              
    y,              
    test_size=0.2,      
    shuffle=True,   
    #=42, # фиксированный случайный seed
)

model = model = RandomForestClassifier(n_estimators=200, max_depth=5, min_samples_leaf=5)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"\nТочность модели: {accuracy * 100:.1f}%")

joblib.dump(model, 'my_model.pkl')
print('Модель сохранена')