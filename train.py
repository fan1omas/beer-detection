import numpy as np
import librosa 
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


X = []
y = [] 

for folder_name, label in (('target', 'TARGET'), ('noise', 'NOISE')):
    for sound in os.listdir(f'data/{folder_name}'):
        filepath = os.path.join('data', folder_name, sound) 

        audio, sr = librosa.load(filepath)
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)

        X.append(np.mean(mfccs, axis=1))
        y.append(label)

X = np.array(X)
y = np.array(y)

print(f"Всего файлов: {len(X)}")
print(f"Форма X: {X.shape}")  # (34, 13)
print(f"Форма y: {y.shape}")  # (34,)

X_train, X_test, y_train, y_test = train_test_split(
    X,              
    y,              
    test_size=0.2,      
    shuffle=True,   
    random_state=42, # фиксированный случайный seed
)

model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"\nТочность модели: {accuracy * 100:.1f}%")

