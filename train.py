import numpy as np
import librosa 
import os

X = []
y = [] 

for folder_name, label in (('target', 'TARGET'), ('noise', 'NOISE')):
    for sound in os.listdir(f'data/{folder_name}'):
        filepath = os.path.join('data', folder_name, sound) 
        print(filepath)

        