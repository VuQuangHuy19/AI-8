import os
import numpy as np
import cv2
from tensorflow.keras.utils import to_categorical

def load_data(data_path):
    images, ages, genders = [], [], []
    for file in os.listdir(data_path):
        try:
            age, gender, _ = file.split('_')
            img = cv2.imread(os.path.join(data_path, file))
            img = cv2.resize(img, (224, 224)) / 255.0  # Normalize
            images.append(img)
            ages.append(int(age))
            genders.append(int(gender))
        except:
            continue
    return np.array(images), np.array(ages), to_categorical(genders, num_classes=2)
