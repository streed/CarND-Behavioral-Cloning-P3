import csv
import cv2
import numpy as np

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True


def load_data():
    images = []
    steering_angles = []

    with open('./data/driving_log.csv') as csv_file:
        reader = csv.reader(csv_file)

        for line in reader:
            image_path = line[0]
            steering_angle = float(line[3])

            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            images.append(image)
            steering_angles.append(steering_angle)

    images = np.array(images)
    steering_angles = np.array(steering_angles)

    return images, steering_angles

def augment_data(X, y):
    new_X = []
    new_y = []

    for i in range(len(X)):
        new_X.append(X[i])
        new_y.append(y[i])

        new_X.append(np.fliplr(X[i]))
        new_y.append(-y[i])

    return np.array(new_X), np.array(new_y)


def build_network():
    from keras.models import Sequential
    from keras.layers import Flatten, Dense, Lambda, Conv2D, MaxPooling2D, Cropping2D

    model = Sequential()
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160, 320, 3)))
    model.add(Cropping2D(cropping=((70, 25), (0, 0))))
    model.add(Conv2D(24,5,5, subsample=(2,2), activation='relu'))
    model.add(Conv2D(36,5,5, subsample=(2,2), activation='relu'))
    model.add(Conv2D(48,5,5, subsample=(2,2), activation='relu'))
    model.add(Conv2D(64,3,3, activation='relu'))
    model.add(Conv2D(64,3,3, activation='relu'))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))

    print(model.summary())

    model.compile(loss='mse', optimizer='adam')

    return model

def train_network(model, X, y):
    model.fit(X, y, validation_split=0.2, shuffle=True, epochs=10)
    return model


if __name__ == '__main__':
    X_data, y_data = load_data()
    X_data, y_data = augment_data(X_data, y_data)
    model = build_network()
    model = train_network(model, X_data, y_data)
    model.save('model.h5')
