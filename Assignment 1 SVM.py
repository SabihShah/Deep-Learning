import cv2 as cv
import numpy as np
import os
import matplotlib.pyplot as plt


class SVMClassifier:
    def __init__(self, learning_rate=0.01, num_epochs=1000, C=1.0):
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.C = C  # Regularization parameter
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        num_samples, num_features = X.shape
        self.weights = np.zeros(num_features)
        self.bias = 0

        for _ in range(self.num_epochs):
            for i in range(num_samples):
                condition = y[i] * (np.dot(X[i], self.weights) + self.bias) >= 1
                if condition:
                    self.weights -= self.learning_rate * (2 * self.C * self.weights)
                else:
                    self.weights -= self.learning_rate * (2 * self.C * self.weights - np.dot(X[i], y[i]))
                    self.bias -= self.learning_rate * y[i]

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        predictions = np.sign(linear_output)
        return predictions

def extract_features_from_images(folder_path):
    features = []
    labels = []    
        
    for file in os.listdir(folder_path):
        if folder_path == 'Person':
            label = 1  # Assign label 1 for 'Person'
        elif folder_path == 'Not a person':
            label = 0  # Assign label -1 for 'Not a person'
        else:
            continue  # Skip other subfolders
        if file.endswith('.jpeg') or file.endswith('.jpg') or file.endswith('.png'):
            image_path = os.path.join(folder_path, file)
            image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
            if image is not None:
                        # Extract features (e.g., using HOG, SIFT, or other methods)
                        # For simplicity, we'll flatten the image as an example
                flattened_image = image.flatten()
                features.append(flattened_image)
                labels.append(label)

    return features, labels

x_train_1, y_train_1 = extract_features_from_images('Not a person')
x_train_2, y_train_2 = extract_features_from_images('Person')

X_train = x_train_1 + x_train_2
y_train = y_train_1 + y_train_2

X_train = np.array(X_train)
X_train = X_train.reshape(-1, X_train.shape[1])
y_train = np.array(y_train)

svm = SVMClassifier(learning_rate=0.001, num_epochs=1000, C=1.0)
svm.fit(X_train, y_train)

# Make predictions on new image data
new_image_path = 'Not a person/images (1).jpg'
new_image = cv.imread(new_image_path, cv.IMREAD_GRAYSCALE)
if new_image is not None:
    new_features = new_image.flatten()

    prediction = svm.predict(new_features.reshape(1, -1))  # Reshape for single sample
    if prediction == 1:
        print('Prediction: Person')
    elif prediction == 0:
        print('Prediction: Not a person')
else:
    print('Failed to load the new image.')
