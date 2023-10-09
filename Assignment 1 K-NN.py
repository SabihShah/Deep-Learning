import cv2 as cv
import numpy as np
from collections import Counter
import os


def load_image_from_folder(folder_path, label):
    images = []
    labels = []
    for file in os.listdir(folder_path):
        if file.endswith('jpeg') or file.endswith('jpg') or file.endswith('png'):
            image_path = os.path.join(folder_path, file)
            image = cv.imread(image_path)
            image = cv.resize(image, (100, 100))
            if image is not None:
                images.append(image)
                labels.append(label)

    return images, labels

def euclidean_distance(x_test, x):
    return np.sqrt(np.sum((x_test - x) ** 2))

def KNN(folder1, folder2, test_images, k=3):
    test_outputs = []

    images1, labels1 = load_image_from_folder(folder1, label=os.path.basename(folder1))
    images2, labels2 = load_image_from_folder(folder2, label=os.path.basename(folder2))

    images = images1 + images2
    labels = labels1 + labels2

    images_X = np.array(images)
    labels_y = np.array(labels)

    for x_test in test_images:
        x_test = cv.resize(x_test, (100, 100))
        distances = []
        k_labels = []

        for x in images_X:
            dst = euclidean_distance(x_test, x)
            distances.append(dst)

        sorted_distances = np.argsort(distances)[:k]

        for i in sorted_distances:
            k_labels.append(labels_y[i])

        output_label = Counter(k_labels).most_common(1)[0][0]
        test_outputs.append(output_label)

    return test_outputs

def Accuracy(predictions, correct_labels):
    correct = 0
    total = len(correct_labels)

    for pred, true_label in zip(predictions, correct_labels):
        if pred == true_label:
            correct += 1

    accuracy = correct / total
    return accuracy

def loss(correct_labels, predictions):
    miscalculation = 0
    for true, pred in zip(correct_labels, predictions):
        if true != pred:
            miscalculation += 1
    # print(len(correct_labels))
    # print(miscalculation)
    error = miscalculation/len(correct_labels)

    return error


test_images_path = ['Person/images (2).jpeg', 
                    'test_images/Person/Untitled.jpg', 
                    'test_images/Person/images.jpg', 
                    'test_images/Person/images2.jpg']
true_labels = []
correct_labels = []
for label in test_images_path:
    true_labels.append(os.path.dirname(label).split('/')[-1:])

for i in true_labels:
    for j in i:
        correct_labels.append(j)


test_images = []
for i in range(len(test_images_path)):
    image = cv.imread(test_images_path[i])
    test_images.append(image)


predictions = KNN('Not a person', 'Person', test_images, k=3)
print('Predictions:', predictions)

accuracy = Accuracy(predictions, correct_labels)
print('Accuracy:', accuracy)

loss = loss(correct_labels, predictions)
print('Loss:', loss)
