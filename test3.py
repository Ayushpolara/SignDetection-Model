import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("C://Users//ayush//PycharmProjects//SignDetect//Model//keras_model.h5", "C://Users//ayush//PycharmProjects//SignDetect//Model//labels.txt")
offset = 20
imgSize = 300
labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]

char_count = {label: 0 for label in labels}
last_count = {label: None for label in labels}
sentence = []

def display_accuracy(img, label, index, accuracy):
    cv2.rectangle(img, (10, 10), (200, 50), (255, 255, 255), -1)
    cv2.putText(img, f"Detected Sign: {label}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cv2.putText(img, f"Accuracy: {accuracy:.2f}%", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

def display_in_sentence(img, label, count):
    cv2.rectangle(img, (x - offset, y - offset-50), (x - offset+90, y - offset-50+50), (255, 0, 255), cv2.FILLED)
    cv2.putText(img, label, (x, y - 26), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)
    cv2.putText(img, str(count), (x + 70, y - 26), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)

while True:
    success, img = cap.read()
    imgOutput = img.copy()
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]
        imgCropShape = imgCrop.shape
        aspectRatio = h / w
        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wCal + wGap] = imgResize
            prediction, index = classifier.getPrediction(imgWhite, draw=False)
            accuracy = prediction[index] * 100
            char_count[labels[index]] += 1
            last_count[labels[index]] = char_count[labels[index]] % 10
            display_accuracy(imgOutput, labels[index], index, accuracy)
            display_in_sentence(imgOutput, labels[index], last_count[labels[index]])

            user_input = cv2.waitKey(0)
            if user_input == ord('y'):
                sentence.append(labels[index])
            elif user_input == ord('n'):
                pass
            elif user_input == ord('s'):
                sentence.append(' ')

        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize
            prediction, index = classifier.getPrediction(imgWhite, draw=False)
            accuracy = prediction[index] * 100
            char_count[labels[index]] += 1
            last_count[labels[index]] = char_count[labels[index]] % 10
            display_accuracy(imgOutput, labels[index], index, accuracy)
            display_in_sentence(imgOutput, labels[index], last_count[labels[index]])

            user_input = cv2.waitKey(0)
            if user_input == ord('y'):
                sentence.append(labels[index])
            elif user_input == ord('n'):
                pass
            elif user_input == ord('s'):
                sentence.append(' ')

        cv2.rectangle(imgOutput, (x - offset, y - offset),
                      (x + w + offset, y + h + offset), (255, 0, 255), 4)
        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)

    cv2.imshow("Image", imgOutput)

    if cv2.waitKey(1) == ord('q'):
        break

print("Character Count:")
for char, count in char_count.items():
    print(f"{char}: {count % 10}")

print("Sentence:")
print(''.join(sentence))

cv2.destroyAllWindows()
