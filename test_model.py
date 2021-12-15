from tensorflow import keras
import cv2
model = keras.models.load_model("final_79.h5")
categories = ["NORMAL", "SHRINKAGE"]

def classify(image):
    try:
        Frame = cv2.imread(image)
        Frame = cv2.resize(Frame, (200,200))
        Frame = Frame.reshape(-1, 200, 200, 3)
    except Exception as e:
        print(e)
    predictions = model.predict([Frame])
    return predictions[0]

def predict(image):
    c = classify(image)
    print(c)
    p = 0
    for a in c:
        if a > p: p = a
    return categories[(list(c).index(p))]


image_path = "/home/amazing/Documents/MRI_API/test.jpg"
label = predict(image_path)
print(label)

