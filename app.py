from flask import Flask, request, Response
app = Flask(__name__)
import numpy as np
from tensorflow import keras
import cv2
import jsonpickle

model = keras.models.load_model("final_79.h5")
def prepare(pic):
    pic = pic.reshape(-1, 200, 200, 3)
    return pic/255


categories = ["NORMAL", "SHRINKAGE"]

def classify(image):
    try:
        Frame = cv2.resize(image, (200,200))
        Frame = Frame.reshape(-1, 200, 200, 3)
    except Exception as e:
        print("------ERROR------->",e)
    predictions = model.predict([Frame])
    return predictions[0]

def predict(image):
    c = classify(image)
    print(c)
    p = 0
    for a in c:
        if a > p: p = a
    return categories[(list(c).index(p))]


app = Flask(__name__)

@app.route("/predict", methods=['POST'])
def get_n():
    r = request
    nparr = np.fromstring(r.data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    resp = predict(img)

   

    # build a response dict to send back to client
    response = {'message': 'image received. {} Shrinkage'.format(resp)}
    # encode response using jsonpickle
    response_pickled = jsonpickle.encode(response)
    print(resp)
    return Response(response=response_pickled, status=200, mimetype="application/json")


@app.route("/")
def home():
    return "MRI_MODEL_API"


if __name__ == '__main__':
    while True:
        try:
            app.run()
        except Exception as e:
            print(e)
            print("resatart....")

# #######################################################################################################

