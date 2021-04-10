# import pandas as pd
from flask import Flask, jsonify, request
import pickle
import cv2
import numpy as np
from tensorflow import keras
# from keras.models import load_model

def process_data(data):
    data = data.replace('[', '')
    data = data.replace(']', '')
    data = data.split(', ')
    lst = list(map(int, data))
    arr = np.array(lst)
    arr = arr.reshape(200, 200, 3)
    arr = arr.astype('uint8')
#     arr = str(arr)
    image = cv2.resize(arr, (64, 64))
    image = np.array(image)
    image = image.astype('float32')/255.0
    image = image.reshape(-1, 64, 64, 3)
    
    
    # load model
    my_model = keras.models.load_model("ASL1.h5")
    print('model loaded')
    
    result = my_model.predict(image)
    print('predicted successfully')
    
    result = np.argmax(result, axis = 1)
    data = str(result[0])
    return data

# app
app = Flask(__name__)

# routes
@app.route('/', methods=['POST'])

def predict():
    # get data
    data = request.get_json(force=True)
    data = str(data)
    data = process_data(data)
    
#     data = np.argmax(model.predict(image), axis=1)

#     # convert data into dataframe
#     data.update((x, [y]) for x, y in data.items())
#     data_df = pd.DataFrame.from_dict(data)

#     # predictions
#     result = model.predict(data_df)

#     # send back to browser
#     output = {'results': int(result[0])}

    # return data
    return jsonify(data)

if __name__ == '__main__':
    app.run(port = 5000, debug=True)

