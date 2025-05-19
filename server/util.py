import pickle
from flask import json
import numpy as np

__locations = None
__data_columns = None
__model = None

def predict_price(location, sqft, bath, bhk):
    try:
        loc_index = __data_columns.index(location.lower()) 
    except:
        loc_index = -1

    x = np.zeros(len(__data_columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk

    if loc_index >= 0:
        x[loc_index] = 1

    return round(__model.predict([x])[0],2)
def get_location_names():
    return __locations

def load_saved_artifacsts():
    print("Loading saved artifacts...start")
    global __locations
    global __data_columns

    with open('artifacts/columns.json', 'r') as f:
        __data_columns = json.load(f)['data_columns']
        __locations = __data_columns[3:]       
    global __model
    with open('artifacts/home_prices_model.pickle', 'rb') as f:
        __model = pickle.load(f)
    print("Loading saved artifacts...done")                   


if __name__ == "__main__":
    load_saved_artifacsts()
    print(get_location_names())
    print(predict_price('1st Phase JP Nagar', 1000, 2, 2))
    print(predict_price('1st Phase JP Nagar', 1000, 2, 3))
    print(predict_price('Indira Nagar', 1000, 2, 2))
    print(predict_price('Indira Nagar', 1000, 2, 3))
    print(predict_price('Indira Nagar', 1000, 3, 2))
    print(predict_price('Indira Nagar', 1000, 3, 3))
    