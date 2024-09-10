import pickle
import json
import numpy as np
import os

# Global variables
__locations = None
__data_columns = None
__model = None

# Function to get estimated price
def get_estimated_price(location, total_sqft, bhk, bath):
    try:
        loc_index = __data_columns.index(location.lower())
    except:
        loc_index = -1

    x = np.zeros(len(__data_columns))
    x[0] = total_sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1

    return round(__model.predict([x])[0], 2)

# Function to get the list of locations
def get_location_names():
    return __locations

# Function to load the saved artifacts
def load_saved_artifacts():
    print("loading saved artifacts...start")
    global __data_columns
    global __locations

    # Absolute path to the artifacts directory
    artifacts_dir = r"C:\Users\Admin\OneDrive\Desktop\College\DWM\PBL\Server\artifacts"
    
    # Load columns.json
    columns_file = os.path.join(artifacts_dir, "columns.json")
    if not os.path.exists(columns_file):
        print(f"File not found: {columns_file}")
        return

    with open(columns_file, "r") as f:
        __data_columns = json.load(f)['data_columns']
        __locations = __data_columns[3:]  # first 3 columns are sqft, bath, bhk

    # Load banglore_home_prices_model.pickle
    global __model
    model_file = os.path.join(artifacts_dir, "banglore_home_prices_model.pickle")
    if __model is None:
        if not os.path.exists(model_file):
            print(f"File not found: {model_file}")
            return
        
        with open(model_file, "rb") as f:
            __model = pickle.load(f)
    print("loading saved artifacts...done")
