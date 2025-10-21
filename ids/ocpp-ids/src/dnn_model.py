import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input

def create_model(input_shape, num_classes):
    '''
    ## MODIFIED: Creates a model that adapts to binary or multi-class problems.
    '''
    model = Sequential([
        # Use the modern Input layer, which is more flexible
        Input(shape=(input_shape,)),
        Dense(64, activation='relu'),
        Dense(64, activation='relu'),
        Dense(64, activation='relu'),
    ])

    if num_classes == 2:
        model.add(Dense(1, activation='sigmoid'))
    else:
        model.add(Dense(num_classes, activation='softmax'))
    # ---------------------------------------------------------
    
    return model
