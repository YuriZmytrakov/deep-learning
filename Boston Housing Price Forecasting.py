# Imports
import pandas as pd
from sklearn.datasets import load_boston
from keras import utils, Model , Input, optimizers
from keras.layers import Dense
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

def standardize(x):
    """Standardize the original data set."""
    return (x - x.mean(axis=0))/ x.std(axis=0)

def convert_age_to_binary(age):
    if age < 30:
        return 0
    else:
        return 1

X_column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']
y_column_names = ['PRICE'] # actually 'MEDV' in the dataset
X, y = load_boston(return_X_y=True)

# Take out 'AGE' col so it doesn't get standardized
age_col = X[:,6]

# normalize!
X = standardize(X)

# Put 'AGE' back in
X[:,6] = age_col

df_X = pd.DataFrame(X, columns=X_column_names)
df_y = pd.DataFrame(y, columns=y_column_names)

df_y['AGE'] = df_X['AGE'].apply(convert_age_to_binary)
df_X.drop('AGE', axis=1, inplace=True)

def build_model(input_data):
    losses = {
	    "Output_1_Price": "mean_squared_error",
	    "Output_2_Age": "binary_crossentropy",
    }
    metrics = {
        "Output_1_Price": "mse",
	    "Output_2_Age": "binary_accuracy",
    }
    opt = optimizers.gradient_descent_v2.SGD(learning_rate=0.005,momentum=0.9, clipnorm=1) 

    input_layer = Input(shape=input_data.shape[1], name='Input_Layer') 
    dense_1 = Dense(128, activation='relu', name='Dense_Layer_1')(input_layer)
    dense_2 = Dense(128, activation='relu', name='Dense_Layer_2')(dense_1)
    output_1_cont = Dense(1, activation='linear', name='Output_1_Price')(dense_1)
    output_2_binary = Dense(1, activation='sigmoid', name='Output_2_Age')(dense_2)

    outputs = {
        "Output_1_Price": output_1_cont,
	    "Output_2_Age": output_2_binary,
    }

    model = Model(inputs=input_layer, outputs=outputs)
    model.compile(optimizer=opt, loss=losses, metrics=metrics)
    return model

model = build_model(df_X)
utils.vis_utils.plot_model(model, show_shapes=True, show_layer_activations=True)
print("See directory for 'model.png'")