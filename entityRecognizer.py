import argparse
import pandas as pd
import numpy as np
import re
import pickle
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint

parser = argparse.ArgumentParser(description="""Takes two arguments:
                                                the path of the excel file containing the column 'Narration', 
                                                the path to the directory where the resulting file 'result.xlsx'
                                                with 'Enitity' column is to be saved """)

parser.add_argument('-i','--inp', help='Path to the input excel file', required=True)
parser.add_argument('-o','--out', help='Path to save the output result.xlsx file', required=True)
args = vars(parser.parse_args())

def tokenise(narration):
    
    """
    Takes a narration string, and returns a list of tokens,
    obtained by splitting the string with the special characters inside the parenthesis:
    (,-_!/\)
    In the absence of all the mentioned characters, the string is split with simple whaitespace character.
    """
    
    narration = str(narration)
    tokens = re.split(', |_|-|!|/|\+', narration)
    if len(tokens) == 1:
        tokens = ''.join(tokens).split()
    return [token.lower() for token in tokens]

def filter_numbers(narration):
    
    """
    Takes a list of tokens, and checks if the token is purely numerical, in which case it is replaced 
    with a simple 'num' string formatted with the index of that token in the list.
    
    The numbers at the start and end of strings are completely removed from the string.
    """
    filtered_narration = []
    for token in narration:
        if token.isdigit():
            token = f'num{narration.index(token)}'
        else:
            token = re.sub("^\d+|\d+$", "", token)
        filtered_narration.append(token)
    return filtered_narration

df = pd.read_excel(args['inp'])
narrations = df['Narration'].apply(lambda x:str(x))

tokenised_X = []
for narration in narrations:
    tokenised_X.append(tokenise(narration))
    
print('Preprocessing data')
    
X_data = []
for narration in tokenised_X:
    X_data.append(filter_numbers(narration))
    
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
    
X_encoded = tokenizer.texts_to_sequences(X_data)
    
X = pad_sequences(X_encoded, padding='post')

model = load_model('model.hdf5')

print('Making Predictions')

y = np.argmax(model.predict(X), axis=-1)

predicted_entities = []
for i in range(len(X)):
    narration = tokenised_X[i]
    if y[i] >= len(narration):
        entity = 'nan'
    else:
        entity = narration[y[i]]
    predicted_entities.append(entity)
    
results = pd.DataFrame(df['Narration'])
    
results['Entity'] = pd.Series(predicted_entities)
                          
print('saving results')

results.to_excel(args['out']+'/results.xlsx')
                          
print('Done.')
