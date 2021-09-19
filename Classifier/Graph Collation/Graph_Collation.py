import pickle
import matplotlib.pyplot as plt

file_loc = 'D:/Thesis_Model/Headlines - Non Pretrained'
outcome = 'Full_Articles'

models = ['BILSTM',
          'CNN',
          #'CNN_LSTM',
          'GRU',
          'LSTM',
          'RNN'
          ]

BERT = []#'History-Headlines-Final.pkl']#'History-Article-NonPretrained-1200.pkl']

data = {}

for model in models:
    loc = f'{file_loc}/{model}/History.pkl'
    with open(loc, 'rb') as f:
        history = pickle.load(f)
    
    data[f'{model}'] = {}
    for key in history.keys():
        data[f'{model}'][key] = history[key]

for b in BERT:
    loc = f'{file_loc}/BERT/{b}'
    with open(loc, 'rb') as f:
        history = pickle.load(f)
    
    data[f'BERT'] = {}
    for key in history.keys():
        data[f'BERT'][key] = history[key]

#Prep Data Tables
plotter = 'val_accuracy'
max_length = 15

for key in data.keys():
    y = data[key][plotter][:max_length]
    x = [i for i in range(1, max_length+1)]
    plt.plot(x, y, label=f'{key}')
plt.legend()
plt.show()
