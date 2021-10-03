import pickle
import matplotlib.pyplot as plt

file_loc = 'D:/Thesis_Model/Headlines - Glove'

models = ['LSTM',
          'RNN',
          'CNN',
          'GRU',
          'BILSTM',
          ]

BERT = ['History-Headlines-Final.pkl']#'History-Article.pkl']

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

name_dict = {'RNN-relu-fat':'RNN',
             }

#Prep Data Tables
plotter = 'val_accuracy'#'val_loss'#
max_length = 15

for key in data.keys():
    if key in name_dict:
        keyer = name_dict[key]
    else:
        keyer = key
    y = data[key][plotter][:max_length]
    x = [i for i in range(1, max_length+1)]
    plt.plot(x, y, label=f'{keyer}')
plt.legend()
plt.show()

