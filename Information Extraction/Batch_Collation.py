import pickle
import numpy as np

dir_ = 'C:/Users/User/OneDrive/Python_scripts/Thesis/Information_Extraction'

files = [
        'Training_data_Batch1.pkl',
        'Training_data_Batch2.pkl',
        'Training_data_Batch3.pkl'
        ]

training_data = []

for file in files:
    with open(f'{dir_}/{file}', 'rb') as f:
        data = pickle.load(f)
    if training_data == []:
        training_data = data
    else:
        training_data += data

replacement_dict = {'Penalty':'Penalties',
                    'Plaintiff':'Platiff',
                    'PLAINTIFF':'Platiff',
                    'REASON':'Reason',
                    'PENALTY':'Penalties',
                    'DEFENDANT':'Defendant',
                    'OUTCOME':'Outcome'
                    } #I know this is incorrectly spelt but it is consistent with the rest of the code

for i in range(len(training_data)):
    for n in range(len(training_data[i][1]['entities'])):
        type = training_data[i][1]['entities'][n][2]
        if type in replacement_dict:
            change = list(training_data[i][1]['entities'][n])
            change[2] = replacement_dict[type]
            training_data[i][1]['entities'][n] = tuple(change)


ran = np.random.permutation(len(training_data))
training_data = np.array(training_data)[ran].tolist()

with open(f'{dir_}/Compiled_Labeled_Data_Final.pkl', 'wb') as f:
    pickle.dump(training_data, f)
