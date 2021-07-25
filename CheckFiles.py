import pickle
import os

with open('Finished_List.pkl', 'rb') as f:
    fin = pickle.load(f)

with open('Asset_List.pkl', 'rb') as f:
    to_do = pickle.load(f)

new_fin = []

file_loc = 'D:/Thesis_Data/News Headlines Investing_com'

count = 0
for ass in fin:
    list_dir = os.listdir(f'{file_loc}/{ass}')
    if len(list_dir) == 0:
        count+= 1
    else:
        new_fin.append(ass)

for new in new_fin:
    to_do.remove(new)
    

with open('Asset_List.pkl', 'wb') as f:
    pickle.dump(to_do, f)
