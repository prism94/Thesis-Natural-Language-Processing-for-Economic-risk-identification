import json
import os
import pickle

batch_loc = 'C:/Users/User/OneDrive/University/Masters/Thesis/Labeling Articles/Batches/Batch1'
proof_loc = 'C:/Users/User/OneDrive/Python_scripts/Thesis/Information_Extraction'

with open(f'{proof_loc}/Proofing_list.pkl', 'rb') as f:
    proofing_list = pickle.load(f)

with open(f'{proof_loc}/Dropping_list.pkl', 'rb') as f:
    dropped_list = pickle.load(f)


labeled_files = os.listdir(f'{batch_loc}/Labels')
text_files = os.listdir(f'{batch_loc}/Text')

text_match = {}

reject_list = []

"""
for text in text_files:
    t = text.split('.txt')[0]
    t = t.split('-')[1]
    if t in text_match:
        reject_list.append(t)
    else:
        text_match[t] = text
"""

for text in text_files:
    t = text.split('.plain')[0]
    if f'{t}.ann.json' in labeled_files:
        text_match[t] = text
    else:
        reject_list.append(t)

_input_ = []

count = 0

replace_dict = {'\n':' ',
                "/'":"'",
                '&quot;':'"',
                '&amp;':'&',
                'Â©':'©',
                'â€™':'’',
                'â€œ': '"',
                'â€”':'-',
                'â€“': '-',
                '—':'-'}


def replacing(text, replace_dict):
    for key in replace_dict.keys():
        text = text.replace(key, replace_dict[key])
    return text

training_data = []

broken_count = []

entity_dict = { "e_3" : "Reason",
  "e_2" : "Defendant",
  "e_5" : "Penalties",
  "e_4" : "Outcome",
  "e_1" : "Platiff"}


for l in labeled_files:

    key = l.split('.ann.json')[0]

    try:
        if key not in reject_list:

            with open(f'{batch_loc}/Labels/{l}', 'r', encoding='utf-8') as f:
                labeled = json.load(f)
            print('labels')
            with open(f'{batch_loc}/Text/{text_match[key]}', 'r', encoding='utf-8') as f:
                _text_ = f.read()
            print('text')

            _text_ = _text_.split('<pre id="s1v1">')[1]
            _text_ = _text_.split('</pre>')[0]
            _text_ = replacing(_text_, replace_dict)
            #paragraphs = _text_.split('/n')        ###Will need to fix and test text allignment
            count_differ = 0

            training = []
            for lab in labeled['entities']:
                ent = lab['classId']
                start = lab['offsets'][0]['start']
                word = lab['offsets'][0]['text']
                start_len = len(word)
                word = replacing(word, replace_dict)
                end_len = len(word)
                count_differ = (end_len - start_len)
                loc_word = _text_[(start):(len(word)+start)]
                print(word, loc_word)
                if loc_word != word:
                    print('Did not work')
                    breakpoint()
                else:
                    training.append((start, len(word)+start, entity_dict[ent]))

            training_data.append((_text_, {'entities':training}))

            count += 1
            print('Done')
    except Exception as e:
        print('did not work')
        print(e)
        breakpoint()
        broken_count.append(l)

def anaylise_data(text, labels):

    checker = True

    for i in range(len(labels)):
        print('\n\n')
        start = labels[i][0]
        end = labels[i][1]
        lab = labels[i][2]
        print(text[:start])
        print('\n\n')
        print(f'{lab}')
        print('\n\n')
        print(text[start:end])
        print('\n\n')
        print(text[end:])
        checking_ = False
        while checking_ == False:
            checking = input('Correct?: ')
            if checking == 'y':
                break
            else:
                if checking == 'n':
                    checker = False
                    break

        if checker == False:
            break

    print('\nDone\n')
    return checker

breakpoint()


if len(labeled_files) == len(training_data):
    for i in range(len(labeled_files)):

        file = labeled_files[i]
        cont = True

        if file in proofing_list:
            cont = False
        else:
            if file in dropped_list:
                cont = False

        check = anaylise_data(training_data[i][0], training_data[i][1]['entities'])

        if check == True:
            proofing_list.append(file)
            with open(f'{proof_loc}/Proofing_list.pkl', 'wb') as f:
                pickle.dump(proofing_list, f)
        else:
            dropped_list.append(file)
            with open(f'{proof_loc}/Dropping_list.pkl', 'wb') as f:
                pickle.dump(dropped_list, f)
