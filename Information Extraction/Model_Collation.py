import json
import os
import pickle

batch_loc = 'C:/Users/User/OneDrive/University/Masters/Thesis/Labeling Articles/Batches/Batch1_Final'

labeled_files = os.listdir(f'{batch_loc}/Labels')
text_files = os.listdir(f'{batch_loc}/Text')

text_match = {}

reject_list = []

for text in text_files:
    t = text.split('.plain')[0]
    if f'{t}.ann.json' in labeled_files:
        text_match[t] = text
    else:
        reject_list.append(t)

_input_ = []

count = 0

replace_dict = {
                '/n':' ',
                "/'":"'",
                '&quot;':'"',
                '&amp;':'&',
                'Â©':'©',
                'â€™':'’',
                'â€œ': '"',
                'â€”':'-',
                'â€“': '-'
                }


def replacing(text, replace_dict):
    for key in replace_dict.keys():
        text = text.replace(key, replace_dict[key])
    return text

training_data = []

broken_count = []

entity_dict = {
  "e_3" : "Reason",
  "e_2" : "Defendant",
  "e_5" : "Penalties",
  "e_4" : "Outcome",
  "e_1" : "Platiff"
}


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

with open(f'Training_data_Batch1.pkl', 'wb') as f:
    pickle.dump(training_data, f)
