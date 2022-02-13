import pickle
import spacy
breakpoint()

with open('Training_data_Batch2.pkl', 'rb') as f:
  training_data = pickle.load(f)

model = spacy.blank('en')

###Will need to adjust for BERT
#ner = model.add_pipe('ner')
model.add_pipe('ner')
ner = model.get_pipe('ner')

for i, ann in training_data:
    for ent in ann['entities']:
        ner.add_label(ent[2])

#Need to filter on NER only for BERT -> filter on pipenames == 'ner'

itter = 100

optimizer = model.begin_training()
for it in range(itter):

    losses = {}

    for t, ann in tqdm(training_data):
        breakpoint()


breakpoint()
