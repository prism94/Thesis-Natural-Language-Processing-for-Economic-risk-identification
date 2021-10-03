# Thesis-Natural-Language-Processing-for-Economic-risk-identification


Collate News and Headlines and Articles by from Investing.com
Part 1: Collation of News headlines and hrefs:
This script collates news headlines for each asset saved in a list and pickled as ‘asset_list.pkl’
	Run ScriptManagement.py
o	Key files providing functions:
	RSS_news_scrapper_Headlines_def.py (set destination in this file)
	asset_list.pkl
Part 2: Collate Headlines into CSV
Collate headlines scrapped into one document.
	Collate_Headlines.py
Part 3: Newspaper Collation
Extract full articles from hrefs in collated file from Part 2.
	Get_Data_From_file.py

Collate labelled data from various sources:
Run the following scripts and manage the selenium driver to collate labelled datasets of news
USNEWS:
	USNEWS_Lawsuit_Scrapper.py (Run end code when ready to collate all headlines and hrefs)
Global News:
	GlobalNews_Lawsuit_Scrapper.py
These files collate Headlines and Hrefs into csv files
Collect full news articles through:
	Get_Data_From_file.py
Other scrappers were also built, however not used in the project for various labelling and restriction limitations. The code for these is included in legacy code.


Feature Selection:
Complete these after collating labelled dataset under classification….
Mutual Information:
Calculate mutual information of dataset:
	Feature_Selection.py
Clean Dataset:
Identify Problematic phrases and remove them from the dataset:
	Sentence_Feature_Analysis.py

Classification:
Headlines:
Pre-processing:
Load labelled data from the individual csv files, clean relevant text errors from scrapping
	Headline_Dataset_Preprocessing.py (Input collated csv files)
Glove embedding and tokenisation:
Create tokenizer for dataset, and pre-trained weights for glove embeddings
	Glove_To_Embedding.py

Train model variations for Headlines:
Train models on labelled data. Change model structure through ‘model_dict’. Also implement pre-trained weights or not by adjusting embedding layer.
	Headline_Classification_Modeling.py
Full Articles:
Collate full articles:
Collate the full articles into one large csv file
	Full_Article_Collation.py
Pre-process data and collate labelled dataset:
Create labelled dataset from csv file
	Dataset_Preprocessing.py
Glove embeddings and tokenization:
Create Glove embeddings and tokenisation of dataset.
	Glove_To_Embeddings.py
Model data on full articles:
With collated word embeddings and dataset, model data:
	Classification_Modelling_Full_Articles.py

BERT Implementation:
Run the code on both headline and full articles and adjust settings to deal with length of dataset. (Note: this code was run on Google colab as it was more computationally demanding)
	BERT.ipynb

Collate Model Results:
Collate model results using the saved model weights.
Record model trainable and non-trainable parameters:
	Interpret_Results_Parameters.py
Record Model Evaluation Metrics:
(Adjust embedding layer to include non-glove models)
	Interpret_Glove_Results.py

