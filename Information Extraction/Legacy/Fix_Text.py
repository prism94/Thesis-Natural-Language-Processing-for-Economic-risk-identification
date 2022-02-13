import pandas as pd

df = pd.read_csv('C:/Users/User/OneDrive/Python_scripts/Thesis/Information_Extraction/Application/Application_Full_Full.csv')

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

df = df[~df['Article'].isnull()]
df['Article_fixed'] = ''

for i in range(len(df)):
    text = df['Article'].values[i]
    for key in replace_dict.keys():
        text.replace(key, replace_dict[key])
    df['Article_fixed'].values[i] = text
breakpoint()

#Turns out there is no need for this script as the Text does not have the same issues
