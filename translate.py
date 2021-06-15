import pandas as pd
from easynmt import EasyNMT
import os

import warnings
warnings.filterwarnings("ignore")


def _translateeasynmt(text, model):
    '''
    helpfunction: translating text to Norwegian using easynmt
    '''
    try:
        res = model.translate(text, source_lang="da", target_lang="no")
        return res
    except:
        print("\n.....................\n")
        print(text, "was not translated")
        print("\n.....................\n")
    return text


def easynmt_translate(df, col, model_name):
    '''
    df: dataframe
    col: text column to translate
    model: model used for translation ['opus-mt', 'mbart50_m2m' 'm2m_100_418M', 'm2m_100_1.2B']
    notes:
    opus-mt does not translate very well for da-no
    mbart50_m2m does not support da-no

    '''
    model = EasyNMT(model_name)
    df["easynmt_no" + "_" + model_name] = df[col].apply(_translateeasynmt, model=model)
    return df

if __name__ == "__main__":

    path = os.getcwd() + '/data/'
    df = pd.read_csv(path + 'dk.csv')

    df_trans = easynmt_translate(df, "cleaned", "m2m_100_418M")
    df_trans = easynmt_translate(df_trans, "cleaned", "m2m_100_1.2B")

    df_trans.to_csv(path + 'dk_test.csv', index=False)
    df = pd.read_csv(path + "dk_test.csv")
    print(df.head(6))
