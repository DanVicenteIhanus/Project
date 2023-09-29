import pandas as pd

df=pd.read_csv('project_train.csv')

def clean(df):
    # Replace values in the DataFrame
    df.loc[df['loudness'] < -100, 'loudness'] = df.drop(df[df['loudness'] < -100].index)['loudness'].mean()
    df.loc[df['energy'] > 500, 'energy'] = df.drop(df[df['energy'] > 500].index)['energy'].mean()
    return df
