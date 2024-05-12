import pandas as pd
import numpy as np
from ast import literal_eval
import re

def write_to_disk(df, file_name):
    """
    Write dataframe to disk

    Parameters
    ----------
    df : pandas DataFrame
        DataFrame to write to disk.
    file_name : str
        File path.

    Returns
    -------
    None.

    """
    df.to_csv(file_name, index=False)

# Convert duration column to number of minutes
def convert_duration(duration):
    """
    Parse scraped duration data

    Parameters
    ----------
    duration : str
        Scraped duration text data.

    Returns
    -------
    duration_mins : int
        Parsed duration in number of minutes.

    """
    duration = duration.split(' ')
    duration_mins = 0
    curr_min = 1/60
    for char in duration[::-1]:
        if 'min' in char:
            curr_min = 1
        elif 'hr' in char:
            curr_min = 60
        elif char.isnumeric():
            duration_mins += int(char) * curr_min
    return duration_mins

def calc_score(row):
    """
    Calculate mean store for each Title in the DataFrame

    Parameters
    ----------
    row : DataFrame row
        Row of the DataFrame for scraped data.

    Returns
    -------
    int
        Calculated mean score.

    """
    total, count = 0, 0
    for i in range(1,11):
        col = 'Score-'+str(i)
        count += int(row[col])
        total += int(row[col])*i
    return total/count

def clean_text(text):
    """
    Clean scraped text of escape characters

    Parameters
    ----------
    text : str
        Scraped text data.

    Returns
    -------
    text : str
        Cleaned text data.

    """
    text = text.replace('\t','').replace('\n',' ').replace('\r',' ')
    text = re.sub(' +', ' ',text).rstrip('\\').strip()
    return text

def zero_pad(date):
    """
    Parse scraped date into consistent format 

    Parameters
    ----------
    date : str
        Scraped date string.

    Returns
    -------
    str
        Parsed date string in format ready to convert into datetime.

    """
    if date is None:
        return
    if len(date) < 8:
        return date
    date = date.replace(' ','').replace(',','')
    month = date[:3]
    year = date[-4:]
    day = date[3:-4]
    if len(day) == 1:
        day = str(0) + day
    return ' '.join([month, day, year])


def preprocess_detailed_info(df, top_anime_df):
    """
    Preprocess anime info dataset

    Parameters
    ----------
    df : DataFrame
        Scraped anime info dataset.
    top_anime_df : DataFrame
        Scraped top anime dataset.

    Returns
    -------
    df : DataFrame
        Preprocessed anime info dataset.

    """
    df['Episodes'] = pd.to_numeric(df['Episodes'], errors='coerce')
    for i in range(1,10):
        col = 'Score-' + str(i)
        df.loc[df[col]=='?', col] = 0
        df[col] = df[col].astype('int64')
    
    df['Score'] = df.apply(calc_score, axis=1)
    df[['Aired_Start','Aired_End']] = df['Aired'].str.split('to', expand=True)
    df['Aired_Start'] = pd.to_datetime(df['Aired_Start'].apply(zero_pad), format='mixed', errors='coerce')
    df['Aired_End'] = pd.to_datetime(df['Aired_End'].apply(zero_pad), format='mixed', errors='coerce')
    df['Premiered_Season'] = df['Aired_Start'].dt.month%12//3+1
    
    df = df.merge(top_anime_df[['Id','Rank']], how = 'left', left_on = 'MAL_Id', right_on = 'Id').drop(['Id','Ranked'], axis=1)
    df = df.drop(['Synonyms_Name', 'Japanese_Name', 'English_Name', 'Demographic','Premiered','Aired'], axis=1)
    df['Duration'] = df['Duration'].apply(convert_duration)
    return df

def preprocess_reviews(df):
    """
    Preprocess anime review dataset

    Parameters
    ----------
    df : DataFrame
        Scraped anime review dataset.

    Returns
    -------
    df : DataFrame
        Preprocessed anime review dataset.

    """
    df.Review = df.Review.apply(clean_text)
    df.Tags = df.Tags.apply(literal_eval)
    df = df.rename(columns={'index': 'review_id'})
    return df

def preprocess_users(df):
    """
    Preprocess user ratings dataset

    Parameters
    ----------
    df : DataFrame
        Scraped user ratings dataset.

    Returns
    -------
    df : DataFrame
        Preprocessed user ratings dataset.

    """
    df = df.drop_duplicates(['Username','Anime_Id'], keep='last')
    df.Updated = pd.to_datetime(df.Updated)
    df.Start_Date = pd.to_datetime(df.Start_Date, errors='coerce')
    return df

def import_seasonal(file_name):
    """
    Improt and preprocess seasonal anime dataset

    Parameters
    ----------
    file_name : str
        File path to seasonal anime dataset.

    Returns
    -------
    df : DataFrame
        Imported and preprocessed seasonal anime dataset.

    """
    df = pd.read_csv(file_name, delimiter='|')
    df = df.drop_duplicates('MAL_Id', keep='last')
    df = df[(df.Score!=0) & (df.Members>500)]
    df['18plus'] = 0
    df.loc[df.Genres.str.contains('Hentai'), '18plus'] = 1
    df.Synopsis = df.Synopsis.str.split('Studio').str[0]
    df.Genres = df.Genres.apply(literal_eval)
    df.Themes = df.Themes.apply(literal_eval)
    return df