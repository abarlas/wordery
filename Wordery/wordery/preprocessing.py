import re
import numpy as np
import pandas as pd
from collections import defaultdict
import nltk
from nltk.probability import FreqDist
from nltk.corpus import wordnet as wn
from nltk import pos_tag
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')


def clean_text(df_column, replace=[], master_clean=True, lowercase=True, caps_sentence=False, remove_space=True, prebuilt=True, caps_start=True):
  '''
  Function for basic preprocessing of text, making it lowercase 
  and removing HTML formatting, etc.
  
  Inputs:
  df_column - column of DataFrame you wish to alter e.g. df['blob']
  replace - list of tuples containing:
    * first element: the regex pattern to search for
    * second element: what to replace this pattern with
    * example: [('\s+', ' '), ('<.*?>', ''), ...]
  prebuilt - if True, overrides the 'replace' list and instead uses a premade list of pre-processing steps
  master_clean - if True, remove everything that isn't letters
  lowercase - if True, will turn all letters to lowercase
  remove_space - if True, remove all multiple white spaces and those in the start and end of docs
  caps_sentence - if True, capitalises first letter after sentence end
  caps_start - if True, capitalises first letter of the text
  
  Returns:
  df_column - DataFrame column with cleaned text
  '''
  
  to_replace = [('\s+', ' '), # # replaces white spaces to single whitespace (inc. linebreak)
             ('<.*?>', ''), # # removes html snippets
             (' \. ', '. '), # reformats periods with white space on both sides to period with only one whitespace
             ('//', '/'), # replaces double forward-slashes into single
             ('^\s', ''), # replaces initial leading whitespace with empty string
             ('\[.*?\]', ' '), # remove everything between []
             ('\/\*.*?;\}', ' '), # remove extra stubborn font formatting
             ('^:\.', ''), # replaces initial colon and period with empty string
             ('\xa0', ' '), # remove Latin1 non-breaking space
             ('^: ', '') # replaces initial colon with empty string
             ]
  
   # perform regex cleaning functions
  if prebuilt==True:
    replace=to_replace
  
  # turn all descriptions into string and lower case
  if lowercase:
    df_column = df_column.str.lower()
  
  if master_clean:
    df_column = df_column.apply(lambda row: re.split('([0-9,.?:;~!@#$%^&*()])=<>', row)[0]) # separate numbers and symbols from letters
    df_column = df_column.str.replace('[^a-zA-Z]', ' ', regex=True) # remove anything that isn't letters
  
  if remove_space:
    df_column = df_column.apply(lambda row: " ".join(row.split()))
  
  # capitalises first letter of text  
  if caps_start:
    df_column = df_column.str.replace('^[a-z]', lambda x: x.group(0).upper(), regex = True)
    
  # capitalises the first letter of a word at the beginning of the sentence and after punctuation
  if caps_sentence:
    df_column = df_column.str.replace("(?<=[\.\?\!]\s)([a-z])", lambda x: x.group(0).upper(), regex=True)
    
  for pattern, replacement in replace:
    df_column = df_column.str.replace(pattern, replacement, regex=True)
  
  return df_column


def remove_stopwords(df_column, stopwords):
  '''
  Function to remove stopwords from a particular column
  of text in a DataFrame.
  
  Inputs:
  df_column - DataFrame column you wish to alter e.g. df['blob']
  stopwords - list of stopwords we want to remove
  
  Returns:
  df_column - DataFrame column with stopwords removed
  '''
  df_column = df_column.apply(lambda row: ' '.join([word for word in row.split() if word not in stopwords]))
  
  return df_column


def spellcheck(df_data, col_name, spellcheck_freq):
  '''
  Function to spell check least common words in corpus

  Inputs:
  df_data - Dataframe you wish to alter 
  col_name - column name to alter
  spellcheck_freq - min token frequency, below which we 
                    conduct spellcheck
                    
  Outputs:
  df_column - Spellchecked DataFrame column
  '''
  from autocorrect import Speller
  spell = Speller(fast=True)

  # create dataframe with frequency of each unique token:
  desc_list = df_data[col_name].tolist() # convert corpus to list of documents
  tokens = ' '.join(desc_list).split() # join into one continuous list and then split by word
  # obtain frequency of each token in corpus, arrange in dataframe:
  tokens_freq = FreqDist(tokens)
  freq_frame = pd.DataFrame(tokens_freq.most_common(), columns=["token", "frequency"])

  # extract least common words to correct:
  least_common = list(freq_frame.loc[freq_frame['frequency'] <= spellcheck_freq]['token'].values)
  # spell check all least common words:
  least_common_spell = spell(' '.join(least_common)).split()
  # extract words that were corrected, in their original spelling:
  to_correct = [item for item in least_common if item not in least_common_spell]

  for index, row in df_data.iterrows():
    tokens = row[col_name].split()
    # iterate through tokens, correct ones in to_correct list:
    for i, word in enumerate(tokens):
      if word in to_correct:
        tokens[i] = spell(word)
    # join corrected tokens back into string, and save to row:
    resultwords = ' '.join(tokens)
    df_data.at[index, col_name] = resultwords
    
  return df_data[col_name]


def unpack_acronyms(df_column, acro_tuples):
  '''
  Function to unpack acronyms into whatever form is preferred.
  Adapted from original function by Alex White.
  
  Inputs:
  df_column - column of DataFrame you wish to alter e.g. df['blob']
  acro_tuples - list of tuples containing:
    * first element: the regex pattern of acronym to seach for
    * second element: what to replace this acronym pattern with
  
  Returns:
  df_column - DataFrame column with unpacked acronyms text
  '''
  # perform regex cleaning function
  for pattern, replacement in acro_tuples:
    df_column = df_column.str.replace(pattern, replacement, regex=True)
  
  # remove any redundant empty spaces
  df_column = df_column.apply(lambda row: " ".join(row.split()))
  
  return df_column


def do_stemming(df_column, stemmer='Snowball'):
  '''
  Function to stem words in a DataFrame column.
  
  Inputs:
  df_column - column of DataFrame you wish to alter e.g. df['blob']
  stemmer - which Stemmer to use    
    * Porter: original Porter stemmer
    * Snowball: Porter2 stemmer, generally agreed to be better
    
  Returns:
  df_column - DataFrame column with stemmed text
  '''
  
  # import relevant stemmer and conduct stemming:
  if stemmer == 'Porter':
    ps = nltk.stem.PorterStemmer()
    df_column = df_column.apply(lambda row: ' '.join([ps.stem(w) for w in row.split()]))
  elif stemmer == 'Snowball':
    sno = nltk.stem.SnowballStemmer('english')
    df_column = df_column.apply(lambda row: ' '.join([sno.stem(w) for w in row.split()]))
    
  return df_column


def do_lemmatizing(df_data, col_name):
  '''
  Function to lemmatize words in corpus

  Inputs:
  df_data - Dataframe you wish to alter 
  col_name - column name to alter
                    
  Outputs:
  df_column - Lemmatized DataFrame column
  '''
  # calling nltk lemmatizer:
  wnl = nltk.WordNetLemmatizer()

  # create dictionary transforming nltk pos tags into wordnet ones:
  tag_map = defaultdict(lambda : wn.NOUN) # anything else is noun
  tag_map['J'] = wn.ADJ
  tag_map['V'] = wn.VERB
  tag_map['R'] = wn.ADV

  # iterate through each row in your dataframe, lemmatizing:
  for index, row in df_data.iterrows():
    tokens = row[col_name].split()
    # pos tag each token and lemmatize it using its tag:
    tokens_lemma = [wnl.lemmatize(token, tag_map[tag[0]]) for token, tag in pos_tag(tokens)]
    # join lemmatized tokens back into string, and save to row:
    tokens_lemma = ' '.join(tokens_lemma)
    df_data.at[index, col_name] = tokens_lemma
  
  return df_data[col_name]

def pairwise_duplicate_eraser(col1, col2, df):
  
  '''
  Checks if two strings are duplicates or if a substring exists within another, extracts the longer of the two \
  col1: the name of the first dataframe column. \
  col2: the name of the second dataframe column. 
  Adapted from original function by Fernando Mendez.
  '''
  
  # transform two columns into type string
  df[col1] = df[col1].astype(str)
  df[col2] = df[col2].astype(str)
  
  # Get both strings cut to the smallest length
  df['len1'] = df[col1].str.len()
  df['len2'] = df[col2].str.len()
  df['min_len'] = df[['len1','len2']].min(axis=1).astype(int)
  
  # reset dataframe index
  df.reset_index(drop=True, inplace=True)
  
  # truncate both strings to the minimum length 
  coltrunk1 = [ df[col1][i][0:df['min_len'][i]] for i in range(len(df)) ]
  coltrunk2 = [ df[col2][i][0:df['min_len'][i]] for i in range(len(df)) ]
  
  # compare strings (in lower case)
  trunk_cols = pd.DataFrame(list(zip(coltrunk1, coltrunk2)), columns = ['c1','c2'])
  
  # boolean check if subset of shorter string is withi longer string
  trunk_cols['isSubset'] = trunk_cols.c1.str.lower() == trunk_cols.c2.str.lower()
  
  # merge data
  df['isSubset'] = trunk_cols.isSubset
  df['which_max'] = df[['len1','len2']].idxmax(axis=1)
  
  # check if subset is within longer string col2, replace with empty string
  df[col1] = np.where(
    df['isSubset'] & (df.which_max == 'len2'),
    '',
    df[col1]
  )
  
  # check if subset is within longer string col1, replace with empty string
  df[col2] = np.where(
    df['isSubset'] & (df.which_max == 'len1'),
    '',
    df[col2]
  )
  
  # drops helper/intermediate columns
  df.drop(columns = ['len1', 'len2','min_len', 'isSubset', 'which_max'], inplace = True)
  
  return df