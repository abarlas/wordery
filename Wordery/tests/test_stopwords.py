import pandas as pd

# import wordery package files:
import sys
import preprocessing as pp


def main():
  # setup testing data:
  stopwords = ['train', 'door', 'platform']
  data = pd.DataFrame({'id': [0, 1, 2], 'text': ['this train is', 'door a unit', 'test platform']})
  data_clean = pd.DataFrame({'id': [0, 1, 2], 'text': ['this is', 'a unit', 'test']})
  
  # run stopword function:
  data['text'] = pp.remove_stopwords(data['text'], stopwords)
  
  # check our result matches hardcoded result:
  assert(data.equals(data_clean))

  
if __name__ == "__main__":
    main()