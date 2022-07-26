import pandas as pd

# import wordery package files:
import sys
import preprocessing as pp


def main():
  # setup testing data:
  replace = [('<.*?>', ''), ('\S*\d\S*', '')]
  data = pd.DataFrame({'id': [0, 1, 2], 'text': ['tHis<?> is', 'a nice4 uNit', 'test']})
  data_clean = pd.DataFrame({'id': [0, 1, 2], 'text': ['this is', 'a unit', 'test']})
  
  # run cleaning function:
  data['text'] = pp.clean_text(data['text'], replace)
  
  # check our result matches hardcoded result:
  assert(data.equals(data_clean))
  
  
if __name__ == "__main__":
    main()