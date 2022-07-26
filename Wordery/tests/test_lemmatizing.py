import pandas as pd

# import wordery package files:
import sys
import preprocessing as pp


def main():
  # setup testing data:
  data = pd.DataFrame({'id': [0, 1, 2], 'text': ['here be', 'a units', 'tests']})
  data_clean = pd.DataFrame({'id': [0, 1, 2], 'text': ['here be', 'a unit', 'test']})

  # run spellcheck function:
  data['text'] = pp.do_lemmatizing(data, 'text')
  
  # check our result matches hardcoded result:
  assert(data.equals(data_clean))
  
  
if __name__ == "__main__":
    main()