import pandas as pd

# import wordery package files:
import sys
import preprocessing as pp


def main():
  # setup testing data:
  data = pd.DataFrame({'id': [0, 1, 2], 'text': ['tihs is', 'a uint', 'tets']})
  data_clean = pd.DataFrame({'id': [0, 1, 2], 'text': ['this is', 'a unit', 'test']})
  
  # run spellcheck function:
  data['text'] = pp.spellcheck(data, 'text', 5)
  
  # check our result matches hardcoded result:
  assert(data.equals(data_clean))

  
if __name__ == "__main__":
    main()