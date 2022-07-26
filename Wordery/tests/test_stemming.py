import pandas as pd

# import wordery package files:
import sys
import preprocessing as pp


def main():
  # setup testing data:
  data = pd.DataFrame({'id': [0, 1, 2], 'text': ['here is', 'a stemming unit', 'test']})
  data_clean = pd.DataFrame({'id': [0, 1, 2], 'text': ['here is', 'a stem unit', 'test']})
  
  # run stemming function:
  data['text_sno'] = pp.do_stemming(data['text'], stemmer='Snowball')
  data['text_ptr'] = pp.do_stemming(data['text'], stemmer='Porter')
  
  # check our result matches hardcoded result:
  assert(data['text_sno'].equals(data_clean['text']))
  assert(data['text_ptr'].equals(data_clean['text']))
  
  
if __name__ == "__main__":
    main()