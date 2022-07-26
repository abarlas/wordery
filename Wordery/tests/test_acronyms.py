import pandas as pd

# import wordery package files:
import sys
import preprocessing as pp


def main():
  # setup testing data:
  acro_dict = [(r"(\A|\s)u(|\s*)t(\s|$)", "unit test"),
             (r"(\A|\s)t(|\s*)i(\s|$)", "this is")]
  data = pd.DataFrame({'id': [0, 1, 2], 'text': ['ti', 'a', 'ut']})
  data_clean = pd.DataFrame({'id': [0, 1, 2], 'text': ['this is', 'a', 'unit test']})
  
  # run acronym function:
  data['text'] = pp.unpack_acronyms(data['text'], acro_dict)
  
  # check our result matches hardcoded result:
  assert(data.equals(data_clean))
  
  
if __name__ == "__main__":
    main()