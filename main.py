import bert_lib as bert
import sys
import json
import warnings
warnings.filterwarnings("ignore")
from transformers import logging
logging.set_verbosity_error()
import torch
from torch import cuda


if __name__ == "__main__":
    '''
    takes one argument which specifies the config file to configure datasets, models etc. 
    '''
    print("\n\n\n")
    print("\n...............................\n")
    print("RUNNING main.py")
    print("\n...............................\n")

    arguments = sys.argv[1:]
    config_name = arguments[0]
    print("CUDA Available: ", cuda.is_available(), flush=True)

    with open('configs/' + config_name, 'r') as f:
        config = json.load(f)


    print("\n...............................\n")
    print("Configs:", config)
    print("\n...............................\n")
    print("CUDA Available: ", torch.cuda.is_available())
    print("\n...............................\n")

    if config['evaluate'] == False:
        bert.run_bert_cv(config)

    elif config['evaluate'] == True:
        bert.run_bert_evaluation(config)

