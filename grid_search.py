import bert_lib as bert
import sys
import json
from torch import cuda
import warnings
warnings.filterwarnings("ignore")
from transformers import logging
logging.set_verbosity_error()


if __name__ == "__main__":
    '''
    takes one argument which specifies the config file to configure datasets, models etc. 
    '''
    print("\n\n\n")
    print("\n...............................\n", flush=True)
    print("RUNNING hyperparameter optimization", flush=True)
    print("\n...............................\n", flush=True)

    arguments = sys.argv[1:]
    config_name = arguments[0]
    print("CUDA Available: ", cuda.is_available(), flush=True)

    f = open('configs/' + config_name, 'r')
    config = json.load(f)
    f.close()
    bert.run_hyperparameter_optimization(config)
