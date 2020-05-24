import json
import msgpack 
import multiprocessing 
import re
import string 
import torch 
from tqdm import tqdm 
from collections import Counter 
import os 


class Preprocess():
    def __init__(self,opt):
        self.opt=opt 
        self.spacyDir=opt['FEATURE_FLODER']
        self.train_file=os.path.join(opt['datadir'],opt['CoQA_TRAIN_FILE'])
        self.dev_file=os.path.join(opt['datadir'],opt['CoQA_DEV_FILE'])
        self.glove_file=os.path.join(opt['datadir'],opt['INIT_WORD_EMBEDDING_FILE'])
        self.glove_dim=300
        self.official='OFFICIAL' in opt 
        self.data_prefix='coqa-'

        