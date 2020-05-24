import argparse
import os
import sys
import torch
from models.SDNetTrainer import SDNetTrainer
from utils.Arguments import Arguments




opt=None

parser=argparse.ArgumentParser(description="SDNet")
parser.add_argument('command')
parser.add_argument('conf_file')

cmdline_args=parser.parse_args()
print(cmdline_args)
command=cmdline_args.command
conf_file=cmdline_args.conf_file

conf_args=Arguments(conf_file) 
opt=conf_args.readArguments()

opt['cuda']=torch.cuda.is_available()
opt['confFile']=conf_file
opt['datadir']=os.path.dirname(conf_file)

for key,val in vars(cmdline_args).items():
    if val is not None and key not in ['command','conf_file']:
        opt[key]=val 

print(opt)
