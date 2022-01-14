import os
import time
import sys
import shutil
import random
from argparse import ArgumentParser
import numpy as np
import torch
import torch.utils.data
import torch.nn.functional as F
torch.multiprocessing.set_sharing_strategy('file_system')
import train_utils as utils
from model import Network


### get parameters
parser = ArgumentParser()

# main parameters (required)
parser.add_argument('exp_name', type=str, help='exp name')

# main parameters (optional)
parser.add_argument('--device', type=str, default='cuda:0', help='cpu or cuda:x for using cuda on GPU number x')
parser.add_argument('--seed', type=int, default=3124256514, help='random seed (for reproducibility) [specify -1 means to generate a random one]')

# network settings
parser.add_argument('--model_epoch', type=int, default=-1)

# parse args
conf = parser.parse_args()
conf.flog = None

# load train config
train_conf = torch.load(os.path.join(conf.exp_name, 'conf.pth'))

# set up device
device = torch.device(conf.device)
print(f'Using device: {device}')

# find the model_epoch
if conf.model_epoch < 0:
    for item in os.listdir(os.path.join(conf.exp_name, 'ckpts')):
        if '_net_network.pth' in item:
            conf.model_epoch = max(int(item.split('_')[0]), conf.model_epoch)

# check if eval results already exist. If so, delete it.
result_dir = os.path.join(conf.exp_name, f'randgen-{conf.model_epoch}')
if os.path.exists(result_dir):
    response = input('Eval results directory "%s" already exists, overwrite? (y/n) ' % result_dir)
    if response != 'y':
        sys.exit()
    shutil.rmtree(result_dir)
os.mkdir(result_dir)
print(f'\nTesting under directory: {result_dir}\n')

# disable the variational part -- only useful during training
train_conf.probabilistic = False

# create models
network = Network(train_conf)
utils.printout(conf.flog, '\n' + str(network) + '\n')

# load pretrained model
print('Loading ckpt from ', os.path.join(conf.exp_name, 'ckpts'), conf.model_epoch)
data_to_restore = torch.load(os.path.join(conf.exp_name, 'ckpts', '%d_net_network.pth' % conf.model_epoch))
network.load_state_dict(data_to_restore, strict=False)
print('DONE\n')

# send parameters to device
network.to(device)

# set eval mode
network.eval()

# main
with torch.no_grad():
    zs = torch.randn(32, 128).to(device)

    # decode the pcs
    feats_dec = network.sample_decoder(zs)
    pcs_dec = network.decoder(feats_dec)

    # visu
    for j in range(32):
        utils.render_pts(os.path.join(result_dir, 'randomgen-%02d.png'%j), pcs_dec[j].cpu().numpy())

