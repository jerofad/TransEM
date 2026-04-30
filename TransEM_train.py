import faulthandler
faulthandler.enable()
import numpy as np
from BuildGeometry import BuildGeometry
from modellib import  Trainer, TransEM,dotstruct, PETMrDataset
import os
import torch


os.environ["CUDA_VISIBLE_DEVICES"] = "1"

save_training_dir = "/home/jeremiah/Datasets/PicoPET/output/" # training_dataset_dir

g = dotstruct()
g.is3d = False
g.temPath = "/home/jeremiah/Documents/PicoPET/TransEM/system_matrix" #system matrix path
g.radialBinCropFactor = 0.5
g.psf_cm = 0.25
g.niters = 2  # keep low during training — 8×6=48 unrolled RSTR calls OOMs; use more at test time
g.nsubs = 2
g.training_flname = [save_training_dir+os.sep,'data-']
g.save_dir = '/home/jeremiah/Datasets/PicoPET/Model'+os.sep
g.device = 'cpu' #torch.device("cuda:0")
g.num_workers = 0
g.batch_size = 1  # 4 is too large on CPU; increase if running on GPU
g.test_size = 0.2
g.valid_size = 0.1
g.num_train = 100
g.depth =1
g.in_channels = 1 # with or without mrImg
g.lr = 1e-5
g.epochs = 10
g.model_name = 'TransEM'
g.save_from_epoch = 0
g.crop_factor = 0.3
g.do_validation = True


# build PET object
PET = BuildGeometry('mmr',g.radialBinCropFactor)
print('Loading System Matrix...')
PET.loadSystemMatrix(g.temPath,is3d=False )
print('System Matrix loaded.')
# load dataloaders
print('Loading training data...')
train_loader, valid_loader, test_loader = PETMrDataset(g.training_flname, num_train=g.num_train, is3d=g.is3d, \
                                                       batch_size=g.batch_size, test_size=g.test_size, valid_size=g.valid_size, num_workers = g.num_workers)
print('Training data loaded.')
# build model
print('Building model...')
model = TransEM(g.in_channels, g.is3d).to(g.device, dtype=torch.float32)
print('Model built. Starting training...')
# train
Trainer(PET,model, g, train_loader, valid_loader)