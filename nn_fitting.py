#!/usr/bin/env python
# coding: utf-8

# In[1]:


import ase.io as aseio
import pickle
from rascal.neighbourlist.structure_manager import mask_center_atoms_by_id
import numpy as np
from rascal.utils import get_optimal_radial_basis_hypers
from rascal.representations import SphericalInvariants as SOAP
import itertools
import torch
from torch import nn
import random
from tqdm import tqdm
from matplotlib import pyplot as plt
import argparse


# In[2]:


torch.manual_seed(0)
random.seed(0)
np.random.seed(0)


# In[ ]:


parser = argparse.ArgumentParser(description='Script for fitting NN on training configurations and part of the manifold')
parser.add_argument('num_use_from_manifold', metavar='num_use_from_manifold', type=int,
                    help='number of configurations from manifold to use for training')
parser.add_argument('replicate_manifold', metavar='replicate_manifold', type=int,
                    help='number of manifold replications to balance classes')
parser.add_argument('n_neurons', metavar='n_neurons', type=int,
                    help='width of the neural network, i.e., number of neurons in all intermediate layers')
parser.add_argument('output_file', metavar='output_file', type=str,
                    help='filename to save the results, should ends with .npz')
parser.add_argument('input_path', metavar='input_path', type=str,
                    help='path to folder with the input xyz data')
args = parser.parse_args()

if not args.output_file.endswith('npz'):
    raise ValueError("output file should ends with .npz")


# In[4]:

N_EPOCH = 20000
BATCH_SIZE = 4096
DEVICE = 'cuda'
LR_SCHEDULER_PATIENCE = 1000


# In[5]:


BOHR2A = 0.529177

def normalized_cross_product(a,b):
    # | x  y  z  |
    # | a0 a1 a2 |
    # | b0 b1 b2 |
    # = x(a1b2-a2b1) - y(a0b2-a2b0) + z(a0b1-a1b0)
    nhat = np.array([a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]])
    norm = np.dot(nhat,nhat)
    return nhat/np.sqrt(norm)

def torsion_angle(rxyz): # rxyz is an np.array of shape (nat,3) where the first atom is C
   perms = [[1,2,3],[1,2,4],[2,3,4],[1,3,4]]
   d = 0.
   for it1, perm in enumerate(perms):
        A = rxyz[perm[0]]
        B = rxyz[perm[1]]
        C = rxyz[perm[2]]
        nhat = normalized_cross_product(B-A, C-A)
        for it2 in range(it1+1, len(perms)):
            A_prime = rxyz[perms[it2][0]]
            B_prime = rxyz[perms[it2][1]]
            C_prime = rxyz[perms[it2][2]]
            nhat_prime = normalized_cross_product(B_prime-A_prime, C_prime-A_prime)
            d += np.dot(nhat, nhat_prime)**2
   return d
bohr2a = 0.529177
def fourbody(struc, conversion=1.0/bohr2a):    
    return torsion_angle(struc.positions*conversion)

def fix_structures(frames, conversion=bohr2a):
    for i, f in enumerate(frames):            
        mask_center_atoms_by_id(f, [0])
        f.positions *= conversion
        f.info["fourbody"] = fourbody(f)
        f.cell = [100,100,100]
        f.positions += 50
        f.wrap(eps=1e-12)
    return frames


# In[6]:


frames_train = fix_structures(aseio.read(f'{args.input_path}/train_without_traj.xyz', ':'))
frames_val = fix_structures(aseio.read(f'{args.input_path}/val.xyz', ':'))

with open(f'{args.input_path}/quasiconstant-manifolds.pickle', 'rb') as f:
    all_manifolds = pickle.load(f)
frames_manifold = all_manifolds[(1.0, "s0.2-n8-l8")]

for struc in frames_manifold:
    mask_center_atoms_by_id(struc, [0])
    

# In[7]:


Hsoap = {
    'soap_type': 'PowerSpectrum',
    'interaction_cutoff': 2.5,
    'max_radial': 8,
    'max_angular': 8,
    'gaussian_sigma_constant': 0.2,
    'gaussian_sigma_type': 'Constant',
    'cutoff_smooth_width': 0.0,
    'radial_basis': 'GTO',
    'normalize' : False
}

# computes an optimal radial basis for the expansion
Hsoap = get_optimal_radial_basis_hypers(Hsoap, frames_train[::10], expanded_max_radial=20)

# this is just to avoid getting C coefficients, given it's always the same. 
nnl = np.array(list(itertools.product(range(Hsoap['max_radial']), range(Hsoap['max_radial']), 
                                      range(Hsoap['max_angular']+1))), dtype=int)
Hsoap['coefficient_subselection']= {       
        "a" : np.ones(Hsoap['max_radial']**2*(Hsoap['max_angular']+1), dtype=int).tolist(),
        "b" : np.ones(Hsoap['max_radial']**2*(Hsoap['max_angular']+1), dtype=int).tolist(),
        "n1": nnl[:,0].tolist(),
        "n2": nnl[:,1].tolist(),
        "l":  nnl[:,2].tolist()
    }


# In[8]:


soap = SOAP(**Hsoap)
manifold_ps = soap.transform(frames_manifold).get_features(soap)
train_ps = soap.transform(frames_train).get_features(soap)
val_ps = soap.transform(frames_val).get_features(soap)

train_ps = torch.FloatTensor(train_ps).to(DEVICE)
val_ps = torch.FloatTensor(val_ps).to(DEVICE)
manifold_ps = torch.FloatTensor(manifold_ps).to(DEVICE)


perm = torch.randperm(len(frames_manifold)).to(DEVICE)
train_manifold_indices = perm[:args.num_use_from_manifold]
val_manifold_indices = perm[args.num_use_from_manifold:]

frames_train_manifold = [frames_manifold[i] for i in train_manifold_indices]
frames_val_manifold = [frames_manifold[i] for i in val_manifold_indices]
train_manifold_ps = manifold_ps[train_manifold_indices]
val_manifold_ps = manifold_ps[val_manifold_indices]

train_manifold_indices_repeated = train_manifold_indices.repeat(args.replicate_manifold)

frames_full_train = frames_train + [frames_manifold[i] for i in train_manifold_indices_repeated]
full_train_ps = torch.cat([train_ps, manifold_ps[train_manifold_indices_repeated]], dim = 0)

# normalize powerspectrum
ps_std = torch.sqrt((full_train_ps**2).sum(axis=1).mean(axis=0)) 
full_train_ps/=ps_std
train_manifold_ps/=ps_std
val_manifold_ps/=ps_std
val_ps/=ps_std
train_ps/=ps_std
manifold_ps /=ps_std

perm = torch.randperm(len(frames_full_train))
frames_full_train = [frames_full_train[i] for i in perm]
full_train_ps = full_train_ps[perm]

energies_train = torch.FloatTensor([frame.info['fourbody'] for frame in frames_train]).to(DEVICE)
energies_val = torch.FloatTensor([frame.info['fourbody'] for frame in frames_val]).to(DEVICE)
energies_train_manifold = torch.FloatTensor([frame.info['fourbody'] for frame in frames_train_manifold]).to(DEVICE)
energies_val_manifold = torch.FloatTensor([frame.info['fourbody'] for frame in frames_val_manifold]).to(DEVICE)
energies_full_train = torch.FloatTensor([frame.info['fourbody'] for frame in frames_full_train]).to(DEVICE)

mean = torch.mean(energies_train)

energies_train -= mean
energies_val -= mean
energies_train_manifold -= mean
energies_val_manifold -= mean
energies_full_train -= mean


# In[13]:


train = {'ps' : train_ps, 'energies' : energies_train}
val = {'ps' : val_ps, 'energies' : energies_val}
train_manifold = {'ps' : train_manifold_ps, 'energies' : energies_train_manifold}
val_manifold = {'ps' : val_manifold_ps, 'energies' : energies_val_manifold}
full_train = {'ps' : full_train_ps, 'energies' : energies_full_train}


# In[14]:


def iterate_minibatches(data, batch_size, shuffle = True):
    key = list(data.keys())[0]
    num_total = data[key].shape[0]
    if shuffle:
        permutation = torch.randperm(num_total).to(DEVICE)
        
    for index in range(0, num_total, batch_size):
        if shuffle:
            indices = permutation[index : index + batch_size]
        current = {}
        for key in data.keys():
            if shuffle:
                current[key] = data[key][indices]
            else:
                current[key] = data[key][index : index + batch_size]
        yield current


# In[15]:

model = nn.Sequential(nn.GroupNorm(4, full_train_ps.shape[1]),
                                        nn.Linear(full_train_ps.shape[1], args.n_neurons), nn.GroupNorm(2 , args.n_neurons), nn.Tanh(),
                                        nn.Linear(args.n_neurons, args.n_neurons), nn.GroupNorm(2, args.n_neurons), nn.Tanh(),
                                        nn.Linear(args.n_neurons, 1)).to(DEVICE)

# In[16]:


def get_loss(predictions, targets):
    delta = predictions - targets
    return torch.mean(delta * delta)

def get_mae(first, second):
    return np.mean(np.abs(first - second))

def get_rmse(first, second):
    delta = first - second
    return np.sqrt(np.mean(delta * delta))


# In[17]:


def get_stats(predictions, targets):
    mae = get_mae(predictions, targets)
    rmse = get_rmse(predictions, targets)
    return mae, rmse

def get_batched_predictions(model, data):
    all_predictions = []
    for batch in iterate_minibatches(data, batch_size = BATCH_SIZE, shuffle = False):
        predictions = model(batch['ps'])[:, 0]
        all_predictions.append(predictions.data.cpu().numpy())
    return np.concatenate(all_predictions, axis = 0)

optim = torch.optim.Adam(model.parameters())
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, factor = 0.5, patience = LR_SCHEDULER_PATIENCE)


best_val_rmse = None
best_train_rmse = None
pbar = tqdm(range(N_EPOCH))
for epoch_num in pbar:
    train_predictions = []
    train_targets = [] # with the same random permutation as used in training
    model.train(True)
    for batch in iterate_minibatches(full_train, batch_size = BATCH_SIZE):
        predictions = model(batch['ps'])[:, 0]
        
        train_predictions.append(predictions.data.cpu().numpy()) 
        train_targets.append(batch['energies'].data.cpu().numpy())
        loss = get_loss(predictions, batch['energies'])
        loss.backward()
        optim.step()
        optim.zero_grad()
    train_predictions = np.concatenate(train_predictions, axis = 0)
    train_targets = np.concatenate(train_targets, axis = 0)
    train_mae, train_rmse = get_stats(train_predictions, train_targets)
    
    model.train(False)
    val_predictions = get_batched_predictions(model, val)
    val_mae, val_rmse = get_stats(val_predictions, val['energies'].data.cpu().numpy()) 
    '''val_manifold_predictions = get_batched_predictions(model, val_manifold)
    val_manifold_mae, val_manifold_rmse = get_stats(val_manifold_predictions, 
                                                    val_manifold['energies'].data.cpu().numpy())'''
    
    pbar.set_description(f"val rmse: {val_rmse}; lr: {optim.param_groups[0]['lr']}")
    '''if epoch_num % 10 == 0:
        print("full train mae: ", train_mae)
        print("full train rmse: ", train_rmse)
        print("val mae: ", val_mae)
        print("val rmse: ", val_rmse)
        print("val manifold mae: ", val_manifold_mae)
        print("val manifold rmse: ", val_manifold_rmse)
        print("learning rate now", optim.param_groups[0]['lr'])'''
        
    
    lr_scheduler.step(train_rmse)
   


# In[ ]:


model.train(False)
train_predictions = get_batched_predictions(model, train)
val_predictions = get_batched_predictions(model, val)
if train_manifold['ps'].shape[0] > 0:
    train_manifold_predictions = get_batched_predictions(model, train_manifold)
if val_manifold['ps'].shape[0] > 0:
    val_manifold_predictions = get_batched_predictions(model, val_manifold)

output = {'train_predictions' : train_predictions,
          'val_predictions' : val_predictions,
          'train_ground_truth' : train['energies'].data.cpu().numpy(),
          'val_ground_truth' : val['energies'].data.cpu().numpy()}

if train_manifold['ps'].shape[0] > 0:
    output['train_manifold_predictions']  = train_manifold_predictions
    output['train_manifold_ground_truth'] = train_manifold['energies'].data.cpu().numpy()
    
if val_manifold['ps'].shape[0] > 0:
    output['val_manifold_predictions'] = val_manifold_predictions
    output['val_manifold_ground_truth'] = val_manifold['energies'].data.cpu().numpy()

np.savez(args.output_file, **output)




