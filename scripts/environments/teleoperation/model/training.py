import copy
import math
import argparse
import numpy as np
import pickle
from collections import deque
import torch
import os

from config import *
from helpers import *  # this code only uses the normalize_data function
from models import NCBF, Normalizer  # 现在使用的是PyTorch版本的模型

import pickle
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--init', action='store_true', help='initialize training model')
args = parser.parse_args()

def load_formatted_data(data_filepath):
    """
    (safe_h_obj_xs, safe_h_obj_nxs, safe_ee_xs, safe_ee_nxs, unsafe_h_obj_xs, unsafe_ee_xs)
    """
    print(f"Loading formatted training data from {data_filepath}...")
    if data_filepath.endswith('.npz'):
        data = np.load(data_filepath)
        safe_h_obj_xs = data['safe_local_xs']
        safe_h_obj_nxs = data['safe_local_nxs']
        safe_ee_xs = data['safe_ee_xs']
        safe_ee_nxs = data['safe_ee_nxs']
        unsafe_h_obj_xs = data['unsafe_local_xs']
        unsafe_ee_xs = data['unsafe_ee_xs']
    else:
        with open(data_filepath, 'rb') as f:
            data = pickle.load(f)
        safe_h_obj_xs = data['safe_h_obj_xs']
        safe_h_obj_nxs = data['safe_h_obj_nxs']
        safe_ee_xs = data['safe_ee_xs']
        safe_ee_nxs = data['safe_ee_nxs']
        unsafe_h_obj_xs = data['unsafe_h_obj_xs']
        unsafe_ee_xs = data['unsafe_ee_xs']
    print(f"Loaded training data:")
    print(f"  safe_h_obj_xs shape: {safe_h_obj_xs.shape}")
    print(f"  safe_h_obj_nxs shape: {safe_h_obj_nxs.shape}")
    print(f"  safe_ee_xs shape: {safe_ee_xs.shape}")
    print(f"  safe_ee_nxs shape: {safe_ee_nxs.shape}")
    print(f"  unsafe_h_obj_xs shape: {unsafe_h_obj_xs.shape}")
    print(f"  unsafe_ee_xs shape: {unsafe_ee_xs.shape}")
    return safe_h_obj_xs, safe_h_obj_nxs, safe_ee_xs, safe_ee_nxs, unsafe_h_obj_xs, unsafe_ee_xs
    
if __name__ == "__main__":
    print("Hello World\nNCBF Arm training starts.")
    print("-------------------------------\n")
    
    # specify formatted data file path
    formatted_data_path = "ncbf_trajectories/trajectories_reformat.npz"
    
    # load formatted data
    safe_h_obj_xs, safe_h_obj_nxs, safe_ee_xs, safe_ee_nxs, unsafe_h_obj_xs, unsafe_ee_xs = load_formatted_data(formatted_data_path)
    
    # data is already formatted, so just skip the original data extraction and processing steps
    
    # data normalization (may still need)
    print("\nNormalizing the data...")
    h_obj_inputDim = safe_h_obj_xs.shape[-1]  # 4 (tilt_angle, rel_x, rel_y, rel_z)
    ee_inputDim = safe_ee_xs.shape[-1]  # 4 (end effector pos + vel)
    
    h_obj_normalizer = Normalizer(h_obj_inputDim)
    ee_normalizer = Normalizer(ee_inputDim)
    
    # normalization
    data = normalize_data(safe_h_obj_xs, safe_h_obj_nxs, unsafe_h_obj_xs,
                       safe_ee_xs, safe_ee_nxs, unsafe_ee_xs,
                       h_obj_normalizer, ee_normalizer, update=True)
    
    h_obj_normalizer.save_model(h_obj_normalizer_path)
    ee_normalizer.save_model(ee_normalizer_path)
    print("Normalizing finished.")
    
    # build model - using PyTorch version
    model = NCBF(
        seqInputHorizon=Horizon, 
        seqInputSize=h_obj_inputDim, 
        inputSize=ee_inputDim,
        hiddens=Hiddens, 
        seq_hiddens=SeqHiddens,
        activation="relu",
        regularizer_factor=RegFactor,
        CBF_gamma=Gamma
    )
    
    os.makedirs(os.path.dirname(initial_model_path), exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if args.init:
        print("\n---------Initial phase starts----------")
        
        # using PyTorch version of fit method, no compile
        model.fit(
            data=data, 
            epoch=Epoch, 
            verbose_num=100,
            lr=LR,
            margin_threshold=MarginThreshold,
            derivative_threshold=DerivativeThreshold,
            save_path=os.path.dirname(initial_model_path),
            save_iters=Epoch//100,
            batch_size=256,   
            device=device     
        )
        
        # save final model
        model.save(initial_model_path)
        print(f"\nfinished. Model saved at {initial_model_path}")
        print("-----------------------------------------\n")
    else:
        print(f"\nLoading model from {pretrained_initial_model_path}")
        # using PyTorch's load method
        model = NCBF.load(
            pretrained_initial_model_path,
            hiddens=Hiddens,
            seq_hiddens=SeqHiddens
        )
    
    print("Training completed. Refinement skipped for now.")