from typing import Dict
from pathlib import Path
import numpy as np
import torch
import h5py

def get_params(
    fname: str | Path,
    index: str | int,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
) -> Dict[str, torch.Tensor]:
    """Returns the parameters of the model at the selected time index.

    Args:
        fname (str | Path): Name of the file containing the model.
        index (str | int): Index identifying the training time at which the model has been saved.
        device (torch.device): device.
        dtype (torch.dtype): dtype.

    Returns:
        Dict[str, torch.Tensor]: Parameters of the model.
    """
    params = {}
    index = str(index)
    with h5py.File(fname, "r") as f:
        for k in f.keys():
            if "update_" in k:
                base_key = "update"
                break
            elif "epoch_" in k:
                base_key = "epoch"
                break
        key = f"{base_key}_{index}"
        for k in f[key].keys():
            if k in ["vbias", "hbias", "weight_matrix", "bias", "coupling_matrix"]:
                params[k] = torch.tensor(f[key][k][()], device=device, dtype=dtype)
        
    return params

def get_checkpoints(fname: str | Path) -> np.ndarray:
    """Returns the list of checkpoints at which the model has been saved.

    Args:
        fname (str | Path): filename of the model.

    Returns:
        np.ndarray: List of checkpoints.
    """
    alltime = []
    with h5py.File(fname, "r") as f:
        for key in f.keys():
            if "update" in key:
                alltime.append(int(key.replace("update_", "")))
            elif "epoch" in key:
                alltime.append(int(key.replace("epoch_", "")))
    # Sort the results
    alltime = np.sort(alltime)
    
    return alltime
