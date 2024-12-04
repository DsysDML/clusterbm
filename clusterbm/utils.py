from pathlib import Path
import torch
from clusterbm.models import Ebm
from clusterbm.models.bm import BmCat
from clusterbm.models.rbm import RbmBin, RbmCat
from clusterbm.io import get_checkpoints, get_params

def get_model(
    fname: str | Path,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
) -> Ebm:
    """Returns the proper Ebm model class for importing the model stored in the file.
    The model is also initialized with the last saved parameters.

    Args:
        fname (str | Path): File containing the model.
        device (torch.device, optional): Device. Defaults to torch.device("cpu").
        dtype (torch.dtype, optional): Data type. Defaults to torch.float32.
        
    Returns:
        Ebm: Initialized Energy-based model.
    """
    checkpoints = get_checkpoints(fname)
    params = get_params(
        fname=fname,
        index=checkpoints[-1],
    )
    if "coupling_matrix" in params.keys():
        return BmCat(
            fname=fname,
            device=device,
            dtype=dtype,
        )
    elif "weight_matrix" in params.keys():
        match params["weight_matrix"].dim():
            case 2:
                return RbmBin(
                    fname=fname,
                    device=device,
                    dtype=dtype,
                )
            case 3:
                return RbmCat(
                    fname=fname,
                    device=device,
                    dtype=dtype,
                )
                
def get_device(device: str) -> torch.device:
    """Returns the device where to store the tensors.
    
    Args:
        device (str): Device to be used.
        
    Returns:
        torch.device: Device.
    """
    if "cuda" in device and torch.cuda.is_available():
        device = torch.device(device)
        print(f"Running on {torch.cuda.get_device_name(device)}")
        return device
    else:
        print("Running on CPU")
        return torch.device("cpu")
    
    
def get_dtype(dtype: str) -> torch.dtype:
    """Returns the data type of the tensors.
    
    Args:
        dtype (str): Data type.
        
    Returns:
        torch.dtype: Data type.
    """
    if dtype == "float32":
        return torch.float32
    elif dtype == "float64":
        return torch.float64
    else:
        raise ValueError(f"Data type {dtype} not supported.")