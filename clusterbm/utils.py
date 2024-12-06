from pathlib import Path
import torch
from adabmDCA.utils import get_device, get_dtype
from clusterbm.models import Ebm
from clusterbm.models import RbmBin, RbmCat, BmCat
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