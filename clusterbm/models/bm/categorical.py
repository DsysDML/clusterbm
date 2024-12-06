import torch
from typing import Dict
from adabmDCA.statmech import compute_energy as _compute_energy
from adabmDCA.sampling import gibbs_sampling as _sample

@torch.jit.script
def _mf1_residue(
    idx: int,
    mag: torch.Tensor,
    params: Dict[str, torch.Tensor],
) -> torch.Tensor:
    N, L, q = mag.shape
    coupling_residue = params["coupling_matrix"][idx] # (q, L, q)
    bias_residue = params["bias"][idx] # (q,)
    
    mf_term = bias_residue + mag.view(N, L * q) @ coupling_residue.view(q, L * q).T
    mf_residue = torch.softmax(mf_term, dim=1)
    
    return mf_residue

@torch.jit.script
def _mf2_residue(
    idx: int,
    mag: torch.Tensor,
    params: Dict[str, torch.Tensor],
) -> torch.Tensor:
    N, L, q = mag.shape
    coupling_residue = params["coupling_matrix"][idx] # (q, L, q)
    bias_residue = params["bias"][idx] # (q,)
    mag_i = mag[:, idx] # (n, q)
    
    mf_term = bias_residue + mag.view(N, L * q) @ coupling_residue.view(q, L * q).T
    reaction_term_temp = (
        0.5 * coupling_residue.view(1, q, L, q) + # (1, q, L, q)
        (torch.tensordot(mag_i, coupling_residue, dims=[[1], [0]]) * mag).sum(dim=2).view(N, 1, L, 1) - # nd,djc,njc->nj
        0.5 * torch.einsum("njc,ajc->naj", mag, coupling_residue).view(N, q, L, 1) -                    # njc,ajc->naj
        torch.tensordot(mag_i, coupling_residue, dims=[[1], [0]]).view(N, 1, L, q)                      # nd,djb->njb
    )
    reaction_term = (
        (reaction_term_temp * coupling_residue.view(1, q, L, q)) * mag.view(N, 1, L, q)
    ).sum(dim=3).sum(dim=2) # najb,ajb,njb->na
    tap_residue = torch.softmax(mf_term + reaction_term, dim=1)
    
    return tap_residue


def _sweep_mf1(
    residue_idxs: torch.Tensor,
    mag: torch.Tensor,
    params: Dict[str, torch.Tensor],    
) -> torch.Tensor:
    """Updates the magnetizations using the naive mean field equations.

    Args:
        residue_idxs (torch.Tensor): List of residue indices in random order.
        mag (torch.Tensor): Magnetizations of the residues.
        params (Dict[str, torch.Tensor]): Parameters of the model.

    Returns:
        torch.Tensor: Updated magnetizations.
    """
    for idx in residue_idxs:
        mag[:, idx] = _mf1_residue(idx, mag, params)  
    
    return mag


def _sweep_mf2(
    residue_idxs: torch.Tensor,
    mag: torch.Tensor,
    params: Dict[str, torch.Tensor],    
) -> torch.Tensor:
    """Updates the magnetizations using the TAP equations.

    Args:
        residue_idxs (torch.Tensor): List of residue indices in random order.
        mag (torch.Tensor): Magnetizations of the residues.
        params (Dict[str, torch.Tensor]): Parameters of the model.

    Returns:
        torch.Tensor: Updated magnetizations.
    """
    for idx in residue_idxs:
        mag[:, idx] = _mf2_residue(idx, mag, params)  
    
    return mag


def iterate_mf1(
    X: torch.Tensor,
    params: Dict[str, torch.Tensor],
    epsilon: float = 1e-4,
    max_iter: int = 500,
    rho: float = 1.0,
):
    """Iterates the TAP self consistent equations until convergence.

    Args:
        X (torch.Tensor): Initial magnetizations.
        params (Dict[str, torch.Tensor]): Parameters of the model.
        epsilon (float, optional): Convergence threshold. Defaults to 1e-6.
        max_iter (int, optional): Maximum number of iterations. Defaults to 2000.
        rho (float, optional): Damping parameter. Defaults to 1.0.

    Returns:
        torch.Tensor: Fixed point magnetizations of the mean field self consistency equations.
    """
    X_ = X.clone()
    iterations = 0
    while True:
        X_old = X_.clone()
        X_new = _sweep_mf1(torch.randperm(X_.shape[1]),X_, params)
        X_ = rho * X_old + (1. - rho) * X_new
        diff = torch.abs(X_old - X_).max()
        iterations += 1
        if diff < epsilon or iterations > max_iter:
            break
    
    return X_

def iterate_mf2(
    X: torch.Tensor,
    params: Dict[str, torch.Tensor],
    epsilon: float = 1e-4,
    max_iter: int = 500,
    rho: float = 1.0,
):
    """Iterates the TAP self consistent equations until convergence.

    Args:
        X (torch.Tensor): Initial magnetizations.
        params (Dict[str, torch.Tensor]): Parameters of the model.
        epsilon (float, optional): Convergence threshold. Defaults to 1e-6.
        max_iter (int, optional): Maximum number of iterations. Defaults to 2000.
        rho (float, optional): Damping parameter. Defaults to 1.0.

    Returns:
        torch.Tensor: Fixed point magnetizations of the mean field self consistency equations.
    """
    X_ = X.clone()
    iterations = 0
    while True:
        X_old = X_.clone()
        X_new = _sweep_mf2(torch.randperm(X_.shape[1]),X_, params)
        X_ = rho * X_old + (1. - rho) * X_new
        diff = torch.abs(X_old - X_).max()
        iterations += 1
        if diff < epsilon or iterations > max_iter:
            break
    
    return X_