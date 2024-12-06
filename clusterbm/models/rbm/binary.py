import torch
from typing import Tuple, Dict

@torch.jit.script
def _profile_hiddens(
    v: torch.Tensor,
    hbias: torch.Tensor,
    weight_matrix: torch.Tensor,
) -> torch.Tensor:

    mh = torch.sigmoid(hbias + v @ weight_matrix)
    return mh

@torch.jit.script
def _compute_energy(
    v: torch.Tensor,
    params: Dict[str, torch.Tensor],
) -> torch.Tensor:

    field = v @ params["vbias"]
    exponent = params["hbias"] + (v @ params["weight_matrix"])
    log_term = torch.where(exponent < 10, torch.log(1. + torch.exp(exponent)), exponent)
    return - field - log_term.sum(1)

def _sample_hiddens(
    v: torch.Tensor,
    hbias: torch.Tensor,
    weight_matrix: torch.Tensor,
) -> torch.Tensor:

    h = torch.bernoulli(torch.sigmoid(hbias + v @ weight_matrix))
    return h

def _sample_visibles(
    h: torch.Tensor,
    vbias: torch.Tensor,
    weight_matrix: torch.Tensor,
) -> torch.Tensor:
    
    v = torch.bernoulli(torch.sigmoid(vbias + h @ weight_matrix.T))
    return v

@torch.jit.script
def _sample(
    X: torch.Tensor,
    params: Dict[str, torch.Tensor],
    it_mcmc: int,
) -> torch.Tensor:

    # Unpacking the arguments
    v = X
    for _ in torch.arange(it_mcmc):
        h = _sample_hiddens(v=v, hbias=params["hbias"], weight_matrix=params["weight_matrix"])
        v = _sample_visibles(h=h, vbias=params["vbias"], weight_matrix=params["weight_matrix"])
        
    return v

@torch.jit.script
def iterate_mf1(
    X: Tuple[torch.Tensor, torch.Tensor],
    params: Dict[str, torch.Tensor],
    epsilon: float = 1e-4,
    max_iter: int = 2000,
    rho: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Iterates the mean field self-consistency equations at first order (naive mean field), starting from the state X, until convergence.
    
    Args:
        X (Tuple[torch.Tensor, torch.Tensor]): Initial conditions (visible and hidden magnetizations).
        params (Dict[str, torch.Tensor]): Parameters of the model.
        epsilon (float, optional): Convergence threshold. Defaults to 1e-4.
        max_iter (int, optional): Maximum number of iterations. Defaults to 2000.
        rho (float, optional): Dumping parameter. Defaults to 0.0.
        
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Fixed points of the mean field self consistency equations.
    """
    
    mv, mh = X
    iterations = 0
    while True:
        mv_prev = torch.clone(mv)
        mh_prev = torch.clone(mh)
        field_h = params["hbias"] + (mv @ params["weight_matrix"])
        mh = rho * mh_prev + (1. - rho) * torch.sigmoid(field_h)
        field_v = params["vbias"] + (mh @ params["weight_matrix"].mT)
        mv = rho * mv_prev + (1. - rho) * torch.sigmoid(field_v)
        eps1 = torch.abs(mv - mv_prev).max()
        eps2 = torch.abs(mh - mh_prev).max()
        if (eps1 < epsilon) and (eps2 < epsilon):
            break
        iterations += 1
        if iterations >= max_iter:
            break
        
    return (mv, mh)

@torch.jit.script
def iterate_mf2(
    X: Tuple[torch.Tensor, torch.Tensor],
    params: Dict[str, torch.Tensor],
    epsilon: float = 1e-4,
    max_iter: int = 2000,
    rho: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Iterates the mean field self-consistency equations at second order (TAP equations), starting from the state X, until convergence.
    
    Args:
        X (Tuple[torch.Tensor, torch.Tensor]): Initial conditions (visible and hidden magnetizations).
        params (Dict[str, torch.Tensor]): Parameters of the model.
        epsilon (float, optional): Convergence threshold. Defaults to 1e-4.
        max_iter (int, optional): Maximum number of iterations. Defaults to 2000.
        rho (float, optional): Dumping parameter. Defaults to 0.0.
        
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Fixed points of the mean field self consistency equations.
    """
    mv, mh = X
    weight_matrix2 = torch.square(params["weight_matrix"])
    iterations = 0

    while True:
        mv_prev = torch.clone(mv)
        mh_prev = torch.clone(mh)
        
        dmv = mv - torch.square(mv)
        
        field_h = params["hbias"] \
            + (mv @ params["weight_matrix"]) \
            + (0.5 - mh) * (dmv @ weight_matrix2)
        mh = rho * mh_prev + (1. - rho) * torch.sigmoid(field_h)
        
        dmh = mh - torch.square(mh)
        field_v = params["vbias"] \
            + (mh @ params["weight_matrix"].mT) \
            + (0.5 - mv) * (dmh @ weight_matrix2.mT)
        mv = rho * mv_prev + (1. - rho) * torch.sigmoid(field_v)
        eps1 = torch.abs(mv - mv_prev).max()
        eps2 = torch.abs(mh - mh_prev).max()
        if (eps1 < epsilon) and (eps2 < epsilon):
            break
        iterations += 1
        if iterations >= max_iter:
            break
        
    return (mv, mh)

@torch.jit.script
def iterate_mf3(
    X: Tuple[torch.Tensor, torch.Tensor],
    params: Dict[str, torch.Tensor],
    epsilon: float = 1e-4,
    max_iter: int = 2000,
    rho: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Iterates the mean field self-consistency equations at third order, starting from the state X, until convergence.
    
    Args:
        X (Tuple[torch.Tensor, torch.Tensor]): Initial conditions (visible and hidden magnetizations).
        params (Dict[str, torch.Tensor]): Parameters of the model.
        epsilon (float, optional): Convergence threshold. Defaults to 1e-4.
        max_iter (int, optional): Maximum number of iterations. Defaults to 2000.
        rho (float, optional): Dumping parameter. Defaults to 0.0.
        
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Fixed points of the mean field self consistency equations.
    """
    mv, mh = X

    weight_matrix2 = torch.square(params["weight_matrix"])
    iterations = 0
    
    weight_matrix2 = torch.pow(params["weight_matrix"], 2)
    weight_matrix3 = torch.pow(params["weight_matrix"], 3)
    while True:
        mv_prev = torch.clone(mv)
        mh_prev = torch.clone(mh)
        
        dmv = mv - torch.square(mv)
        dmh = mh - torch.square(mh)
        
        field_h = params["hbias"] \
            + (mv @ params["weight_matrix"]) \
            + (0.5 - mh) * (dmv @ weight_matrix2) \
            + (1/3 - 2 * dmh) * ((dmv * (0.5 - mv)) @ weight_matrix3)
        mh = rho * mh_prev + (1. - rho) * torch.sigmoid(field_h)
        
        dmh = mh - torch.square(mh)
        field_v = params["vbias"] \
            + (mh @ params["weight_matrix"].mT) \
            + (0.5 - mv) * (dmh @ weight_matrix2.mT) \
            + (1/3 - 2 * dmv) * ((dmh * (0.5 - mh)) @ weight_matrix3.mT)
        mv = rho * mv_prev + (1. - rho) * torch.sigmoid(field_v)
        eps1 = torch.abs(mv - mv_prev).max()
        eps2 = torch.abs(mh - mh_prev).max()
        if (eps1 < epsilon) and (eps2 < epsilon):
            break
        iterations += 1
        if iterations >= max_iter:
            break
        
    return (mv, mh)