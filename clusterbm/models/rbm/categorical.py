import torch
from typing import Tuple, Dict
from clusterbm.functional import one_hot

@torch.jit.script
def _profile_hiddens(
    v: torch.Tensor,
    hbias: torch.Tensor,
    weight_matrix: torch.Tensor,
) -> torch.Tensor:
    
    num_visibles, num_states, num_hiddens = weight_matrix.shape
    weight_matrix_oh = weight_matrix.reshape(num_visibles * num_states, num_hiddens)
    v_oh = v.reshape(-1, num_visibles * num_states)
    mh = torch.sigmoid(hbias + v_oh @ weight_matrix_oh)
    
    return mh

@torch.jit.script
def _compute_energy(
    v: torch.Tensor,
    params: Dict[str, torch.Tensor],
) -> torch.Tensor:

    vbias, params["hbias"], weight_matrix = params
    num_visibles, num_states, num_hiddens = weight_matrix.shape
    v_oh = v.reshape(-1, num_visibles * num_states)
    vbias_oh = vbias.flatten()
    weight_matrix_oh = weight_matrix.reshape(num_visibles * num_states, num_hiddens)
    field = v_oh @ vbias_oh
    exponent = params["hbias"] + (v_oh @ weight_matrix_oh)
    log_term = torch.where(exponent < 10, torch.log(1. + torch.exp(exponent)), exponent)
    
    return - field - log_term.sum(1)

def _sample_hiddens(
    v: torch.Tensor,
    hbias: torch.Tensor,
    weight_matrix: torch.Tensor,
) -> torch.Tensor:

    num_visibles, num_states, num_hiddens = weight_matrix.shape
    weight_matrix_oh = weight_matrix.reshape(num_visibles * num_states, num_hiddens)
    v_oh = v.reshape(-1, num_visibles * num_states)
    h = torch.bernoulli(torch.sigmoid(hbias + v_oh @ weight_matrix_oh))
    
    return h

def _sample_visibles(
    h: torch.Tensor,
    vbias: torch.Tensor,
    weight_matrix: torch.Tensor,
) -> torch.Tensor:
    
    num_visibles, num_states, _ = weight_matrix.shape
    mv = torch.softmax(vbias + torch.tensordot(h, weight_matrix, dims=[[1], [2]]), dim=-1)
    v = one_hot(
        torch.multinomial(mv.reshape(-1, num_states), 1).reshape(-1, num_visibles),
        num_classes=num_states,
        ).to(dtype=mv.dtype)
    
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
            field_h = params["hbias"] + torch.tensordot(mv, params["weight_matrix"], dims=[[1, 2], [1, 0]])
            mh = rho * mh_prev + (1. - rho) * torch.sigmoid(field_h)
            field_v = params["vbias"] + torch.tensordot(mh, params["weight_matrix"], dims=[[1], [2]])
            mv = rho * mv_prev + (1. - rho) * torch.softmax(field_v.transpose(1, 2), 2) # (Ns, num_states, Nv)
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
        fW = torch.einsum('svq, vqh -> svh', mv, params["weight_matrix"])
        
        field_h = params["hbias"] \
            + torch.einsum('svq, vqh -> sh', mv, params["weight_matrix"]) \
            + (mh - 0.5) * (
                torch.einsum('svq, vqh -> sh', mv, weight_matrix2) \
                - torch.square(fW).sum(1))
    
        mh = rho * mh_prev + (1. - rho) * torch.sigmoid(field_h)
        Var_h = mh - torch.square(mh)
        fWm = torch.multiply(Var_h.unsqueeze(1), fW) # svh
        field_v = params["vbias"] \
            + torch.einsum('sh, vqh -> svq', mh, params["weight_matrix"]) \
            + (0.5 * torch.einsum('sh, vqh -> svq', Var_h, params["weight_matrix"]) \
            - torch.einsum('svh, vqh -> svq', fWm, params["weight_matrix"]))
        mv = rho * mv_prev + (1. - rho) * torch.softmax(field_v, 2) # (Ns, num_states, Nv)
        eps1 = torch.abs(mv - mv_prev).max()
        eps2 = torch.abs(mh - mh_prev).max()
        if (eps1 < epsilon) and (eps2 < epsilon):
            break
        iterations += 1
        if iterations >= max_iter:
            break
        
    return (mv, mh)