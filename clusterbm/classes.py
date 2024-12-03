from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterable, Dict, Tuple
import numpy as np
from tqdm import tqdm
import torch

from clusterbm.io import get_epochs, get_params

class Ebm(ABC):
    """Abstract class for energy-based models."""
    
    def __init__(
        self,
        fname: str | Path | None = None,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.fname = fname
        self.epochs = get_epochs(fname)
        self.params = None
        self.num_states = None
        self.num_visibles = None
        self.device = device
        self.dtype = dtype
        
    @abstractmethod
    def load(
        self,
        fname: str | Path,
        index: int | None = None,
    ) -> None:
        pass
    
    @abstractmethod
    def compute_energy(
        self,
        X: torch.Tensor,
        params: Dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        pass
    
    @abstractmethod
    def sample(
        self,
        X: torch.Tensor,
        it_mcmc: int,
    ) -> torch.Tensor:
        pass
    
    @abstractmethod
    def get_profile_hiddens(
        self,
        X: torch.Tensor,
    ) -> torch.Tensor:
        pass
    
    @abstractmethod
    def init_chains(
        self,
        n_gen: int,
    ) -> torch.Tensor:
        pass
    
    @abstractmethod
    def iterate_mean_field(
        self,
        X: torch.Tensor | Tuple[torch.Tensor, torch.Tensor],
        order: int,
        epsilon: float,
        max_iter: int,
        rho: float,
    ) -> torch.Tensor:
        pass
    
    def init_sampling(
        self,
        n_gen: int,
        it_mcmc: int,
        epochs: Iterable[int] | None = None,
    ) -> torch.Tensor:
        """Initializes a stack of len(epochs) sets of chains by performing it_mcmc steps of MCMC sampling for each model starting from the initial state of the first model.

        Args:
            n_gen (int): Number of chains for each model.
            it_mcmc (int): Number of Monte Carlo steps to perform for each model in the stack.
            epochs (Iterable[int] | None, optional): List of epochs to use. If None, all epochs present in the model's file are used. Defaults to None.

        Returns:
            torch.Tensor: Stack of n_gen chains.
        """
        
        if epochs is None:
            epochs = self.epochs
        n_models = len(epochs)

        chains_0 = self.init_chains(n_gen)
        chains = torch.zeros(size=(n_models, *chains_0.shape), device=self.device, dtype=self.dtype)
        chains[0] = chains_0
        
        # initialize all models by performing it_mcmc steps starting from the state of the previous model
        pbar = tqdm(total=(len(epochs) - 1) * it_mcmc, colour='red', leave=False, dynamic_ncols=True, ascii='-#')
        for idx, ep in enumerate(epochs[1:]):
            self.load(self.fname, index=ep)
            chains[idx + 1] = self.sample(X=chains[idx], it_mcmc=it_mcmc)
            pbar.update(it_mcmc)
            
        return chains
    
    def match_epochs(
        self,
        chains: torch.Tensor,
        epoch_ref: int,
        target_acc_rate: float = 0.25,
    ) -> int:
        n_chains = len(chains[0])
        idx_ref = np.where(self.epochs == epoch_ref)[0][0]
        params_ref = get_params(filename=self.fname, stamp=epoch_ref, device=self.device)
        chains_ref = chains[idx_ref]
        
        for i in range(idx_ref):
            idx_test = idx_ref - i - 1
            params_test = get_params(filename=self.fname, stamp=self.epochs[idx_test], device=self.device)
            chains_test = chains[idx_test]
            delta_E = (
                - self.compute_energy(chains_ref, params_test)
                + self.compute_energy(chains_test, params_test)
                + self.compute_energy(chains_ref, params_ref)
                - self.compute_energy(chains_test, params_ref)
            )
            swap_chain = torch.bernoulli(torch.clamp(torch.exp(delta_E), max=1.0)).bool()
            acc_rate = (swap_chain.sum() / n_chains).cpu().numpy()
            if (acc_rate < target_acc_rate + 0.1) or (self.epochs[idx_test] == self.epochs[0]):
                print(f"Checkpoint match: {self.epochs[idx_ref]}\t->\t{self.epochs[idx_test]}\t-\tacc_rate = {acc_rate:.3f}")
                
                return self.epochs[idx_test]
            
    def filter_epochs(
        self,
        chains: torch.Tensor,
        target_acc_rate: float,
    ) -> np.ndarray:
        epoch_ref = self.epochs[-1]
        sel_epochs = [epoch_ref]
        while epoch_ref > self.epochs[0]:
            epoch_ref = self.match_epochs(chains=chains, epoch_ref=epoch_ref, target_acc_rate=target_acc_rate)
            if epoch_ref is None:
                epoch_ref = self.epochs[0]
            sel_epochs.append(epoch_ref)
        sel_epochs = np.sort(sel_epochs)
        
        return sel_epochs