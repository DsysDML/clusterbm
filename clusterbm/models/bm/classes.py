from pathlib import Path
from typing import Tuple
from tqdm import tqdm
import torch
from clusterbm.models.classes import Ebm

from clusterbm.models.bm.categorical import (
    _compute_energy as _compute_energy_cat,
    _sample as _sample_cat,
    iterate_mf1 as iterate_mf1_cat,
    iterate_mf2 as iterate_mf2_cat,
    zero_temperature_dynamics as zero_temperature_dynamics_cat,
)

class BmCat(Ebm):
    def __init__(
        self,
        fname: str | Path | None = None,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__(fname=fname, device=device, dtype=dtype)
        if self.params is not None:
            self.num_visibles, self.num_states = self.params["bias"].shape
        
    def compute_energy(
        self,
        X: torch.Tensor,
        params: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Computes the energy of the model for the given visible units.

        Args:
            X (torch.Tensor): Visible units.
            params (dict[str, torch.Tensor] | None, optional): Parameters to be used. If None,
                the parameters of the model are used. Defaults to None.

        Raises:
            ValueError: No parameters can be found.

        Returns:
            torch.Tensor: Energy of the visible units.
        """
        if params is None:
            if self.params is None:
                raise ValueError("No parameters provided.")
            else:
                params = self.params
        
        return _compute_energy_cat(X, params)
    
    def sample(
        self,
        X: torch.Tensor,
        it_mcmc: int,
    ) -> torch.Tensor:
        """Samples from the model using Alternate Gibbs Sampling.

        Args:
            X (torch.Tensor): Initial visible units.
            it_mcmc (int): Number of AGS transitions to perform.

        Raises:
            ValueError: Model parameters not found.

        Returns:
            torch.Tensor: Sampled visible units.
        """
        if self.params is None:
            raise ValueError("Model parameters not found.")
        
        return _sample_cat(
            chains=X,
            params=self.params,
            nsweeps=it_mcmc,
        )
    
    def init_chains(
        self,
        n_gen: int,
    ) -> torch.Tensor:
        """Initializes the chains with random visible units.

        Args:
            n_gen (int): Number of chains to generate.

        Raises:
            ValueError: Number of visible units not found.

        Returns:
            torch.Tensor: Initialized chains.
        """
        if self.num_visibles is None:
            raise ValueError("Number of visible units not found.")
        return torch.randint(
            0, self.num_states, (n_gen, self.num_visibles), device=self.device
        ).to(dtype=self.dtype)
        
        
    def iterate_mean_field(
        self,
        X: Tuple[torch.Tensor,] | torch.Tensor,
        order: int = 2,
        epsilon: float = 1e-4,
        max_iter: int = 2000,
        rho: float = 1.0,
        batch_size: int = 512,
        verbose: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Iterates the mean field self-consistency equations at the specified order, starting from the state X, until convergence.
        
        Args:
            X (Tuple[torch.Tensor,] | torch.Tensor): Initial conditions.
            params (Dict[str, torch.Tensor]): Parameters of the model.
            order (int, optional): Order of the expansion (0, 1, 2). Defaults to 2.
            epsilon (float, optional): Convergence threshold. Defaults to 1e-4.
            max_iter (int, optional): Maximum number of iterations. Defaults to 2000.
            rho (float, optional): Dumping parameter. Defaults to 1.0.
            batch_size (int, optional): Number of samples in each batch. To set based on the memory availability. Defaults to 512.
            verbose (bool, optional): Whether to print the progress bar or not. Defaults to True.
        
        Raises:
            ValueError: If the model parameters are not found.
            NotImplementedError: If the specifiend order of expansion has not been implemented.
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Fixed points of (visible magnetizations, hidden magnetizations)
        """
        
        if self.params is None:
            raise ValueError("Model parameters not found.")
        
        if isinstance(X, tuple):
            X = X[0]
        
        if order == 1:
            mf_function = iterate_mf1_cat
        elif order == 2:
            mf_function = iterate_mf2_cat
        elif order == 0:
            mf_function = zero_temperature_dynamics_cat
        else:
            raise NotImplementedError('Possible choices for the order parameter: (0, 1, 2)')
        
        n_data = X[0].shape[0]
        mv = torch.tensor([], device=self.device)
        num_batches = n_data // batch_size
        num_batches_tail = num_batches
        if n_data % batch_size != 0:
            num_batches_tail += 1
            if verbose:
                pbar = tqdm(total=num_batches + 1, colour='red', ascii="-#")
                pbar.set_description('Iterating Mean Field')
        else:
            if verbose:
                pbar = tqdm(total=num_batches, colour='red', ascii="-#")
                pbar.set_description('Iterating Mean Field')
        for m in range(num_batches):
            X_batch = X[m * batch_size : (m + 1) * batch_size, :]
            mv_batch = mf_function(
                X=X_batch,
                params=self.params,
                epsilon=epsilon,
                rho=rho,
                max_iter=max_iter,
            )
            mv = torch.cat([mv, mv_batch], 0)

            if verbose:
                pbar.update(1)
        # handle the remaining data
        if n_data % batch_size != 0:
            X_batch = X[num_batches * batch_size:, :]
            mv_batch = mf_function(
                X=X_batch,
                params=self.params,
                epsilon=epsilon,
                rho=rho,
                max_iter=max_iter,
            )
            mv = torch.cat([mv, mv_batch], 0)

            if verbose:
                pbar.update(1)
                pbar.close()
                
        return (mv,)
        