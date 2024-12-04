from typing import Any
from pathlib import Path
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from clusterbm.functional import one_hot
from adabmDCA.fasta_utils import import_clean_dataset, encode_sequence, get_tokens

class DatasetAnn(Dataset):
    def __init__(
        self,
        data_path: str | Path,
        ann_path: str | Path | None = None,
        colors_path: str | Path | None = None,
        alphabet: str = "protein",
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ):
        """Initialize the dataset.

        Args:
            data_path (str | Path): Path to the data file (plain text or fasta).
            ann_path (str | Path | None, optional): Path to the annotations file (csv). Defaults to None.
            colors_path (str | Path | None, optional): Path to the color mapping file (csv). If None, colors are assigned automatically. Defaults to None.
            alphabet (str, optional): Selects the type of encoding of the sequences. Default choices are ("protein", "rna", "dna"). Defaults to "protein".
            device (torch.device, optional): Device to store the data. Defaults to torch.device("cpu").
            dtype (torch.dtype, optional): Data type of the data. Defaults to torch.float32.
        """
        self.names = []
        self.data = []
        self.labels = []
        self.colors = []
        self.tokens = None # Only needed for protein sequence data
        
        # Automatically detects if the file is in fasta format and imports the data
        with open(data_path, "r") as f:
            first_line = f.readline()
        if first_line.startswith(">"):
            # Select the proper encoding
            self.tokens = get_tokens(alphabet)
            names, sequences = import_clean_dataset(data_path, self.tokens)
            data = torch.tensor(encode_sequence(sequences, self.tokens), device=device, dtype=torch.int32)
            self.data = one_hot(data, len(self.tokens)).to(dtype)            
            self.names = np.array(names).astype(str)
            
        else:
            with open(data_path, "r") as f:
                for line in f:
                    self.data.append(line.strip().split())
            data = np.array(self.data, dtype=np.float32)
            self.data = torch.tensor(data, device=device, dtype=dtype)
            self.names = np.arange(len(self.data)).astype("str")
        
        # Load annotations
        if ann_path:
            ann_df = pd.read_csv(ann_path).astype(str)
            self.legend = [n for n in ann_df.columns if n != "Name"]

            # Validate the legend format: special characters are not allowed
            special_characters = '!@#$%^&*()-+?=,<>/'
            for leg in self.legend:
                if any(c in special_characters for c in leg):
                    raise KeyError("Legend names can't contain any special characters.")

            for leg in self.legend:
                self.labels.append({str(n) : str(l) for n, l in zip(ann_df["Name"], ann_df[leg])})
        else:
            self.legend = None
            self.labels = None
        
        # Load colors
        if colors_path is not None:
            df_colors = pd.read_csv(colors_path)
            for leg in self.legend:
                df_leg = df_colors.loc[df_colors["Legend"] == leg]
                self.colors.append({str(n) : c for n, c in zip(df_leg["Label"], df_leg["Color"])})

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx : int) -> Any:
        sample = self.data[idx]
        return sample
    
    def get_num_visibles(self) -> int:
        return self.data.shape[1]
    
    def get_num_states(self) -> int:
        return np.max(self.data) + 1