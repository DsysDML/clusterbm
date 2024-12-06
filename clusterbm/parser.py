import argparse
from pathlib import Path
import numpy as np

def add_args_maketree(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser._action_groups.pop()
    required = parser.add_argument_group('Required arguments')
    optional = parser.add_argument_group('Optional arguments')
    
    required.add_argument('-p', '--parameters',       type=Path, help='Path to Ebm model.', required=True)
    required.add_argument('-o', '--output',           type=Path, help='Path to output directory.', required=True)
    required.add_argument('-d', '--data',             type=Path, help='Path to data.', required=True)
    
    optional.add_argument('-a', '--annotations',      type=Path,            default=None,             help='Path to the csv annotation file.',)
    optional.add_argument('-c', '--colors',           type=Path,            default=None,             help='Path to the csv color mapping file.')
    optional.add_argument('-f', '--filter',           type=float,           default=None,             help='(defaults to None). Selects a subset of epochs such that the acceptance rate of swapping two adjacient configurations is the one specified.')
    optional.add_argument("--alphabet",               type=str,             default="protein",        help="(Defaults to protein). Type of encoding for the sequences. Choose among ['protein', 'rna', 'dna'] or a user-defined string of tokens.")
    optional.add_argument('--n_data',                 type=int,             default=512,              help='(Defaults to 512). Number of data to put in the tree.')
    optional.add_argument('--batch_size',             type=int,             default=512,              help='(Defaults to 512). Batch size.')
    optional.add_argument('--max_age',                type=int,             default=np.inf,           help='(Defaults to inf). Maximum age to consider for the tree construction.')
    optional.add_argument('--save_node_features',     action='store_true',  default=False,            help='If specified, saves the states corresponding to the tree nodes.')
    optional.add_argument('--max_iter',               type=int,             default=2000,             help='(Defaults to 2000). Maximum number of TAP iterations.')
    optional.add_argument('--max_depth',              type=int,             default=50,               help='(Defaults to 50). Maximum depth to visualize in the generated tree.')
    optional.add_argument('--order_mf',               type=int,             default=2,                help='(Defaults to 2). Mean-field order of the Plefka expansion.', choices=[1, 2, 3])
    optional.add_argument('--eps',                    type=float,           default=0.1,              help='(Defaults to 0.1). Epsilon parameter of DBSCAN.')
    optional.add_argument('--epsilon',                type=float,           default=1e-4,             help='(Defaults to 1e-4). Convergence threshold of the mean field self consistent equations.')
    optional.add_argument('--colormap',               type=str,             default='tab20',          help='(Defaults to `tab20`). Name of the colormap to use for the labels.')
    optional.add_argument('--device',                 type=str,             default='cuda',           help='(Defaults to `cuda`). Device to use for the computations.')
    optional.add_argument('--dtype',                  type=str,             default='float32',        help='(Defaults to `float32`). Data type to use for the computations.', choices=['float32', 'float64'])
    
    return parser