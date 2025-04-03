import sys
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
import random
import numpy as np
import time
import argparse
import torch
from h5py import File
import logging

from clusterbm.dataset import DatasetAnn
from clusterbm.io import get_checkpoints
from clusterbm.tree import fit, generate_tree
from clusterbm.metrics import l2_dist
from clusterbm.parser import add_args_maketree
from clusterbm.utils import get_device, get_dtype, get_model


def create_parser():
    parser = argparse.ArgumentParser(description='Generates the hierarchical tree of a dataset using the specified RBM model.')
    parser = add_args_maketree(parser)
    
    return parser

def main():
    
    # Define logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s | %(message)s",
        datefmt="%y/%m/%d %H:%M:%S",
        )
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    parser = create_parser()
    args = parser.parse_args()
    if not args.parameters.exists():
        raise FileNotFoundError(args.parameters)
    args.output.mkdir(exist_ok=True)
    
    # Set device and data type
    device = get_device(args.device)
    dtype = get_dtype(args.dtype)

    start = time.time()
    
    # Set random seeds to zero
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
    
    # Load the data and the module
    logger.info('Loading the data')
    dataset = DatasetAnn(
        data_path=args.data,
        ann_path=args.annotations,
        colors_path=args.colors,
        alphabet=args.alphabet,
        device=device,
        dtype=dtype,
    )
    
    # Import the Ebm model and the data
    model = get_model(
        fname=args.parameters,
        device=device,
        dtype=dtype,
    )
    data = dataset.data[:args.n_data]
    leaves_names = dataset.names[:args.n_data]
    labels_dict = [{n : l for n, l in dl.items() if n in leaves_names} for dl in dataset.labels]
    
    # Fit the tree to the data
    alltime = get_checkpoints(args.parameters)
    t_ages = alltime[alltime <= args.max_age]
    logger.info('Fitting the model')
    tree_codes, node_features_dict = fit(
        model=model,
        data=data,
        batch_size=args.batch_size,
        t_ages=t_ages,
        eps=args.eps,
        epsilon=args.epsilon,
        max_iter=args.max_iter,
        order=args.order_mf,
        filter_ages=args.filter,
        )
    max_depth = tree_codes.shape[1]
    # Save the tree codes
    logger.info(f'Saving the model in {args.output}')
    
    # Create the file with the node states
    if args.save_node_features:
        f_nodes = File(args.output / 'node_features.h5', 'w')
        for state_name, state in node_features_dict.items():
            level = int(state_name.split('-')[0].replace('I', ''))
            if level < args.max_depth:
                f_nodes[state_name] = state
        f_nodes.close()
    
    # Generate the tree
    logger.info(f'Generating a tree of depth {min(args.max_depth, max_depth)}. Maximum depth is {max_depth}.')
    if args.colors is not None:
        colors_dict = dataset.colors
    else:
        colors_dict = []
        for ld in labels_dict:
            colors = plt.get_cmap(args.colormap, len(np.unique(list(ld.values()))))
            colors_dict.append({l : to_hex(colors(i)) for i, l in enumerate(np.unique(list(ld.values())))})
    if args.max_depth > max_depth:
        args.max_depth = max_depth 
        
    generate_tree(
        tree_codes=tree_codes,
        leaves_names=leaves_names,
        legend=dataset.legend,
        folder=args.output,
        labels_dict=labels_dict,
        colors_dict=colors_dict,
        depth=args.max_depth,
        node_features_dict=node_features_dict,
        dist_fn=l2_dist
    )

    stop = time.time()
    logger.info(f'Process completed, elapsed time: {round((stop - start) / 60, 1)} minutes')
    sys.exit(0)

if __name__ == '__main__':
    main()
    