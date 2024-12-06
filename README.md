# clusterbm
[![PRE](https://img.shields.io/badge/PhysRevE-108,014110-red.svg)](https://scholar.google.com/citations?view_op=view_citation&hl=it&user=__OKD-kAAAAJ&sortby=pubdate&citation_for_view=__OKD-kAAAAJ:EPG8bYD4jVwC)

Code for the paper "Unsupervised hierarchical clustering using the learning dynamics of RBMs" by Aurélien Decelle, Lorenzo Rosset and Beatriz Seoane.

`clusterbm` generates a hierarchical tree of some input dataset using a previously trained BM or RBM model.

<p align="center">
<image src="/images/tree-MNIST.png" width=456 height=400/>
<p align="center">
<em>Example of hierarchical clustering obtained on the MNIST dataset. Image taken from the paper.</em>
</p>
</p>

## Installation

To install the package do
```
pip install -e .
```

## Usage

The code for training an RBM and saving the model in the correct format can be found in [TorchRBM](https://github.com/AurelienDecelle/TorchRBM.git). Instead, to train a BM model and obtain the appropriate output file use [adabmDCA](https://github.com/spqb/adabmDCApy.git) with the training option `--checkpoints acceptance`.

Once you have a properly trained RBM model, to generate the tree enter:
```bash
clusterbm -p <path_to_model> -d <path_to_data> -a <path_to_annotations> -o <output_folder>
```
This will create a folder called `output_folder` containing the tree in newick format and the annotation files to be given to [iTOL](https://itol.embl.de/).

The list of available arguments can be printed using:
```bash
clusterbm -h
```

### Mandatory arguments

- `--parameters, -p`: Path to the model's parameters;
- `--output, -o`: Name of the repository where to save the output. If not already existent, it will be created;
- `--data, -d`: Path to the data file (fasta for categorical variables and plain text for binary variables (0, 1));

### Optional arguments

- `--annotations, -a`: Path to the `csv` file containing the data annotations. See [Annotation Format](#annotations-format) for the details;
- `--colors, -c`: Path to the `csv` file containing the label-color mapping. See [Color mapping format](#color-mapping-format) for the details;
- `--n_data`: Number of data to include in the tree. By default, the program takes the first 500 data encountered in the data file.
- `--save_node_features`: If this flag is specified, all the fixed points corresponding to the tree nodes will be saved into a `.h5` file;
- `--max_depth`: Maximum depth of the returned tree. The algorithms will use all the ages allowed by `max_age` regardless of this parameter. By default, the full tree is returned;
- `--batch_size`: Defaults to 500. Size of the batches, to be changed based on memory constraints;
- `--max_age`: Maximum age (in terms of epochs) to be considered for the tree construction. All the older models will not be used. By default it uses all the ages present in the model file;
- `--order_mf`: Order of the mean field approximation to be used for estimating the model's free energy, where '1' corresponds to the naive mean field and '2' corresponds to the TAP approximation. Binary variables support (1, 2, 3), while for categorical variables only (1, 2) are implemented. Defaults to '2';
- `--colormap`: If `colors` is omitted, the colors in the tree are assigned automatically according the Matplotlib's colomap specified here. Defaults to "tab20";
- `--max_iter`: Maximum number of iterations of the mean-field equations. Defaults to 10000;
- `--alphabet`: When the input data come from a file in fasta format, an alphabet for the symbols encoding is needed. You can choose among the default alphabets "protein", "dna", "rna", or a coherent alphabet of your choice. Defaults to "protein";
- `--epsilon`: Convergence threshold for the mean-field equations. Defaults to 1e-4;
- `--eps`: Epsilon parameter of the DBSCAN algorithm.

## Data format
The input data can be either binary or categorical.

### Binary data
Binary data are accepted in any text file format, where each line represents a single data vector.

**Note**: Since data points do not have an identifier when using this format, the row index of the annotation appearing in the annotation file must match the row index of the points appearing in the data file.

### Categorical data
Categorical data must be given in `fasta` format. The `fasta` format consists of alternating lines of headers (identifiers for the data points) and data vector. The headers are preceeded by the token '>'.

**Example**:

```
>sequence_1
-KLVIQAAPYGPEWLPGDADDLPLTA
>sequence_2
-KKIILEARVNEYAPRTSNPNIPYTA
```

## Annotations format
Annotations must be contained in a `csv` file. The file must have a mandatory column called "Name" and one or more columns with arbitrary names. We refer to these columns as "Legends". Each row must contain a name for the annotated data and one category for each Legend. If there is more than one Legend but you don't have annotations for all of them for a given data, just put one of {'-1', 'None', 'nan', 'uncategorized'} where the information is missing. Data that do not have any annotation should not be included in this file.

When data come from a plain text file, meaning that they have no name associated, an integer number will be used for the name and it will refer to the data point of the data file at the same position of the annotation in the list.

This is an example of an annotation file:

| Name | Legend 1 | Legend 2 |
|------|----------|----------|
|name_1| label_1  | sublabel_1 |
|name_2| label_2  | -1         |
|name_3| label_3  | sublabel_3 |

**NOTE**: Legend names can't contain any special characters.

## Color mapping format
For custom coloring, a `csv` file that associates each label with a color must be provided. The file must have three columns with the mandatory names: "Legend", "Label" and "Color". Each row must have the specification of the Legend from which the label comes (it has to correspond to one of the annotation file's column names), the label and the color in hex format.

An example of color specification is the following:

| Legend | Label | Color |
|------|----------|----------|
|Legend 1| label_1  | #a6cee3 |
|Legend 1| label_2  | #1f78b4 |
|Legend 2| sublabel_3  | #b2df8a |
