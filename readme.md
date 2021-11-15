FOR REVIEWING PURPOSES ONLY. PLEASE DO NOT DISTRIBUTE.
DROPBOX IS USED FOR ANONYMOUS CODE SHARING.
FINAL CODE SUBMISSION WILL BE THROUGH GITHUB.

This folder contains the source code for ICLR2022 submission 3034: 
Topologically Regularized Data Embeddings

## Setup

### Enviroments
* Python (Jupyter notebook) 
* R (Rstudio)

### Python requirements
* TopologyLayer (https://github.com/bruel-gabrielsson/TopologyLayer)
* numpy
* pandas
* pytorch
* matplotlib
* rpy2
* seaborn
* scikit-learn
* networkx
* scipy
* umap-learn
* Dionysus
* diode

### R Requirements 
* TDA
* ggplot2
* latex2exp
* gridExtra
	
## Datasets
* ICLR acronym data: available from "Data" folder
* Synthetic data (two clusters, cycle) are generated in the Jupyter notebook script
* Cell trajectory: https://zenodo.org/record/1443566 (also available from "Data" folder)
* Karate: partially (graph) loaded from networkx and partially (weights) from "Data" folder
* Harry Potter: https://github.com/hzjken/character-network (also available from "Data" folder)

## Run
* Folder "Scripts": contains all code (Python + R) for producing the visualizations prior to the experiments section in the main paper, with files named accordingly..
* Folder "Experiments": contains all code (Python) for producing the results in the experiments
section and Appendix B in the main paper, with files named accordingly..

### Python 
* Open file in Jupyter notebook
* Cell --> Run Cells (ctrl + enter).

### R
* Open file in Rstudio 
* Source --> Source with Echo (ctrl + shift + enter).

## Results
* Folder "Output": contains all (Python Jupyter notebook) output by code block in PDF format, with files named accordingly.