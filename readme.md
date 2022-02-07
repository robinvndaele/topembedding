This folder contains the code for the paper _'Topologically Regularized Data Embeddings'_, presented at The 10th International Conference on Learning Representations (ICLR), 2022.

arxiv submission: https://arxiv.org/abs/2110.09193.

## Setup

### Enviroments
* Python (Jupyter notebook) 
* R (Rstudio)

### Python channels
* pytorch
* defaults
* conda-forge

### Python requirements
* python=3.9.7
* cudatoolkit=11.1.1
* pytorch=1.9.0
* TopologyLayer (https://github.com/bruel-gabrielsson/TopologyLayer)
* numpy=1.21.2
* matplotlib=3.4.3
* scipy=1.7.1
* pandas
* rpy2
* seaborn
* scikit-learn
* networkx
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