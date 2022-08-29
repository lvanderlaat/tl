# Jupyter (only for running the exmample)
yes | conda install nb_conda_kernels

yes | conda create -n tl

eval "$(conda shell.bash hook)"
conda activate tl

# Jupyter (only for running the exmample)
yes | conda install -c conda-forge jupyterlab
yes | conda install -c conda-forge jupyter_contrib_nbextensions
yes | conda install -c conda-forge jupyter_nbextensions_configurator
yes | conda install ipykernel
ipython kernel install --user --name=tl 
jupyter nbextension toc2/main

# tl dependencies
yes | conda install -c conda-forge pyproj
yes | conda install -c anaconda pandas
yes | conda install -c anaconda scikit-learn
yes | conda install -c anaconda pyyaml
yes | conda install -c conda-forge obspy

# pip packagers
pip install ray
pip install -e .


