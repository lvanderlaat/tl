yes | conda install nb_conda_kernels

yes | conda create -n tl

eval "$(conda shell.bash hook)"

conda activate tl

yes | conda install -c conda-forge jupyterlab
yes | conda install -c conda-forge jupyter_contrib_nbextensions
yes | conda install -c conda-forge jupyter_nbextensions_configurator
yes | conda install ipykernel
ipython kernel install --user --name=tl 
jupyter nbextension toc2/main

yes | conda install -c conda-forge pyproj
yes | conda install -c anaconda pandas
yes | conda install -c anaconda scikit-learn
yes | conda install -c anaconda pyyaml
yes | conda install -c conda-forge obspy


pip install -e .


