# Transfer Learning Tremor Location

This repository contains the code used in [van der Laat *et al.* (2023, in prep.)]() to locate the tremor from the Kilauea caldera collapse in 2018.

# Installation

Download a zip file of the code or clone it (HTTP):

    $ git clone https://github.com/lvanderlaat/tl.git

or (SSH):

    $ git clone git@github.com:lvanderlaat/tl.git
    
Go in the repository directory

    $ cd tl

and create a `conda` environment named `tl` and install the package and its dependencies:
    
    $  conda env create -f environment.yml

Activate the environment

    $ conda activate tl

and install this package:

    (tl) $ pip install -e .

# Run the example

This repository contains a minimal working example that you can run to learn how to use this program:

    (tl) $ cd example
    (tl) $ jupyter-lab example.ipynb
