# FReSCo
Fast Reciprocal Space Correlator

FResCO is an algorithm to impose reciprocal-space (aka Fourier-space, k-space) constraints on a real space system. For example, starting from a random point pattern, we can adjust the positions of the points to find a point pattern whose structure factor looks like Van Gogh's Starry Night:

![](./images/fresco_diagram.png)

This is achievable in quasilinear time largely due to the FINUFFT package https://finufft.readthedocs.io/en/latest/index.html. We borrow their formalism to describe our real- and reciprocal-spaces as 'uniform' (i.e. a grid of intensity values) and 'non-uniform' (a set of point coordinates in continuous space).
Below, we show our algorithm's handling of all permutations of real- and reciprocal-spaces.
For uniform reciprocal-space examples, the imposed structure factor was $S(k)=0$ for a circle centered at $k=0$.
For non-uniform reciprocal-space examples, peaks were imposed in the structure factor at the locations marked with red circles.
The structure factors depicted below are the measured structure factors of the resulting systems.

![](./images/uniform_nonuniform.png)

We recommend reading our preprint *Fast Generation of Spectrally-Shaped Disorder* at https://arxiv.org/abs/2305.15693 for more in-depth details of our algorithm.

## Installation - Docker

We recommend installing using a Docker container

## Installation - conda

If instead you prefer using a conda environment, follow these steps after downloading FReSCo

Navigate to the FReSCo directory:

`cd /path/to/FReSCo`

Create a new conda environment named `fresco` from `fresco.yml`, which will install all packages using the channel `conda-forge`

`conda env create --name fresco --file=fresco.yml`

(Download and install FINUFFT: https://finufft.readthedocs.io/en/latest/install.html)

Activate your new conda environment. After this step your terminal should lead with the environment name (here we made it `fresco`)

`conda activate fresco`

While in the FReSCo directory, open `setup.py` in a text editor and find the variable `finufft_dir=/path/to/finufft`. Edit it to be the path to your finufft directory.

Save and exit the text editor

Run `setup.py` as follows (the flag --omp is optional for allowing parallelization over multiple cores):

`python setup.py build_ext -i --omp`

(If at any point you would like to rebuild the package from scratch, please run `rm -r build/ cythonize.dat` before running `setup` using the previous line again)

Optional packages for some examples:

`pip install jscatter`

## Usage

Remember to add `/path/to/FReSCo` to your `PYTHONPATH` in `~/.bashrc` and either run `source ~/.bashrc`
 or restart your terminal update with these changes.

Our 'potentials' ('loss functions' if that's more your persuasion) can be imported by name:

`from FReSCo.potentials import UwU, UwNU, NUwU, NUwNU`

Similarly, our provided minimizers can also be imported

`from FReSCo.optimize import LBFGS_CPP, ModifiedFireCPP`

Each of these are classes that need to be instantiated. See examples for demonstrations.

## References and Citing

Our algorithm is described in our preprint

> *Fast Generation of Spectrally-Shaped Disorder*
> 
> Aaron Shih, Mathias Casiulis, and Stefano Martiniani
>
> https://arxiv.org/abs/2305.15693

Our algorithm utlizes the Flatiron Nonuniform Fast Fourier Transform

https://finufft.readthedocs.io/en/latest/refs.html

> A parallel non-uniform fast Fourier transform library based on an “exponential of semicircle” kernel. A. H. Barnett, J. F. Magland, and L. af Klinteberg. SIAM J. Sci. Comput. 41(5), C479-C504 (2019).
>
> https://arxiv.org/abs/1808.06736
> 
> Aliasing error of the exp kernel in the nonuniform fast Fourier transform. A. H. Barnett. Appl. Comput. Harmon. Anal. 51, 1-16 (2021).
>
> https://arxiv.org/abs/2001.09405



