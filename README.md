# Code Accompanying "Learning Euler Factors of Elliptic Curves" #

This repository contains code and links to data and ML models used in the paper [*Learning Euler Factors of Elliptic Curves*](https://arxiv.org/abs/2502.10357) by Angelica Babei, François Charton, Edgar Costa, Xiaoyu Huang, Kyu-Hwan Lee, David Lowry-Duda, Ashvni Narayanan, and Alexey Pozdnyakov. We apply transformer models and feedforward neural networks to predict Frobenius traces $a_p$ of elliptic curves when given certain other collections of coefficient data.

The preprint is on the [arxiv:2502.10357](https://arxiv.org/abs/2502.10357).

**Contents**

1. [Code Requirements](#code-requirements)
1. [Code Overview](#code-overview)
1. [Data](#data)
1. [Comments](#comments)
1. [License](#license)

## Code Requirements ##

This code was run using [SageMath](https://www.sagemath.org/) version 10.2 with a small number of additional packages:

- [pandas](https://pandas.pydata.org/)
- [pytorch](https://pytorch.org/)
- [requests](https://pypi.org/project/requests/) for simplicity
- [scikit-learn](https://scikit-learn.org/stable/)

In addition, some of the code uses [a particular version of Int2Int](https://github.com/f-charton/Int2Int/tree/7379f2366fbbc30cbe1dc84653ddb87cfd78851c), which is pinned in the source tree.

It would be straightforward to remove SageMath as a dependency. Doing so would require installing the packages

- [matplotlib](https://matplotlib.org/)
- [numpy](https://numpy.org/)
- [sympy](https://www.sympy.org/en/index.html)

and would require writing some small code snippets for prime generation and related utilities.

In different words, the standard scientific python stack, pytorch, and some utility functions are sufficient.

Note that small numbers of GPUs were used to generate the data for the paper, but the machine learning tasks employed ultimately don't require extensive computation. It would be possible to run this on CPUs with some patience.


## Code Overview ##

The code is separated into three parts, according to the relevant section of the paper.
The code in sections 4 and 5 both generate datafiles that are then given to Int2Int.

STUB ABOUT SECTION 4

In [Code/Section 5/5.2/generate_mod2data_no_duplicates.ipynb](/Code/Section%205/5.2/generate_mod2data_no_duplicates.ipynb), there is simple code that creates datafiles for mod $2$ data for Int2Int.

STUB ON HOW TO USE INT2INT ON THIS DATA

The code in [Code/Section 6](https://github.com/ababei/LearningEulerFactors/tree/main/Code/Section%206/) is more involved, as it contains two different neural network implementations and experiments. This code is self-contained. The jupyter notebook for [Section 6.1](https://github.com/ababei/LearningEulerFactors/blob/main/Code/Section%206/6.1/nn_exp_and_saliency.ipynb) is a complete record of an interactive session generating the data for section 6.1 of the paper.


## Data ##

LINK TO ZENODO DATA


## Comments ##

The transformer experiments with Int2Int use a fixed version of Int2Int.
The API to use Int2Int has changed since these experiments were carried out; to replicate the experiments here, make sure to use the pinned version indicated in this repository!

This repository is not accepting contributions. But the authors are going to consider applications of ML to math (and vice versa) further in the future. Feel free to contact us with ideas, suggestions, or other proposals for projects and collaboration.


## License ##

The code here is made available under the MIT License. See the [LICENSE file](/LICENSE) for more.
