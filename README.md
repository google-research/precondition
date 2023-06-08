# `precondition`: Preconditioning Optimizers

[![Unittests](https://github.com/google-research/precondition/actions/workflows/pytest_and_autopublish.yml/badge.svg)](https://github.com/google-research/precondition/actions/workflows/pytest_and_autopublish.yml)
[![PyPI version](https://badge.fury.io/py/precondition-opt.svg)](https://badge.fury.io/py/precondition-opt)

Installation (note package name is `precondition` but pypi distribution name is `precondition-opt`):
```
pip3 install -U precondition-opt
```

Currently, this contains several preconditioning optimizer implementations. Please refer to the citations below.

Shampoo (`distributed_shampoo.py`)

```
@article{anil2020scalable,
  title={Scalable second order optimization for deep learning},
  author={Anil, Rohan and Gupta, Vineet and Koren, Tomer and Regan, Kevin and Singer, Yoram},
  journal={arXiv preprint arXiv:2002.09018},
  year={2020}
}
```

Sketchy (`distributed_shampoo.py`), logical reference implementation as a branch in Shampoo.
```
@article{feinberg2023sketchy,
  title={Sketchy: Memory-efficient Adaptive Regularization with Frequent Directions},
  author={Feinberg, Vladimir and Chen, Xinyi and Sun, Y Jennifer and Anil, Rohan and Hazan, Elad},
  journal={arXiv preprint arXiv:2302.03764},
  year={2023}
}
```
In Appendix A of the aforementioned paper, S-Adagrad is tested along with other optimization algorithms (including RFD-SON, Adagrad, OGD, Ada-FD, FD-SON) on three benchmark datasets (`a9a`, `cifar10`, `gisette_scale`). To recreate the result in Appendix A, first download the benchmark datasets (available at https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/) to your a local folder (e.g. `~/data`). For each `DATASET = {'a9a', 'cifar10', 'gisette'}`, run the following two commands. The first command will run a sweep over all the hyperparameter sets, and the second command will plot a graph of the best set of hyperparameters from the previous run. Consistent with the fair allocation of hyperparameter training budget, we tune delta and the learning rate, each over 7 points uniformly spaced from 1e-6 to 1 in logarithmic scale, for FD-SON and Ada-FD. For the rest of the optimization algorithms, where delta is taken to be 0, we tune the learning rate over 49 points uniformly spaced from 1e-6 to 1 in logarithmic scale. The commands are provided as following:

For running sweep over different sets of hyperparameters, run: (... hides different values of the hyperparameters to sweep over)
```
python3 oco/sweep.py --data_dir='~/data' --dataset=DATASET --save_dir='/tmp/results' \
--lr=1 ... -lr=1e-6 --alg=ADA, --alg=OGD, --alg=S_ADA, --alg=RFD_SON
```
and
```
python3 oco/sweep.py --data_dir='~/data' --dataset=DATASET --save_dir='/tmp/results' --lr=1 ... --lr=1e-6 \
--delta=1 --delta=1e-6 --alg=ADA_FD, --alg=FD_SON
```
After running the above commands, the results will be saved in the folders named with the time stamps at execution (e.g. '/tmp/results/YYYY-MM-DD' and '/tmp/results/yyyy-mm-dd').

To plot the best set of hyperparameters from the previous run, run:
```
python3 oco/sweep.py --data_dir='~/data' --dataset=DATASET --save_dir='/tmp/results' \
--use_best_from='/tmp/results/YYYY-MM-DD' --use_best_from='/tmp/results/yyyy-mm-dd'
```
For detailed documentations on the supported flags, run:
```
python3 oco/sweep.py --help
```

SM3 (`sm3.py`).
```
@article{anil2020scalable,
  title={Scalable second order optimization for deep learning},
  author={Anil, Rohan and Gupta, Vineet and Koren, Tomer and Regan, Kevin and Singer, Yoram},
  journal={arXiv preprint arXiv:2002.09018},
  year={2020}
}
```

This external repository was seeded from existing open-source work available at [this google-research repository](https://github.com/google-research/google-research/tree/master/scalable_shampoo).


*This is not an officially supported Google product.*
