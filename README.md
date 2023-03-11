# `precondition`: Preconditioning Optimizers

[![Unittests](https://github.com/google-research/precondition/actions/workflows/pytest_and_autopublish.yml/badge.svg)](https://github.com/google-research/precondition/actions/workflows/pytest_and_autopublish.yml)
[![PyPI version](https://badge.fury.io/py/precondition.svg)](https://badge.fury.io/py/precondition)

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
