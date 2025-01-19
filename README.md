# 2nd Place at the PhD Hackathon workshop in Generative Modeling (Zero-Shot Diffusion Based Image Restoration)


This repository is an implementation of the algorithm we pioneered, which involves hybridizing the **DPS** and **DMPS** algorithms. The **DMPS algorithm** offers speedups of 3x but relies on the uninformative prior assumption, which is unrealistic since most images are not proportional to Gaussian noise (Please refer to the [poster](https://drive.google.com/file/d/1-6qPfdAmrMBOs5XRS69fjnb-xAQhaRt9/view) for more details). We hybridized the DMPS algorithm with DPS by treating them as "exploration vs. exploitation" methods. This resulted in a speedup of the method (1-3x) in addition to enhanced robustness. 
Please refer to our [slides](https://drive.google.com/file/d/1We39vRCalNYvez61BQvYhVi2bsy14YMv/view) and [poster](https://drive.google.com/file/d/1-6qPfdAmrMBOs5XRS69fjnb-xAQhaRt9/view) for the full results and comparisons.

Team: Rajit Rajpal, Marcos Obando, Dolly Chen, Bernardin Tamo Amogou

## Installation

Beforehand, ensure to download the code.
You can use ``git`` or download it as ``zip``.

1. Run the following the command to create a fresh Python environment.

```bash
python3 -m venv venv-hackathon
```

1. Activate the environment

```bash
source venv-hackathon/bin/activate
```

1. then install the project on editable mode

```bash
pip install -e .
```

1. Finally, download [FFHQ model checkpoint](https://drive.google.com/drive/folders/1jElnRoFv7b31fG0v6pTSQkelbSX3xGZh) and put it on ``material/checkpoints`` folder.

Do not forget to put the absolute path of the project in ``py_source/local_paths.py``.
Similarly, put the absolute path to FFHQ checkpoint in ``/py_source/configs/ffhq_model.yaml``


## About the repository structure

The ``material`` folder contains external files such images, and model checkpoints.

Essential functions and classes to load pre-trained Diffusion Models, load images, display them, and initialize inverse problem are located in ``py_source/`` folder.
In particular,
- ``py_source/sampling/`` folder contains examples of algorithm for solving inverse problem
- ``py_source/utils.py`` contains functions to load model, images, and plot them

There are two notebooks to help you get started
- ``demo_inverse_problems.ipynb`` shows how to define an inverse problem, solve it with an algorithm, and visualize the result
- ``demo_evaluation.ipynb`` explains and illustrates the evaluation process of an algorithm


## Note

- To avoid installation conflicts, the code of the following repositories was moved/modified inside ``src`` folder
    - https://github.com/gabrielvc/mcg_diff
    - https://github.com/openai/guided-diffusion
- Link to download FFHQ model checkpoint https://drive.google.com/drive/folders/1jElnRoFv7b31fG0v6pTSQkelbSX3xGZh
- Evaluation script and the inverse problems used will be uploaded later during the week
