<div align="center">

# Self-Supervised Contrastive Masked Graph Views for Learning Neuron-level Circuit Network

[![python](https://img.shields.io/badge/-Python_3.8_%7C_3.9_%7C_3.10-blue?logo=python&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![pytorch](https://img.shields.io/badge/PyTorch_2.0+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![hydra](https://img.shields.io/badge/Config-Hydra_1.3-89b8cd)](https://hydra.cc/)
[![black](https://img.shields.io/badge/Code%20Style-Black-black.svg?labelColor=gray)](https://black.readthedocs.io/en/stable/)
[![isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/) <br>
[![tests](https://github.com/ashleve/lightning-hydra-template/actions/workflows/test.yml/badge.svg)](https://github.com/ashleve/lightning-hydra-template/actions/workflows/test.yml)
[![license](https://img.shields.io/badge/License-MIT-green.svg?labelColor=gray)](https://github.com/ashleve/lightning-hydra-template#license)
</div>
<br>

## 📌  Introduction
Learning Neuron-level Circuit Network can be used on automatic neuron cell classification and connection prediction, which are fudamental tasks for connectome reconstruction and deciphering brain functions. Traditional approaches to this learning process have relied on extensive neuron typing and labor-intensive proofread. In this paper, we introduce a self-supervised learning method that automates learning neuron-level circuit
network. We leverage graph augmentation methods to generate various contrastive graph views. By distinguishing positive and negative samples within these views, the proposed method captures structural representations of neurons as flexible latent feature input for neuron classification and connection prediction task. To evaluate our method, we built two new Neuron-level Circuit Network datasets, HemiBrain-C and Manc-C from FlyEM project. Experimental Results show that FlyCGV achieves 73.8\%/57.4\% accuracy of neuron classification, and $>$ 0.95 AUC on connection prediction. Our code and data are available at GitHub 

<br>

## Dataset
The Dataset has been uploaded to google drive:[https://drive.google.com/drive/folders/1QNu7A0NREiR5eTowzqODpONfgbiaY3Bl?usp=sharing]

## Main Technologies

[PyTorch](https://pytorch.org/) Well-known Deep learning framework.

[PyG Documentation](https://pytorch-geometric.readthedocs.io/en/latest/)
PyG (PyTorch Geometric) is a library built upon  PyTorch to easily write and train Graph Neural Networks (GNNs) for a wide range of applications related to structured data.
<br>
## 🚀  Quickstart

```bash
# [OPTIONAL] create conda environment
conda create -n myenv python=3.10
conda activate myenv

# install pytorch according to instructions
# https://pytorch.org/get-started/

# install requirements
pip install -r requirements.txt
```








## How It Works

The model is run in the following manner:
```python
from models.model import Model
m = Model("configs/model.yaml")
m.train()
```
You can also run python files in `tests`.


<br>

## Main Config
All PyTorch Lightning modules are dynamically instantiated from module paths specified in config. Example model config:

```yaml
model:
  name: 'GCL'
  learning_rate: 0.001
  hidden_size: 196
  num_layers: 2
  
training:
  epochs: 200
  batch_size: 100
  optimizer: 'adam'
  gamma: 0.1

data:
  train_hemibrain_path: 'data/hemibrain.pt'
```
Location: [configs/model.yaml](configs/model.yaml) <br>
Main project config contains default training configuration.<br>
It determines how config is composed when simply executing command `python tests/link_pred/gcl.py`.<br>




<br>







## Resources

This template was inspired by:



Other useful repositories:

- [https://neuprint.janelia.org](https://neuprint.janelia.org) 

</details>

<br>

## License

Lightning-Hydra-Template is licensed under the MIT License.

```
MIT License

Copyright (c) 2021 ashleve

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

<br>
<br>
<br>
<br>
