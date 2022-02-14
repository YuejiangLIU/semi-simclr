## Semi-supervised Contrastive Learning for TTT++

This is an implementation of semi-supervised contrastive learning for [improved test-time training](https://github.com/vita-epfl/ttt-plus-plus). 

> Our code allows for jointly training an image classification model on the main task and an auxiliary self-supervised (SimCLR) task. 
> Given such pre-trained models, one can later adapt the model at test time based on the self-supervised task to counter the effect of common image corruptions and natural domain shifts.
> The current version supports experiments on the ResNet as the base model and three datasets, including CIFAR-10, CIFAR-100 and VisDA.

Below are exemplary commands for CIFAR-10/100. For the visual domain adaptation dataset, replacing `cifar` with `visda` should do the job.

### Setup

Specify folder name for datasets, e.g.,
```bash
export DATADIR=/data/cifar
```

Specify folder name for checkpoints, e.g., 
```bash
mkdir save && export SAVEIDR=save
```

### Training

1. Train the model on the main classification task:
```bash
bash scripts/main_cifar.sh
```

2. Fine-tune the model on the main and SSL tasks jointly:
```bash
bash scripts/joint_cifar.sh
```

### Citation

If you find this code useful for your research, please cite our paper:

```bibtex
@inproceedings{liu2021ttt++,
  title={TTT++: When Does Self-Supervised Test-Time Training Fail or Thrive?},
  author={Liu, Yuejiang and Kothari, Parth and van Delft, Bastien Germain and Bellot-Gurlet, Baptiste and Mordan, Taylor and Alahi, Alexandre},
  booktitle={Thirty-Fifth Conference on Neural Information Processing Systems},
  year={2021}
}
```

### Acknowledgements

Our code is built upon [PyContrast](https://github.com/HobbitLong/PyContrast/tree/master/pycontrast).
