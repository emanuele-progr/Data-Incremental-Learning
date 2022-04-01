# Data-incremental Learning

Data-incremental learning is a Python framework for dealing and experimenting with data-incremental scenario.  
This is a particular scenario of incremental learning where, unlike the more famous class incremental learning, the classes are fixed.
The peculiarity of this scenario is that we start with a few examples per class and the examples arrive over time in subsequent tasks and we want to improve the model incrementally using only the new data.

Code started from [here](https://github.com/imirzadeh/stable-continual-learning) and was subsequently modified and strongly expanded for the data-incremental scenario. Also, many class-incremental learning approaches implemented in [FACIL](https://github.com/mmasana/FACIL) have been adapted to the data-incremental scenario.
## Approaches
### Elastic Weight Consolidation
```bash
-- approach ewc ``` [arxiv](https://arxiv.org/abs/1612.00796)

### Learning Without Forgetting
```bash
--approach lwf ```[arxiv](https://arxiv.org/abs/1606.09282)

### iCaRL
```bash
--approach icarl ```[arxiv](https://arxiv.org/abs/1611.07725)| [code](https://github.com/srebuffi/iCaRL)

### Focal distillation
```bash
--approach focal_d ```[arxiv](https://arxiv.org/abs/2011.09161)


## Installation

The code is tested on Python 3.6, PyTorch 1.6.0, and tochvision 0.7.0. In addition, there are some other numerical and visualization libraries that are included in requirements.txt file. However, for convenience, it is provided a script for setup:

```bash
bash setup_and_install.sh
```

## Usage
### arguments
- --dataset : cifar10, cifar100, mnist, imagenet
- --tasks: number of dataset splits
- --epochs-per-task: number of epochs per task
- --lr: learning rate
- --gamma: lr decay rate, value between (0,1)
- --batch-size
- --dropout: dropout regularization. value between [0,1], 0 means no dropout
- --exemplars_per_class: number of exemplar to retain for each class
- --seed: value for dataset random split and reproducibility  
- --net: resnet32, resnet18, resnet50
- --approach: fine_tuning, ewc, lwf, icarl, fd, focal_d, focal_fd
- --compute_joint_incremental[optional]: compute upper bound (joint incremental)
- --grid_search[optional]: starts hyperparameters tuning on task 2 based on "grid_search_config.txt"

### usage examples
- standard experiment on cifar100 with focal distillation approach
```bash
python -m main --dataset cifar100 --tasks 10 --epochs-per-task 50 --lr 0.001 --gamma 1.0 --batch-size 64 --dropout 0.0 --exemplars_per_class 20 --seed 1234 --net resnet18 --approach focal_d

```
- standard experiment on MNIST with feature distillation approach 
```bash
python -m main --dataset mnist --tasks 50 --epochs-per-task 50 --lr 0.001 --gamma 1.0 --batch-size 64 --dropout 0.0 --exemplars_per_class 0 --seed 1234 --net resnet32 --approach fd

```
- experiment on imagenet with focal distillation approach and hyperparameters(lambda, alpha, beta) grid_search on task 2
```bash
python -m main --dataset imagenet --tasks 10 --epochs-per-task 50 --lr 0.001 --gamma 1.0 --batch-size 64 --dropout 0.0 --exemplars_per_class 20 --seed 1234 --net resnet50 --approach focal_d --grid_search

```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.



## License
[MIT](https://choosealicense.com/licenses/mit/)
