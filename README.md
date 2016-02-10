# DS-GA 1008: Deep Learning
#

## Assignment 1 - MNIST Digit Recognition
### Introduction

Digit Recognition using MNIST dataset

### Structure
```
.
├── dat
│   └── mnist.t7 ```dataset```
├── ds-ga-1008-a1-master ```tutorial starter code```
└── src ```source code location```
    ├── results_weightDecay=0 ```subdirectories for results```
    ├── results_weightDecay=0.1
    └── results_weightDecay=1
```

### Usage
 * ```python src/train.py --param1 val1 val2 val3 --param2 val1 val2 val3 val4 --param3 ....```, train models for different cmd args
 * ```python src/predict.py```, generate predictions using model.net files in each result subdirectory

### Note
 * Please add dat to .gitignore and store large data files in dat directory


