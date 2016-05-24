# README.md
Assignment 4, Language Modeling, Jiayi Lu (jl6583), May-04-2016\
jiayi.lu@nyu.edu

Please follow the instructions below to make sure all scripts will work as expected.

### Download model file manually
```sh
$ mkdir models
$ wget https://www.dropbox.com/s/9eq88jzwnd755pr/best_model.net?dl=0 -O ./models/best_model.net
```

**NOTE** : This URL link of model file may encounter some stability issues, if you crashed into any problem please just email jiayi.lu@nyu.edu. You can also choose to run ``th result.lua`` to download the model file automatically

### Running result.lua
```sh
$ th result.lua [-m PATH_TO_MODEL]
```
**NOTE** : Running result.lua will create a ``./models/`` directory and download the best perplexity model file ``./models/best_model.net``, run this script at first if you want to avoid downloading model file manually.

### Running query_sentences.lua
How to run:
```bash
$ th query_sentences.lua [-m PATH_TO_MODEL] [-g SAMPLING_MODEL]
```
Then just follow the pop-up instructions.

**NOTE** : Please make sure you have the model file named as ``best_model.net`` in ``./models/`` directory if you do not want to provide model path explicitly.

### Running nngraph_warmup.lua
Just ``th`` it!

### About other scripts
* ``main_gru.lua`` works to train the language model according to user configurations (LSTM/GRU, dropout rates and etc.) and save the model to any location given by the user. Implementaion of the network (including LSTM/GRU cell) can be found inside.
* All others lua scripts are dependencies to make sure the scripts mentioned above runs correctly, please **DO NOT** modify or move.