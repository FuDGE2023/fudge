This repository contains the implementation of TGN (https://github.com/twitter-research/tgn)

Note:
We set some parameter as example. Please consider the parameters presented in the paper if you want to replicate the experiments.

Parameters semantic:
* --data --> dataset name
* --extension --> dataset extension
* --perc_train --> train set percentage
* --validation_set --> True if the computation has to be on the validation set, False for the test set
* --need_regressor --> True if the dataset can have multiple operations per timestamp, False otherwise
* --factor -> TGN available factors: graph_attention, graph_sum, identity, time

Steps to run the experiments:

1) Dataset processing
```
python utils/preprocess_data.py --data dataset_name --extension dataset_extension
```

2) Training
```
python train_self_supervised.py -d dataset_name --use_memory --n_runs 1 --n_epoch 1000 --gpu 2 --factor graph_attention --perc_train 0.7 --patience 20
```

3) Graph generation
```
python generate_graphs.py -d dataset_name --use_memory --need_regressor True --factor graph_attention --perc_train 0.7 --validation_set True
```

4) WL score
```
python compute_wl.py --data dataset_name --extension dataset_extension --perc_train 0.7 --factor graph_attention --validation_set True
```

5) Graph statistics
```
python compute_statistics.py --data dataset_name --extension dataset_extension --perc_train 0.7 --factor graph_attention --validation_set True
```
