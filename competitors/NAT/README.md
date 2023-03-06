This repository contains the implementation of NAT (https://arxiv.org/abs/2209.01084)(LoG 2022) by Yuhong Luo, and Pan Li.

Note:
We set some parameter as example. Please consider the parameters presented in the paper if you want to replicate the experiments.

Parameters semantic:
* --data --> dataset name
* --extension --> dataset extension
* --perc_train --> train set percentage
* --validation_set --> True if the computation has to be on the validation set, False for the test set
* --need_regressor --> True if the dataset can have multiple operations per timestamp, False otherwise


Steps to run the experiments:
1) Dataset processing
```
python process.py --data dataset_name --extension dataset_extension
```

2) Training
```
python main.py -d dataset_name --n_degree 8,8  --ngh_dim 2 --run 1 --n_epoch 1000 --perc_train 0.7 --gpu 1
```

3) Graph generation
```
python generate_graphs.py -d dataset_name --n_degree 8,8  --ngh_dim 2 --perc_train 0.7 --validation_set True --need_regressor True
```

4) WL score
```
python compute_wl.py --data dataset_name --extension dataset_extension --perc_train 0.7 --n_degree 8,8 --ngh_dim 2 --validation_set True
```

5) Graph statistics
```
python compute_statistics.py --data dataset_name --extension dataset_extension --perc_train 0.7 --n_degree 8,8 --ngh_dim 2 --validation_set True
```
