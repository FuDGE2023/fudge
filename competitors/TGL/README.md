This repository contains the implementation of TGL (https://github.com/amazon-science/tgl)

Note:
We set some parameter as example. Please consider the parameters presented in the paper if you want to replicate the experiments.

Parameters semantic:
* --data --> dataset name
* --extension --> dataset extension
* --perc_train --> train set percentage
* --validation_set --> True if the computation has to be on the validation set, False for the test set
* --need_regressor --> True if the dataset can have multiple operations per timestamp, False otherwise
* --config --> available configurations: config/TGN.yml, config/TGAT.yml, config/JODIE.yml, config/DySAT.yml
* --factor -> TGL available variants: TGN, TGAT, JODIE, DySAT

Steps to run the experiments:
1) Dataset processing
```
python create_data.py --data dataset_name --extension dataset_extension --perc_train 0.7
python gen_graph.py --data dataset_name+perc_train

Example: 
python create_data.py --data snapshots_regression_bitcoin_alpha_edge_weights --extension .pt --perc_train 0.7
python gen_graph.py --data snapshots_regression_bitcoin_alpha_edge_weights_0.7
```

2) Training
```
python train.py --data dataset_name+perc_train --config path_configuration --gpu 0

Example:
python train.py --data snapshots_regression_bitcoin_alpha_edge_weights --config config/TGN.yml --gpu 0
```

3) Graph generation
```
python generate_graphs.py --data dataset_name+perc_train --config path_configuration --gpu 0 --perc_train 0.7 --need_regressor True --validation_set True
```

4) WL score
```
python compute_wl.py --data dataset_name --extension dataset_extension --perc_train 0.7 --factor TGN --validation_set True
```

5) Graph statistics
```
python compute_statistics.py --data dataset_name --extension dataset_extension --perc_train 0.7 --factor TGN --validation_set True
```
