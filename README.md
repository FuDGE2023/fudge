## FuDGE: Modeling Full Dynamic Graph Evolution
This is the accompanying code for the paper **FuDGE: Modeling Full Dynamic Graph Evolution**.

The repository contains all code to re-execute our model and competitors presented in the paper.
You need Pytorch >= 1.12 to run it.

The repository contains the following folders:

- **env**, with the library requirements to create the python environment;
- **config**, it contains experiment configurations for each dataset;
- **utils**, with utility code;
- **model**, with the source code of the model presented in the paper;

main.py contains the code to run the experiments presented in the paper. You have to pass the experiment configuration.

```
python3 main.py ./config/{config_name}.json
```
where config_name is the name of the experimental configuration file.

main.py module imports the needed libraries and set the environment variable to activate the GPU device.

- create_data = 1 &rarr; to create the dataset
- train = 1 &rarr; to train the model
- generate_test = 1 &rarr; to generate test graphs
- generate_val = 1 &rarr; to generate val graphs
- compute_wl_test = 1 &rarr; to compute wl-similarity for test
- compute_wl_val = 1 &rarr; to compute wl-similarity for validation

## Dataset Reference

- [Bitcoin-Alpha](https://snap.stanford.edu/data/soc-sign-bitcoin-alpha.html)
- [Bitcoin-OTC](https://snap.stanford.edu/data/soc-sign-bitcoin-otc.html)
- [UCI-Forum](http://konect.cc/networks/opsahl-ucforum/)
- [EU-Core](https://snap.stanford.edu/data/email-Eu-core-temporal.html)

All datasets must be in a folder named Data.