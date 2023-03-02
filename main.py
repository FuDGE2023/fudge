import json
import os
import sys
import torch

from utils.train_and_inference import traininig_generation_phase, get_results


def main(fname):
    with open(fname) as fp:
        params = json.load(fp)

    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
        os.environ['CUDA_VISIBLE_DEVICES'] = params['n_gpu']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    params['device'] = device

    dataset = params['dataset']
    space_wl_results = params['space_wl_results']
    space_plot_losses = params['space_plot_losses']
    folder_generation_graphs = params['folder_generation_graphs']
    space_results_losses = params['space_results_losses']
    space_results_generation_dataset = os.path.join(params['folder_generation_graphs'], dataset)
    checkpoint = params['checkpoint']
    checkpoint_dataset = os.path.join(checkpoint, dataset)

    params['checkpoint'] = checkpoint_dataset
    params['folder_generation_graphs'] = space_results_generation_dataset

    for dir in [space_wl_results, space_plot_losses, space_results_losses, folder_generation_graphs,
                space_results_generation_dataset, checkpoint, checkpoint_dataset]:
        if not os.path.exists(dir):
            os.mkdir(dir)

    if params['train'] or params['generate_test'] or params['generate_validation']:
        traininig_generation_phase(params)
    get_results(params)


if __name__ == '__main__':
    main(sys.argv[1])
