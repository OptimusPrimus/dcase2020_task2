import os
import pickle
import copy
import experiments
import torch


def load_model(
        experiment_id,
        log_dir=os.path.join('..', 'experiment_logs'),
        checkpoint_num=199
):
    path = os.path.join(log_dir, experiment_id)
    config_path = os.path.join(path, 'conf.py')
    with open(config_path, 'rb') as config_dictionary_file:
        config_dictionary = pickle.load(config_dictionary_file)
        configuration_dict = copy.deepcopy(config_dictionary)

    model = experiments.BaselineExperiment(configuration_dict)
    model_path = os.path.join(path, '_ckpt_epoch_{}.ckpt'.format(checkpoint_num))
    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint['state_dict'])
    return model.cuda().eval()
