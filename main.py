import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import uuid
import argparse
from temp_pred_net import TempPredNet
from executor import Executor
from time import time


# main function
def run(scenario='P0', repetitions=5, tag=None):
    # use UUID as default tag
    tag = tag or str(uuid.uuid4())
    # loading data
    start = time()
    try:
        train_x = np.load(f'data/{scenario}_train_x.npy')
        train_y = np.load(f'data/{scenario}_train_y.npy')
        val_x = np.load(f'data/{scenario}_val_x.npy')
        val_y = np.load(f'data/{scenario}_val_y.npy')
        test_x = np.load(f'data/{scenario}_test_x.npy')
        test_y = np.load(f'data/{scenario}_test_y.npy')
    except FileNotFoundError as e:
        print('File not found. Please make sure that the data files are present. They can be requested from the authors.')
        sys.exit(1)
    except Exception as e:
        print('Error while loading data: ', e)
        sys.exit(1)

    print('loading data took about {} seconds.'.format(time() - start))

    # run trainings for different seeds
    test_maes = []
    for seed in range(5):
        model = TempPredNet(seed=seed)
        executor = Executor(model, results_dir='results', learning_rate=5.0e-4)
        executor.train(train_x, train_y, val_x, val_y)
        test_z = executor.predict(test_x, test_y)
        test_maes.append(np.mean(np.abs(test_y - test_z)))
        del model
        del executor
    print('test MAE', np.mean(test_maes), '+/-', np.std(test_maes))

# parse command line arguments using argparse
if __name__ == '__main__':
    # check for cuda    
    if not torch.cuda.is_available():
        print('CUDA is not available. Please check your installation.')
        sys.exit(1)
    # check for gpu
    if torch.cuda.device_count() < 1:
        print('No GPU found. Please check your installation.')
        sys.exit(1)
    parser = argparse.ArgumentParser(description='Train a u-net based model for temperature field prediction of experimental RBC data.')
    # add an argument to determine the scenario
    parser.add_argument('--scenario', type=str, default='P0', help='Scenario to train the u-net for. Valid options are P0, P1 and P2. Default: P0')
    # add an argument to determine the number of repetitions
    parser.add_argument('--repetitions', type=int, default=5, help='Number of repetitions for each run. Default: 5')
    # add an argument to set a tag
    parser.add_argument('--tag', type=str, default='', help='Tag to be added to the results directory. Default: ""')
    # parse arguments
    scenario = parser.parse_args().scenario
    repetitions = parser.parse_args().repetitions
    tag = parser.parse_args().tag
    run(scenario, repetitions, tag)
