import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import argparse
from temp_pred_net import TempPredNet
from executor import Executor
from time import time


# main function
def run(scenario='P0'):
    # loading data
    start = time()
    train_x = np.load(f'data/{scenario}_train_x.npy')
    train_y = np.load(f'data/{scenario}_train_y.npy')
    val_x = np.load(f'data/{scenario}_val_x.npy')
    val_y = np.load(f'data/{scenario}_val_y.npy')
    test_x = np.load(f'data/{scenario}_test_x.npy')
    test_y = np.load(f'data/{scenario}_test_y.npy')
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
    parser.add_argument('--scenario', type=str, default='P0', help='Scenario to train the u-net for. Valid options are P0, P1 and P2. Default: P0')
    scenario = parser.parse_args().scenario
    run(scenario)