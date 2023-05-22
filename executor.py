import torch
from torch import nn
from torch.optim.lr_scheduler import StepLR

import numpy as np
import pandas as pd
import os
import typing
from time import time


class Executor:
    def __init__(self, model: nn.Module, results_dir: str, learning_rate):
        self.model = model.cuda()
        self.learning_rate = learning_rate
        self.batch_size = 64
        self.step_size = 10
        self.gamma = 0.5
        self.epochs = 500
        self.patience = 10

        self.optimizer = torch.optim.Adam(self.model.parameters(), eps=1e-7, lr=self.learning_rate)
        self.lr_scheduler = StepLR(self.optimizer, step_size=self.step_size, gamma=self.gamma)
        self.results_dir = results_dir
        # create results directory if it does not already exist, if it does, print an error
        try:
            os.makedirs(self.results_dir)
        except FileExistsError:
            print(f'ERROR: Results directory already exists: {self.results_dir}, please delete it or choose a different tag')
            exit(1)

    def _save_history(self):
        df = pd.DataFrame(self.history)
        df.to_csv(self.history_path)

    def train(self, train_x, train_y, val_x, val_y):
        batch_count = train_x.shape[0] // self.batch_size
        indexes = np.array(range(1, train_x.shape[0]))
        best_val_loss = None
        best_epoch = None
        os.makedirs(self.results_dir, exist_ok=True)
        self.history_path = os.path.join(self.results_dir, 'history.csv')
        
        bar_length = 30.0
        max_patience = self.patience
        patience = max_patience
        # iterate over epochs
        self.history = {'epoch': [], 'loss': [], 'val_loss': [], 'lr': []}
        for e in range(1, self.epochs + 1):
            # training
            self.model.train()
            start = time()
            batch_losses = []
            np.random.shuffle(indexes)
            # iterate over batches
            for i in range(0, len(indexes), self.batch_size):
                self.optimizer.zero_grad()
                print('', end='\r')
                index_batch = indexes[i: i + self.batch_size]
                batch = torch.tensor(train_x[index_batch]).type('torch.cuda.FloatTensor')
                target_batch = torch.tensor(train_y[index_batch]).type('torch.cuda.FloatTensor')
                out_batch = self.model(batch)
                loss = self.model.loss_function(target_batch, out_batch)
                loss.backward()
                self.optimizer.step()
                batch_losses.append(loss.item())
                progress = int(bar_length * len(batch_losses) / batch_count)
                progress_output = 'Epoch {e:03}: [{progress}>{no_progress}]'.format(e=e, progress='='*progress, no_progress='.'*int(bar_length - progress))
                print(progress_output, end='', sep='', flush=True)
            progress_output = '\rEpoch {e:03}: [{progress}]'.format(e=e, progress='='*int(bar_length + 1))
            print(progress_output, end='', sep='')

            # validation
            with torch.no_grad():
                self.model.eval()
                val_losses = []
                # iterate over batches
                for i in range(1, val_x.shape[0], self.batch_size):
                    val_batch = torch.tensor(val_x[i: i + self.batch_size]).type('torch.cuda.FloatTensor')
                    val_target_batch = torch.tensor(val_y[i: i + self.batch_size]).type('torch.cuda.FloatTensor')
                    val_out_batch = self.model(val_batch)
                    val_loss = self.model.loss_function(val_target_batch, val_out_batch)
                    val_losses.append(val_loss.item())

            # collect epoch data
            mean_loss = np.mean(batch_losses)
            mean_val_loss = np.mean(val_losses)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # update learning rate
            self.lr_scheduler.step()

            # write epoch data to history
            self.history['epoch'].append(e)
            self.history['loss'].append(mean_loss)
            self.history['val_loss'].append(mean_val_loss)
            self.history['lr'].append(current_lr)
            self._save_history()

            # print epoch data
            end = time()
            print(' - duration: {duration}s loss: {loss:.4e} - val_loss: {val_loss:.4e} - lr: {lr:.4e}'.format(duration=int(end - start), loss=mean_loss, val_loss=mean_val_loss, lr=current_lr))

            # save best model and check for early stopping
            if best_val_loss is None or mean_val_loss < best_val_loss:
                patience = max_patience
                best_val_loss = mean_val_loss
                best_epoch = e
                best_cp_path = os.path.join(self.results_dir, 'best.pt')
                torch.save(self.model, best_cp_path)
            else:
                if patience <= 0:
                    break
                patience -= 1
        return mean_loss, mean_val_loss

    def predict(self, test_x, test_y):
        best_cp_path = os.path.join(self.results_dir, 'best.pt')
        print(f'loading weights: {best_cp_path}')
        self.model = torch.load(best_cp_path)
        self.model.eval()
        with torch.no_grad():
            output = np.zeros_like(test_y)
            test_losses = []
            for i in range(0, len(test_x), self.batch_size):
                target_batch = torch.tensor(test_y[i: i + self.batch_size]).type('torch.cuda.FloatTensor')
                batch = torch.tensor(test_x[i: i + self.batch_size]).type('torch.cuda.FloatTensor')
                out_batch = self.model(batch)[:, :, :, :]

                test_losses.append(self.model.loss_function(target_batch, out_batch).cpu().numpy())
                output[i: i + self.batch_size, :, :, :] = np.array(out_batch.cpu())[:, :, :, :]
            return output
