import argparse
import logging
import os
import sys
import time
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.autograd import Variable
from tqdm import tqdm
from utils import *

import dnn_utils
import net
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.data import DataConfig, TabularDataset


parser = argparse.ArgumentParser()
parser.add_argument('--seed', default=0, help="Seed")
parser.add_argument('--cont_res', default=4, help="Continuous_resolution")
parser.add_argument('--model_dir', default='./base_model/',
                    help="Directory containing params.json")
parser.add_argument('--data', default='Heart', help="dataset name")
parser.add_argument('--restore_file', default=None,
                    help="Optional, name of the file in --model_dir \
                    containing weights to reload before training")  # 'best' or 'train'


def _unpack_batch(batch):
    if isinstance(batch, (tuple, list)) and len(batch) == 2:
        return batch[0], batch[1]
    if isinstance(batch, dict):
        # try common keys
        for xk in ("x", "X", "features", "inputs"):
            if xk in batch:
                x = batch[xk]
                break
        else:
            raise KeyError("Could not find features key in batch dict.")
        for yk in ("y", "Y", "labels", "targets"):
            if yk in batch:
                y = batch[yk]
                break
        else:
            raise KeyError("Could not find labels key in batch dict.")
        return x, y
    raise TypeError(f"Unsupported batch type: {type(batch)}")


def evaluate(model, loss_fn, dataloader, metrics, params):
    model.eval()
    summ = []

    device = torch.device("cuda") if getattr(params, "cuda", False) and torch.cuda.is_available() \
             else torch.device("cpu")
    model.to(device)

    with torch.no_grad():
        for batch in dataloader:
            data_batch, labels_batch = _unpack_batch(batch)
            if params.dataset.startswith("mnist"):
                # image size
                if params.dataset == "mnist":
                    H = W = 28
                elif params.dataset == "mnist20x20":
                    H = W = 20
                else:
                    raise ValueError(f"Unknown MNIST variant: {params.dataset}")
                
                # x is [B, H*W] → [B, 1, H, W]
                if data_batch.dim() == 2:
                    data_batch = data_batch.view(data_batch.size(0), 1, H, W)

            data_batch = data_batch.to(device, non_blocking=True).float()
            labels_batch = labels_batch.to(device, non_blocking=True).long()

            # forward
            output_batch = model(data_batch)
            loss = loss_fn(output_batch, labels_batch)

            output_np = output_batch.detach().cpu().numpy()
            labels_np = labels_batch.detach().cpu().numpy()

            summary_batch = {m: metrics[m](output_np, labels_np) for m in metrics}
            summary_batch["loss"] = float(loss.detach().cpu().item())
            summ.append(summary_batch)

    # mean metrics
    if len(summ) == 0:
        logging.warning("- Eval: no batches found in dataloader.")
        return {}

    metrics_mean = {k: np.mean([x[k] for x in summ]) for k in summ[0].keys()}
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
    logging.info("- Eval metrics : " + metrics_string)
    return metrics_mean



def train(model, optimizer, loss_fn, dataloader, metrics, params):
    model.train()
    summ = []
    loss_avg = dnn_utils.RunningAverage()

    device = torch.device("cuda") if getattr(params, "cuda", False) and torch.cuda.is_available() \
             else torch.device("cpu")
    model.to(device)

    with tqdm(total=len(dataloader)) as t:
        for i, batch in enumerate(dataloader):
            train_batch, labels_batch = _unpack_batch(batch)

            if params.dataset.startswith("mnist"):
                # image size
                if params.dataset == "mnist":
                    H = W = 28
                elif params.dataset == "mnist20x20":
                    H = W = 20
                else:
                    raise ValueError(f"Unknown MNIST variant: {params.dataset}")
                
                # x is [B, H*W] → [B, 1, H, W]
                if train_batch.dim() == 2:
                    train_batch = train_batch.view(train_batch.size(0), 1, H, W)

            # move to device
            train_batch = train_batch.to(device, non_blocking=True).float()
            labels_batch = labels_batch.to(device, non_blocking=True).long()

            # forward
            output_batch = model(train_batch)         # logits: [B, num_classes]
            loss = loss_fn(output_batch, labels_batch)

            # backward
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            # summaries
            if i % params.save_summary_steps == 0:
                output_np = output_batch.detach().cpu().numpy()
                labels_np = labels_batch.detach().cpu().numpy()

                summary_batch = {m: metrics[m](output_np, labels_np) for m in metrics}
                summary_batch["loss"] = float(loss.detach().cpu().item())
                summ.append(summary_batch)

            # running loss
            loss_avg.update(float(loss.detach().cpu().item()))
            t.set_postfix(loss='{:05.3f}'.format(loss_avg()))
            t.update()

    if len(summ) > 0:
        metrics_mean = {k: np.mean([x[k] for x in summ]) for k in summ[0].keys()}
        metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
        logging.info("- Train metrics: " + metrics_string)
    else:
        logging.info(f"- Train loss: {loss_avg():05.3f} (no metric summaries)")


def train_and_evaluate(model, data, train_dataloader, val_dataloader, optimizer,
                       loss_fn, metrics, params, model_dir, restore_file=None):
    """
    Train the model and evaluate every epoch.

    Args:
        model: (torch.nn.Module) the neural network
        params: (Params) hyperparameters
        model_dir: (string) directory containing config, weights and log
        restore_file: (string) - name of file to restore from (without its extension .pth.tar)
    """
    # reload weights from restore_file if specified
    if restore_file is not None:
        restore_path = os.path.join(args.model_dir, args.data, f"seed_{str(params.seed)}",
                                    f"cont_res{str(params.continuous_resolution)}", args.restore_file + '.pth.tar')
        logging.info("Restoring parameters from {}".format(restore_path))
        dnn_utils.load_checkpoint(restore_path, model, optimizer)

    best_val_acc = 0.0

    scheduler = StepLR(optimizer, step_size=100, gamma=0.1)
    
    for epoch in range(params.num_epochs):     
        scheduler.step()
        logging.info("Epoch {}/{}".format(epoch + 1, params.num_epochs))
        train(model, optimizer, loss_fn, train_dataloader, metrics, params)
        
        val_metrics = evaluate(model, loss_fn, val_dataloader, metrics, params)        
        val_acc = val_metrics['accuracy']
        is_best = val_acc>=best_val_acc

        # save weights
        dnn_utils.save_checkpoint({'epoch': epoch + 1,
                                   'state_dict': model.state_dict(),
                                   'optim_dict' : optimizer.state_dict()},
                                  is_best=is_best,
                                  checkpoint=os.path.join(model_dir, data, f"seed_{str(params.seed)}",
                                                          f"cont_res{str(params.continuous_resolution)}"))

        if is_best:
            logging.info("- Found new best accuracy")
            best_val_acc = val_acc

            best_json_path = os.path.join(model_dir, data, f"seed_{str(params.seed)}", f"cont_res{str(params.continuous_resolution)}",
                                          "metrics_val_best_weights.json")
            dnn_utils.save_dict_to_json(val_metrics, best_json_path)

        last_json_path = os.path.join(model_dir, data, f"seed_{str(params.seed)}", f"cont_res{str(params.continuous_resolution)}",
                                      "metrics_val_last_weights.json")
        dnn_utils.save_dict_to_json(val_metrics, last_json_path)



if __name__ == '__main__':

    # load params
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, args.data, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = dnn_utils.Params(json_path)
    params.seed = int(args.seed)
    params.continuous_resolution = int(args.cont_res)

    # GPU
    params.cuda = torch.cuda.is_available()

    # set seed
    random.seed(params.seed)
    torch.manual_seed(params.seed)
    if params.cuda: torch.cuda.manual_seed(params.seed)
    
    dnn_utils.set_logger(os.path.join(args.model_dir, args.data, f"seed_{str(params.seed)}",
                                      f"cont_res{str(params.continuous_resolution)}", 'train.log'))

    logging.info("Loading the datasets...")

    if params.dataset == 'breast_cancer':
        train_set = uci_datasets.BreastCancerDataset('./data-uci', split='train', download=False, with_val=False)
        test_set = uci_datasets.BreastCancerDataset('./data-uci', split='test', with_val=False)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=params.batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=int(1e6), shuffle=False)
    elif params.dataset.startswith('monk'):
        style = int(params.dataset[4])
        train_set = uci_datasets.MONKsDataset('./data-uci', style, split='train', download=False, with_val=False)
        test_set = uci_datasets.MONKsDataset('./data-uci', style, split='test', with_val=False)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=params.batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=int(1e6), shuffle=False)
    else:
        args_dict = dict(vars(params))
        data_config = DataConfig(**(filter_dict_to_dataclass_fields(DataConfig, args_dict)))
        tabular_dataset = TabularDataset(data_config)
        # add input dim, num_classes
        params.input_dim = tabular_dataset.input_dim
        params.num_classes = tabular_dataset.num_classes
        
        train_loader, val_loader, test_loader = tabular_dataset.get_dataLoader(params.num_workers, val_split=0.8,
                                                                               split_seed=params.seed)

    
    logging.info("- done.")

    # initialize model
    model = net.Net(params).cuda() if params.cuda else net.Net(params)

    # compute inference cost
    hidden_dim = getattr(params, "hidden_dim", getattr(params, "num_channels", 128))
    num_hl_ops, num_bg_ops, ops_breakdown = net.get_mlp_basic_gate_ops(
        input_dim=params.input_dim,
        hidden_dim=hidden_dim,
        num_classes=params.num_classes,
    )
    print(f"Inference cost — high-level ops: {num_hl_ops}, basic gate ops: {num_bg_ops}")
    print(f"Ops breakdown: {ops_breakdown}")
    
    optimizer = optim.Adam(model.parameters(), lr=params.learning_rate)
    loss_fn = net.loss_fn
    metrics = net.metrics

    # train model
    logging.info("Starting training for {} epoch(s)".format(params.num_epochs))
    train_and_evaluate(model, args.data, train_loader, test_loader, optimizer, loss_fn, metrics, params, args.model_dir,
                       args.restore_file)
