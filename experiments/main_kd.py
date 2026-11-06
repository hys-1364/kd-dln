import argparse
import os
import pandas as pd
import torch
import json
import sys
import glob

import dnn_utils
import net

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.data import DataConfig, TabularDataset
from experiments.model import *
from experiments.train_logger import ResultsLogger, WandbLogger
from experiments.simplify_model import SimplifyLogicNet
from experiments.utils import *
from experiments.settings import *

import logging
logging_level = os.getenv('LOGGING_LEVEL', 'DEBUG').upper()
logging.basicConfig(level=logging.getLevelName(logging_level), format='%(message)s')


torch_num_threads = os.environ.get("TORCH_NUM_THREADS")
if torch_num_threads:
    logging.info(f"Setting torch num threads to {torch_num_threads}")
    torch.set_num_threads(int(torch_num_threads))

num_workers = 0
if os.environ.get("NUM_WORKERS"):
    num_workers = int(os.environ.get("NUM_WORKERS"))


def get_parser():
    # to accept bool values in str format
    def str2bool(v):
        return v.lower() in ('true', '1')

    # to accept None or int values
    def int_or_none(value):
        if value is None or value.lower() == 'null' or value.lower() == 'none':
            return None
        return int(value)

    parser = argparse.ArgumentParser(description='Train and/or evaluate differentiable logic network (DLN).')

    # Data
    parser.add_argument('--dataset', type=str, required=True,
                        help='The dataset to use. Note: input features should be scaled to [0, 1].')
    parser.add_argument('--seed', type=int, default=0, help='Random seed (default: 0).')

    # Train/Eval
    parser.add_argument('--train_model', type=str2bool, default=True,
                        help='Whether to train the model (default: True).')
    parser.add_argument('--evaluate_model', type=str2bool, default=False,
                        help='Whether to evaluate the model based on saved results (default: False).')

    # Training
    parser.add_argument('--num_epochs', type=int, default=1000, help='Number of epochs (default: 1000).')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size (default: 64).')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate (default: 0.01).')
    parser.add_argument('--balance_class_weights', type=str2bool, default=True,
                        help='Whether to adjust weights inversely proportional to class frequencies.')

    # Model
    parser.add_argument('--num_hidden_layers', type=int, default=3, help='Number of hidden layers (default: 3).')
    parser.add_argument('--first_hl_size', type=int, default=100, help='Number of neurons in the first hidden layer (default: 100).')
    parser.add_argument('--last_hl_size_wrt_first', type=float, default=0.5,
                        help='Number of neurons in the last hidden layer with respect to the first hidden layer size (default: 0.5).')
    parser.add_argument('--tau_out', type=float, default=1., help='The temperature for the output layer (default: 1.0).')
    parser.add_argument('--grad_factor', type=float, default=1., help='The gradient factor for all layers (default: 1.0).')
    parser.add_argument('--discretize_strategy', type=str, default='tree',
                        choices=['tree', 'uniform', 'quantile', 'kmeans'],
                        help='Strategy for discretizing continuous attributes, for thresholds initialization.')
    parser.add_argument('--continuous_resolution', type=int, default=4,
                        help='Number of thresholds for each continuous attributes discretization (default: 4).')
    parser.add_argument('--num_phase_rounds', type=int_or_none, default=None,
                        help='Number of full (neuron, link) phase rounds. '
                             'If not specified, will default to half of num_epochs, i.e., alternatively update neurons and links.')

    # Subsetting
    parser.add_argument('--subset_links', type=str2bool, default=True,
                        help='Whether to only search a subset of links in the LogicLayer (default: True).')
    parser.add_argument('--subset_link_num', type=int, default=8,
                        help='Number of subset links to keep if subset_links is True (default: 8).')
    parser.add_argument('--subset_gates', type=str2bool, default=True,
                        help='Whether to only search a subset of logic gates in the LogicLayer (default: True).')
    parser.add_argument('--subset_gate_num', type=int, default=8,
                        help='Number of subset gates to keep if subset_gates is True (default: 8).')

    # Distillation
    parser.add_argument('--alpha', type=float, default=0.1, help='The alpha value for knowledge distillation (default: 0.1).')
    parser.add_argument('--T', type=float, default=0.1, help='The temperature for knowledge distillation (default: 0.1).')


    # Optional arguments
    parser.add_argument('--ste_threshold_layer', type=str2bool, default=True,
                        help='Use the Straight-Through Estimator for ThresholdLayer (default: True).')
    parser.add_argument('--ste_logic_layer', type=str2bool, default=True,
                        help='Use the Straight-Through Estimator for LogicLayer (default: True).')
    parser.add_argument('--ste_sum_layer', type=str2bool, default=True,
                        help='Use the Straight-Through Estimator for SumLayer (default: True).')
    parser.add_argument('--phase_unified', type=str2bool, default=False,
                        help='Perform a unified phase (no separate neuron vs link phase) if True (default: False).')
    parser.add_argument('--gumbel_softmax', type=str2bool, default=False,
                        help='Use Gumbel-Softmax for the LogicLayer gates/links (default: False).')
    parser.add_argument('--gumbel_noise_scale', type=float, default=0.5,
                        help='Noise scale factor for Gumbel-Softmax sampling (default: 0.5).')
    parser.add_argument('--concat_input', type=str2bool, default=False,
                        help='Concatenate a ThresholdLayer to all intermediate (second to second last) LogicLayers.')
    parser.add_argument('--concat_bool', type=str2bool, default=False,
                        help='Concatenate non-trainable boolean inputs ([True, False]) to all LogicLayers.')

    # Evaluation frequency
    parser.add_argument('--eval_freq', type=int, default=10, help='Evaluation frequency during training (default: 10).')

    # After training
    parser.add_argument('--log_training', action='store_true',
                        help='Log and visualize model training process. Use eval_freq to control the frequency of logging.')
    parser.add_argument('--save_model', action='store_true', help='Save the trained model.')
    parser.add_argument('--simplify_model', action='store_true',
                        help='Simplify the trained model to a Python file that prints symbolic expressions. '
                             'To visualize the simplified model, run: python experiments/DLN_viz.py sympy_code_path graph_path.')

    return parser


def load_teacher(args, tabular_dataset):
    # load teacher model
    json_path = os.path.join('./experiments/base_model', args.dataset, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = dnn_utils.Params(json_path)

    setattr(params, "input_dim", tabular_dataset.input_dim)
    setattr(params, "num_classes", tabular_dataset.num_classes)
    setattr(params, "cuda", torch.cuda.is_available())
    setattr(params, "seed", int(args.seed))

    # save previous seed
    torch_rng_state = torch.get_rng_state()
    cuda_rng_state = torch.cuda.get_rng_state_all()
    np_rng_state = np.random.get_state()
    py_rng_state = random.getstate()
    
    # set to teacher seed
    set_seed(params.seed)
    teacher_model = net.Net(params)
    
    # return back
    torch.set_rng_state(torch_rng_state)
    torch.cuda.set_rng_state_all(cuda_rng_state)
    np.random.set_state(np_rng_state)
    random.setstate(py_rng_state)
    
    device = torch.device("cuda" if params.cuda and torch.cuda.is_available() else "cpu")
    teacher_model.to(device)

    # load checkpoint
    checkpoint = torch.load(f"./experiments/base_model/{args.dataset}/seed_{str(args.seed)}/"
                            f"cont_res{str(args.continuous_resolution)}/best.pth.tar", map_location=device)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        teacher_model.load_state_dict(checkpoint["state_dict"])
    else:
        teacher_model.load_state_dict(checkpoint)

    # eval mode only
    teacher_model.eval()
    return teacher_model


def main(args):
    if args.train_model:
        train_model(args)
    if args.evaluate_model:
        eval_results = evaluate_model(args.dataset, args.seed, args)


def train_model(args):
    # create path
    parent_dir_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_path = f"{parent_dir_path}/results/{args.dataset}/seed_{args.seed}"
    if True and not os.path.exists(results_path):
        os.makedirs(results_path)

    results = ResultsLogger(path=results_path)
    results.store_args(args)
    wandb_logger = WandbLogger(args.log_training, f"{args.dataset}_{args.seed}")

    # set seed
    set_seed(args.seed)

    # load data
    args_dict = dict(vars(args))
    data_config = DataConfig(**(filter_dict_to_dataclass_fields(DataConfig, args_dict)))
    tabular_dataset = TabularDataset(data_config)
    train_loader, val_loader, test_loader = tabular_dataset.get_dataLoader(num_workers, val_split=0.8, split_seed=args.seed)
    train_loader_eval = tabular_dataset.get_dataLoader(num_workers, isEval=True, val_split=0.8, split_seed=args.seed)[0]
    loaders = {'train': train_loader_eval, 'val': val_loader, 'test': test_loader}
    metric_functions = {'loss': get_loss, 'acc': get_bl_acc}
    modes = [True, False]  # True for train, False for eval

    # create student model
    model, criterion, optimizer = get_model(args_dict, tabular_dataset)
    results.store_results({'model_str': str(model)})

    # teacher model
    teacher_model = load_teacher(args, tabular_dataset)
    device = next(teacher_model.parameters()).device
    

    ######### START TRAINING #########
    best_score = -float('inf')
    best_epoch = None
    
    for cur_epoch in range(1, args.num_epochs + 1):
        # TODO: use train_phase_kd_epoch() with same params for single phase KD
        train_kd_epoch(model, teacher_model, train_loader, args.alpha, args.T, criterion, optimizer)

        # evaluate
        is_eval_epoch = (cur_epoch == 1 or cur_epoch % args.eval_freq == 0 or cur_epoch == args.num_epochs)
        if not is_eval_epoch:
            continue
        metrics = {}
        for loader_name, loader in loaders.items():
            for metric_name, metric_function in metric_functions.items():
                for mode in modes:
                    mode_name = 'train_mode' if mode else 'eval_mode'
                    key = f"{metric_name}_{loader_name}Loader_{mode_name}"
                    if metric_name == 'loss':
                        metrics[key] = metric_function(model, loader, criterion, mode=mode)
                    else:
                        metrics[key] = metric_function(model, loader, mode=mode)
                        
        results.store_results(metrics)
        wandb_logger.log_metrics(metrics, cur_epoch)
            
        cur_score = metrics['acc_valLoader_eval_mode'] + metrics['acc_trainLoader_eval_mode']
        print(f"cur_score:{cur_score}")
        
        if cur_score >= best_score:
            best_score = cur_score
            best_epoch = cur_epoch
            if args.save_model:
                model.save_model(f"{results_path}/model.pth")
                print(f"{results_path}/model.pth")
    ######### END TRAINING #########
    
    args.best_kd_epoch = best_epoch
    args.best_kd_score = best_score
          
    results.save()
    wandb_logger.log_params(vars(args))
    wandb_logger.finish()

    # saving model
    if args.save_model:
        model.save_model(f"{results_path}/model_last.pth")
        print(f"{results_path}/model_last.pth")

    # simplying model
    if args.simplify_model:
        # use the best model if available
        if args.save_model and os.path.exists(f"{results_path}/model.pth"):
            model.load_model(f"{results_path}/model.pth")

        sympy_model = SimplifyLogicNet(
            model=model,
            feat_names=tabular_dataset.feat_names,
            scaler_params=tabular_dataset.scaler_params,
            dtype_dict=tabular_dataset.dtype_dict,
        )
        sympy_code = sympy_model.get_sympy_code()

        with open(f"{results_path}/sympy_code.py", 'w') as f:
            f.write(sympy_code)


def evaluate_model(dataset, seed, args):
    eval_results = {'dataset': dataset, 'seed': seed}
    bits_for_values = DTYPE_TO_BITS[dtype_for_floats]

    # establish paths
    parent_dir_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_path = f"{parent_dir_path}/results/{args.dataset}/seed_{args.seed}"
    results_json_path = os.path.join(results_path, "results.json")
    if not os.path.exists(results_json_path):
        logging.error(f"Training results file not found. dataset: {dataset}, seed: {seed}")
        return eval_results

    # open results file
    with open(results_json_path, 'r') as f:
        all_results = json.load(f)
    if "args" not in all_results:
        logging.error(f"Training results file is missing 'args' key. dataset: {dataset}, seed: {seed}")
        return eval_results
    params = all_results["args"]

    # load data
    data_config = DataConfig(**(filter_dict_to_dataclass_fields(DataConfig, params)))
    tabular_dataset = TabularDataset(data_config)
    _, val_loader, test_loader = tabular_dataset.get_dataLoader(num_workers=num_workers, isEval=True, val_split=0.8, split_seed=seed)

    # retrieve best model
    model_path = os.path.join(results_path, "model.pth")
    if not os.path.exists(model_path):
        logging.error(f"Model file not found. dataset: {dataset}, seed: {seed}")
        return eval_results
    model = get_model(params, tabular_dataset)[0]
    model.load_model(model_path)
    model.eval()

    # metrics
    val_metrics = evaluate_val_accuracy(model, val_loader)
    acc_metrics = evaluate_accuracy(model, test_loader)
    
    # get the simplified model
    sympy_code_path = f"{results_path}/sympy_code.py"
    sympy_model = SimplifyLogicNet(
        model=model,
        feat_names=tabular_dataset.feat_names,
        scaler_params=tabular_dataset.scaler_params,
        dtype_dict=tabular_dataset.dtype_dict,
    )
    sympy_code = sympy_model.get_sympy_code()
    with open(sympy_code_path, 'w') as f:
        f.write(sympy_code)

    # for vizualiation
    graph_path = os.path.join(results_path, "DLN_viz")

    # final metrics
    comp_metrics = evaluate_computation(
        model=model,
        sympy_code_path=sympy_code_path,
        graph_path=graph_path,
        bits_for_values=bits_for_values,
        bits_for_indices=bits_for_indices,
        simplify_timeout_second=simplify_timeout_second,
    )

    eval_results.update({**val_metrics, **acc_metrics, **comp_metrics})

    # create final eval file
    with open(f"{results_path}/eval_results.json", "w") as f:
            json.dump(eval_results, f, indent=4)

    return eval_results



if __name__ == '__main__':
    parser = get_parser()
    args, _ = parser.parse_known_args()
    main(args)
