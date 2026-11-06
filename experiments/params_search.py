import numpy as np
import pandas as pd
import json
import os
import torch
import time
import shutil
import ray

from ray import tune
from ray.tune.search import ConcurrencyLimiter
from ray.tune.search.optuna import OptunaSearch
from ray.train import RunConfig
from ray.air import session

import dnn_utils
import net
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.data import DataConfig, TabularDataset, get_data_stats_tune
from experiments.model import get_model, train_epoch, train_kd_epoch, get_bl_acc, is_valid_config
from experiments.train_logger import NumpyEncoder
from experiments.utils import *
from experiments.settings import *

import logging
logging_level = logging.getLevelName(logging_level)
logging.basicConfig(level=logging_level, format='%(message)s')
logger_ray = logging.getLogger("ray")
logger_ray.setLevel(logging_level)

os.environ["RAY_AIR_NEW_OUTPUT"] = "0"
os.environ["RAY_IGNORE_UNHANDLED_ERRORS"] = "1"
os.environ["TUNE_WARN_EXCESSIVE_EXPERIMENT_CHECKPOINT_SYNC_THRESHOLD_S"] = "0"
os.environ["RAY_SCHEDULER_EVENTS"] = "0"

import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.filterwarnings(action="ignore", category=UserWarning, module="ray.tune")
if tune_search_alg != 'Grid':
    # some trials may be already initiated before stopper's stop_all has returned True
    warnings.filterwarnings("ignore", message=".*Could not fetch metrics.*")  # does not work


def params_search(datasets, seeds):
    ray.init(ignore_reinit_error=True, log_to_driver=True, include_dashboard=False)
    ray_tasks = []
    for seed in seeds:
        for dataset in datasets:
            # in slurm, have trouble to make HPO tasks run in parallel. Ray version: 2.41.0
            if local_laptop:
                ray_tasks.append(params_search_tune_wrapper.options(
                    num_cpus=total_num_cpus_hpo, num_gpus=total_num_gpus_hpo).remote(dataset, seed))
            else:
                params_search_tune(dataset, seed)
    if local_laptop:
        results = ray.get(ray_tasks)
    time.sleep(1)  # wait for console to display the last print statement, if applicable
    ray.shutdown()


@ray.remote(num_cpus=total_num_cpus_hpo, num_gpus=total_num_gpus_hpo)
def params_search_tune_wrapper(dataset, seed):
    return params_search_tune(dataset, seed)


def params_search_tune(dataset, seed):
    input_dim, num_classes, num_train_samples = get_data_stats_tune(dataset, seed)

    search_space = {
        'dataset': dataset,
        'seed': seed,

        'num_epochs': tune_max_num_epochs,
        # set batch_size to [32 * (smallest power of 2 that's >= num_classes)], but no more than 128
        'batch_size': 32 * min(4, max(2, 2 ** ((num_classes - 1).bit_length()))),
        'balance_class_weights': True,
    }

    lr_options = [1e-2, 3e-2, 1e-1, 3e-1]  # DLN has many softmax functions, so using large lr
    search_space.update({
        'learning_rate':
            tune.grid_search(lr_options) if tune_search_alg == 'Grid' else
            tune.loguniform(lr_options[0], lr_options[-1]),
    })

    hl_options = [2, 3, 5, 7, 9]
    search_space.update({
        'num_hidden_layers':
            tune.grid_search(hl_options) if tune_search_alg == 'Grid' else
            tune.randint(hl_options[0], hl_options[-1] + 1),
    })

    fhs_options = get_first_hidden_dim_options(input_dim, num_classes, num_train_samples)
    search_space.update({
        'first_hl_size':
            tune.grid_search(fhs_options) if tune_search_alg == 'Grid' else
            tune.randint(fhs_options[0], fhs_options[-1] + 1),
    })

    lhs_options = [2**i for i in [-2, -1, 0]]
    search_space.update({
        'last_hl_size_wrt_first':
            tune.grid_search(lhs_options) if tune_search_alg == 'Grid' else
            tune.loguniform(lhs_options[0], lhs_options[-1], base=2),
    })

    search_space.update({
        'discretize_strategy': 'tree',
        'continuous_resolution':
            tune.grid_search([4, 6]) if tune_search_alg == 'Grid' else
            tune.choice([4, 6]),
    })

    tau_out_options = [1, 10, 30]
    search_space.update({
        'tau_out':
            tune.grid_search(tau_out_options) if tune_search_alg == 'Grid' else
            tune.loguniform(tau_out_options[0], tau_out_options[-1], base=2),
    })

    grad_factor_options = [1, 1.5, 2]
    search_space.update({
        'grad_factor':
            tune.grid_search(grad_factor_options) if tune_search_alg == 'Grid' else
            tune.uniform(grad_factor_options[0], grad_factor_options[-1]),
    })

    # KD params
    T_options = [1, 2, 4, 8, 16]
    search_space.update({
        'T':
            tune.grid_search(T_options) if tune_search_alg == 'Grid' else
            tune.loguniform(T_options[0], T_options[-1], base=2),
    })

    alpha_options = [0.1, 0.3, 0.5, 0.7, 0.9]
    search_space.update({
        'alpha':
            tune.grid_search(alpha_options) if tune_search_alg == 'Grid' else
            tune.uniform(alpha_options[0], alpha_options[-1]),
    })

    
    search_space.update({
        'num_phase_rounds': None,  # None means alternating phase

        'concat_input': True,

        'concat_bool': False,

        'tau_init': 1,
        'tau_decay': 1,
        'tau_min': 1,
    })

    num_cpus_trial, num_gpus_trial = get_resource_per_run(dataset, 'hpo')
    trainable = tune.with_resources(
        train_nn_tune,
        resources={"cpu": num_cpus_trial, "gpu": num_gpus_trial},
    )

    if tune_search_alg == 'Grid':
        search_alg = None
    else:  # Optuna
        search_alg = ConcurrencyLimiter(
            OptunaSearch(metric='val_score', mode='max', seed=seed),
            max_concurrent=tune_max_concurrent,)

    _num_samples = 1 if tune_search_alg == 'Grid' else -1
    tune_config = tune.TuneConfig(
        num_samples=_num_samples,
        search_alg=search_alg,
        scheduler=None,
    )

    _stop = None if tune_search_alg == 'Grid' else NumValidTrialsStopper(tune_samples)
    storage_path = get_params_path(dataset, seed, make_dir=True, is_ray_tune=True)
    run_config = RunConfig(
        log_to_file=False,
        storage_path=storage_path,
        verbose=tune_verbose,
        stop=_stop,
    )

    tuner = tune.Tuner(
        trainable=trainable,
        param_space=search_space,
        tune_config=tune_config,
        run_config=run_config,
    )
    start_time = time.time()
    results = tuner.fit()
    time_cost = int(time.time() - start_time)

    trials_dataframe = results.get_dataframe()
    best_result = results.get_best_result('val_score', 'max', scope='last')
    params_folder = get_params_path(dataset, seed, make_dir=True)
    with open(f"{params_folder}/params.json", 'w') as wf:
        wf.write(json.dumps(best_result.config, cls=NumpyEncoder, indent=4))
    trials_dataframe.to_csv(f"{params_folder}/hpo.csv", index=False)

    # Note: time measure might not be accurate if run multiple HPOs in parallel
    logging.info(f"HPO done. {dataset, seed}. "
                 f"Time: {'{}:{:02d}'.format(time_cost//60, time_cost%60)}.")

    # remove temporary tune files
    experiment_path = results.experiment_path
    if remove_tune_dir:
        if os.path.exists(experiment_path):
            shutil.rmtree(experiment_path, ignore_errors=True)

            
def load_teacher_for_tune(config, tabular_dataset):
    # mirror load_teacher, but device match student
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    script_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(script_dir, "base_model", config['dataset'], "params.json")
    assert os.path.isfile(json_path), f"No json configuration file found at {json_path}"
    params = dnn_utils.Params(json_path)

    setattr(params, "input_dim", tabular_dataset.input_dim)
    setattr(params, "num_classes", tabular_dataset.num_classes)
    setattr(params, "cuda", torch.cuda.is_available())
    setattr(params, "seed", config['seed'])

    # save seed
    torch_rng_state = torch.get_rng_state()
    cuda_rng_state = torch.cuda.get_rng_state_all()
    np_rng_state = np.random.get_state()
    py_rng_state = random.getstate()

    set_seed(params.seed)
    teacher_model = net.Net(params)

    # return seed
    torch.set_rng_state(torch_rng_state)
    torch.cuda.set_rng_state_all(cuda_rng_state)
    np.random.set_state(np_rng_state)
    random.setstate(py_rng_state)

    # final teacher model
    teacher_model.to(device)
    checkpoint = torch.load(
        os.path.join(script_dir, "base_model", config['dataset'], f"seed_{str(config['seed'])}", f"cont_res{str(config['continuous_resolution'])}", "best.pth.tar"),
        map_location=device)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        teacher_model.load_state_dict(checkpoint["state_dict"])
    else:
        teacher_model.load_state_dict(checkpoint)

    # eval mode only
    teacher_model.eval()
    return teacher_model


def train_nn_tune(config):
    torch.set_num_threads(1)
    
    data_config = DataConfig(**(filter_dict_to_dataclass_fields(DataConfig, config)))
    tabular_dataset = TabularDataset(data_config)
 
    if not is_valid_config(config, tabular_dataset):
        if tune_verbose > 1:
            logging.info(f"Invalid input config, stopping the trial. config: {config}, dataset: {tabular_dataset}")
        session.report({'val_score': -1e-9, 'done': True})
        return
 
    set_seed(config['seed'])
    num_workers = 0

    # load teacher model
    teacher_model = load_teacher_for_tune(config, tabular_dataset)

    # set cross fold validation
    tune_cvn = get_hpo_cvn(tabular_dataset.num_train_samples)
    num_checkpoints = int(config['num_epochs'] // tune_eval_freq)
    fold_scores = np.zeros((tune_cvn, num_checkpoints))
    
    fold_evaluated = np.zeros((tune_cvn, num_checkpoints), dtype=bool)
    data_loader_generator = tabular_dataset.get_cv_dataLoader(tune_cvn, num_workers)
    for fold_idx in range(tune_cvn):
        train_loader, val_loader = next(data_loader_generator)
        model, criterion, optimizer = get_model(config, tabular_dataset)
        best_acc, best_epoch = 0, 0
        early_stop = False

        # train
        for epoch in range(1, config['num_epochs'] + 1):
            if early_stop:
                break
            train_kd_epoch(model, teacher_model, train_loader,
                           config['alpha'], config['T'], criterion, optimizer)
            if epoch % tune_eval_freq == 0:
                val_acc = get_bl_acc(model, val_loader, False)
                checkpoint_idx = epoch // tune_eval_freq - 1
                fold_scores[fold_idx, checkpoint_idx] = val_acc
                fold_evaluated[fold_idx, checkpoint_idx] = True
                if val_acc > best_acc:
                    best_acc, best_epoch = val_acc, epoch
                elif epoch - best_epoch >= tune_early_stop_patience:
                    early_stop = True
 
    masked_scores = np.ma.masked_array(fold_scores, mask=~fold_evaluated)
    mean_scores = np.ma.mean(masked_scores, axis=0)
    std_scores = np.ma.std(masked_scores, axis=0)

    # checkpoints no fold reached become -inf so np.max never selects them
    penalized_scores = np.ma.filled(mean_scores - 0.1 * std_scores, -np.inf)
    best_val_score = np.max(penalized_scores)
    session.report({'val_score': best_val_score, 'done': True})
    return


# NOTE: for new datasets, check whether this outputs reasonable values
def get_first_hidden_dim_options(input_dim, output_dim, num_train_samples) -> list[int]:
    option_1 = input_dim
    option_2 = input_dim * 2
    options = [option_1, option_2]
    option_3 = num_train_samples // (2 * (input_dim + output_dim))  # ref: https://stats.stackexchange.com/q/136542
    if option_3 != option_1 and option_3 != option_2 and option_3 >= option_1 // 2:
        options.append(min(option_3, input_dim * 3))
    elif input_dim > 1:
        options.append(input_dim // 2)
    return sorted(options)



class NumValidTrialsStopper(tune.Stopper):
    def __init__(self, num_trials=100):
        self.num_trials = num_trials
        self.num_valid_trials = 0

    def __call__(self, trial_id: str, result: dict) -> bool:
        if result['val_score'] < 0:
            return True
        if result['done']:
            self.num_valid_trials += 1
            return True
        return False

    def stop_all(self) -> bool:
        return self.num_valid_trials >= self.num_trials


