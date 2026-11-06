# Knowledge Distillation from Neural Networks to Differentiable Logic Networks

This repository contains the implementation of our paper/preprint: **"Knowledge Distillation from Neural Networks to Differentiable Logic Networks"**, Hee Yun Suh and Niraj K. Jha.
## Getting Started

1. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```
   We conducted experiments using Python 3.12.

2. Prepare datasets by following some standards (i.e., data path, a specific order of categorical features, continuous features, and target). An example notebook can be found in [`quickstart/prepare_dataset.ipynb`](quickstart/prepare_dataset.ipynb).

3. Train the teacher MLP model from the experiments folder.
   ```
   python train.py \
   --data Heart \
   --seed 0 \
   --cont_res 4
   ```
5. Train a DLN with knowledge distillation and evaluate. An example notebook can be found in [`quickstart/train_eval_viz.ipynb`](quickstart/train_eval_viz.ipynb). An example command is:
   ```
   python experiments/main_kd.py \
   --train_model True \
   --evaluate_model True \
   --dataset Heart \
   --seed 0 \
   --num_epochs 1000 \
   --batch_size 64 \
   --learning_rate 0.2 \
   --tau_out 3 \
   --grad_factor 1.2 \
   --first_hl_size 50 \
   --last_hl_size_wrt_first 0.25 \
   --num_hidden_layers 4 \
   --discretize_strategy tree \
   --continuous_resolution 4 \
   --alpha 0.1 \
   --T 0.3 \
   --concat_input True \
   --save_model
   ```
   When wandb is enabled, you can add ```--log_training``` to log the training process. There are many other features, and we encourage you to explore the arguments in our `experiments/main_kd.py` file.

6. Visualize the model using the SymPy code saved in the previous step:
   ```
   python experiments/DLN_viz.py results/Heart/seed_0/sympy_code.py quickstart/example/viz
   ```
   A graph named `viz.png` will be generated. If the generation takes too long, check the parts marked with "NOTE" in the `experiments/simplify_model.py` file.

## Running multiple experiments in parallel

We use Ray to run experiments in parallel. Here are the steps:

1. Specify datasets, seed sets, computing resource settings, etc. in `experiments/settings.py`.

2. Hyperparameter optimization (HPO) using Ray Tune:
   ```
   TORCH_NUM_THREADS=1 python experiments/run_experiments.py --params-search
   ```
   The HPO results and the selected hyperparameters will be saved in the `model_params` directory. You can use [`experiments/params_search_analysis.ipynb`](experiments/params_search_analysis.ipynb) to research how different factors affect performance. In general, learning rate (and its scaling factor, output temperature) is the most important factor, and for DLNs, a large initial learning rate is needed because they have layers of Softmax functions.

3. Train models using selected hyperparameters:
   ```
   TORCH_NUM_THREADS=1 python experiments/run_experiments.py --train
   ```
   The trained models, training logs, and corresponding SymPy code will be saved in the `results` folder.

4. Evaluate models:
   ```
   python experiments/run_experiments.py --evaluate
   ```
   A summary of evaluation results for accuracies and model sizes will be saved in `results/evaluation.csv`.

## Citation

