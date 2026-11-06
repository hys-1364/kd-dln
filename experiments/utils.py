import random
import numpy as np
import torch
from dataclasses import fields
import os


# --------------------------------------------------------------------
# utility functions

def filter_dict_to_dataclass_fields(dataclass_type, input_dict):
    field_names = {field.name for field in fields(dataclass_type)}
    return {k: v for k, v in input_dict.items() if k in field_names}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_params_path(dataset, seed, make_dir=False, is_ray_tune=False):
    if not is_ray_tune:
        folder_name = "model_params"
    else:  # params search in Ray Tune
        folder_name = "ray_results"
    parent_dir_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    abs_param_path = os.path.join(parent_dir_path, f"{folder_name}/{dataset}/seed_{seed}")

    if make_dir and not os.path.exists(abs_param_path):
        os.makedirs(abs_param_path)
    return abs_param_path


def get_results_path(dataset, seed, make_dir=False):
    parent_dir_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_path = f"{parent_dir_path}/results/{dataset}/seed_{seed}"
    if make_dir and not os.path.exists(results_path):
        os.makedirs(results_path)
    return results_path


def get_rule_features_path(dataset, seed, make_dir=False):
    parent_dir_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_path = f"{parent_dir_path}/rule_features/{dataset}/seed_{seed}"
    if make_dir and not os.path.exists(results_path):
        os.makedirs(results_path)
    return results_path



# --------------------------------------------------------------------
# Approximated computation of number of logic gate operations
# NOTE: This is a simplified version of the actual computation, which should be both hardware and software dependent.

# return the number of basic logic gate level operations from higher-level operations of a specific data type
def get_num_basic_gate_ops_from_dtype(op, dtype):
    assert ('float' in dtype) or ('int' in dtype) or (dtype == 'bool')

    if 'float' in dtype:
        logic_ops_dict = get_basic_gate_ops_dict_from_floats(op, dtype)
    else:
        assert dtype in ['int8', 'int16', 'int32', 'int64', 'bool']
        logic_ops_dict = get_basic_gate_ops_dict_from_inp_bits(op, DTYPE_TO_BITS[dtype])

    return sum([LOGIC_TO_BASIC_GATE_OPS[k] * v for k, v in logic_ops_dict.items()])


# return the dict of 2-input logic gate level operations from higher-level floating point operations
def get_basic_gate_ops_dict_from_floats(op, dtype):
    assert op in ['equality', 'comparison', 'addition', 'subtraction', 'multiplication', 'division', 'logarithm']
    assert dtype in ['float16', 'float32', 'float64']

    # get the number of exponent bits and mantissa bits
    if dtype == 'float16':
        exponent_bits = 5
        mantissa_bits = 10
    elif dtype == 'float32':
        exponent_bits = 8
        mantissa_bits = 23
    else:  # dtype == 'float64'
        exponent_bits = 11
        mantissa_bits = 52

    if op == 'equality':
        return get_basic_gate_ops_dict_from_inp_bits('equality', exponent_bits + mantissa_bits + 1)

    if op == 'comparison':
        return get_basic_gate_ops_dict_from_inp_bits('comparison', exponent_bits + mantissa_bits + 1)

    if op == 'addition':
        return merge_dicts(get_basic_gate_ops_dict_from_inp_bits('comparison', exponent_bits),
                           get_basic_gate_ops_dict_from_inp_bits('addition', mantissa_bits + 1))

    if op == 'subtraction':
        return get_basic_gate_ops_dict_from_floats('addition', dtype)

    if op == 'multiplication':
        return merge_dicts({'XOR': 1},
                           merge_dicts(get_basic_gate_ops_dict_from_inp_bits('addition', exponent_bits),
                                       get_basic_gate_ops_dict_from_inp_bits('multiplication', mantissa_bits + 1)))

    if op == 'division':
        return merge_dicts({'XOR': 1},
                           merge_dicts(get_basic_gate_ops_dict_from_inp_bits('subtraction', exponent_bits),
                                       get_basic_gate_ops_dict_from_inp_bits('division', mantissa_bits + 1)))

    if op == 'logarithm':
        return merge_dicts(get_basic_gate_ops_dict_from_inp_bits('multiplication', exponent_bits),
                           merge_dicts(get_basic_gate_ops_dict_from_inp_bits('logarithm', mantissa_bits + 1),
                                       get_basic_gate_ops_dict_from_inp_bits('addition', mantissa_bits + 1)))


# return dict of number of 2-input logic gate level operations from higher-level operations with n bits of precision
def get_basic_gate_ops_dict_from_inp_bits(op, n):
    assert op in ['equality', 'comparison', 'addition', 'subtraction', 'multiplication', 'division', 'logarithm']
    # n represents the number of bits in the input
    assert isinstance(n, int) and n > 0

    if op == 'equality':
        return {'XNOR': n, 'AND': n-1}

    if op == 'comparison':
        return {'XNOR': n, 'OR': n-1, 'AND': n*(n-1)//2}

    if op == 'addition':
        return {'NAND': 5 + 9*(n-1)}

    if op == 'subtraction':
        return get_basic_gate_ops_dict_from_inp_bits('addition', n)

    if op == 'multiplication':
        return {'AND': n**2, 'NAND': 9*n**2 - 13*n + 9}

    if op == 'division':
        return merge_dicts(scale_dict_values(get_basic_gate_ops_dict_from_inp_bits('comparison', n), n + 1),
                           scale_dict_values(get_basic_gate_ops_dict_from_inp_bits('subtraction', n), n + 1))

    if op == 'logarithm':
        return scale_dict_values(
            merge_dicts(scale_dict_values(get_basic_gate_ops_dict_from_inp_bits('addition', n), n),
                        scale_dict_values(get_basic_gate_ops_dict_from_inp_bits('multiplication', n), n)),
            n)


def get_num_basic_gate_ops_aggregation(num_binary_inputs):
    logic_ops_dict = {'NAND': 5 * num_binary_inputs + 9 * num_binary_inputs}
    return sum([LOGIC_TO_BASIC_GATE_OPS[k] * v for k, v in logic_ops_dict.items()])


def get_num_ops_info_from_ops_dict(ops_dict):
    num_hl_ops, num_bg_ops = 0, 0
    for inp_dict in ops_dict.values():
        num_hl_ops += sum(inp_dict.values())

    for op, inp_dict in ops_dict.items():
        # basic gate
        if op in LOGIC_TO_BASIC_GATE_OPS.keys():
            if op in ['AND', 'OR', 'NAND', 'NOR']:
                num_bg_ops += sum([LOGIC_TO_BASIC_GATE_OPS[op] * (int(k) - 1) * v for k, v in inp_dict.items()])
            else:
                num_bg_ops += sum([LOGIC_TO_BASIC_GATE_OPS[op] * v for k, v in inp_dict.items()])

        elif op in ['aggregation']:
            num_bg_ops += sum([get_num_basic_gate_ops_aggregation(int(k)) * v for k, v in inp_dict.items()])

        elif op in ['equality', 'comparison', 'addition', 'subtraction', 'multiplication', 'division', 'logarithm']:
            num_bg_ops += sum([get_num_basic_gate_ops_from_dtype(op, k) * v for k, v in inp_dict.items()])

        else:
            raise ValueError(f"Unknown operation: {op}")
    return num_hl_ops, num_bg_ops


# update the ops_dict with the gate indices of the logic layer
def update_ops_dict_from_gate_indices(ops_dict, gate_indices):
    for idx in gate_indices:
        op_name = LOGIC_LAYER_IDX_TO_GATE[idx]
        ops_dict[op_name]["2"] += 1


LOGIC_TO_BASIC_GATE_OPS = {
    'NOT': 0,

    'AND': 1,
    'OR': 1,
    'NAND': 1,
    'NOR': 1,

    'XOR': 3,
    'XNOR': 3,

    'IMPLY': 1,
    'NIMPLY': 1,

    'False': 0,
    'True': 0,
    'PASS': 0,
    'NPASS': 0,
}


LOGIC_LAYER_IDX_TO_GATE = {
    0: 'False',
    1: 'AND',
    2: 'NIMPLY',
    3: 'PASS',
    4: 'NIMPLY',
    5: 'PASS',
    6: 'XOR',
    7: 'OR',
    8: 'NOR',
    9: 'XNOR',
    10: 'NPASS',
    11: 'IMPLY',
    12: 'NPASS',
    13: 'IMPLY',
    14: 'NAND',
    15: 'True',
}


DTYPE_TO_BITS = {
    'float16': 16,
    'float32': 32,
    'float64': 64,
    'int8': 8,
    'int16': 16,
    'int32': 32,
    'int64': 64,
    'bool': 1,
}



def scale_dict_values(original_dict, factor):
    return {k: v * factor for k, v in original_dict.items()}


def merge_dicts(d1, d2):
    merged_dict = d1.copy()
    for key, value in d2.items():
        if key in merged_dict:
            merged_dict[key] += value
        else:
            merged_dict[key] = value
    return merged_dict


def merge_defaultdict_to_another(d1, d2):
    for key, inner_dict in d2.items():
        for inner_key, value in inner_dict.items():
            d1[key][inner_key] += value


