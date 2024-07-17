import main
import main_fwt
import cil_inference
import dil_inference
import argparse
import yaml
import sys
import json
import time
import datetime
import numpy as np
from pathlib import Path
import copy


def convert_to_serializable(obj):
    if isinstance(obj, np.float32):
        return float(obj)
    elif isinstance(obj, np.int32):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()  # Convert NumPy arrays to lists
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    # Handle other types as needed (e.g., datetime objects)
    else:
        return str(obj)


DIFFERENT_LAYERS = {
    "f1": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "i1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    "f4": [0, 1, 2, 3, 4, 5, 6, 7],
    "i4": [4, 5, 6, 7, 8, 9, 10, 11],
    "i3f3": [3, 4, 5, 6, 7, 8],
    "odd": [0, 2, 4, 6, 8, 10],
    "even": [1, 3, 5, 7, 9, 11],
    "all": [],
}

DIFFERENT_LAYER_EMBEDDINGS = {
    # "learnable": "learnable",
    # "fixed_random": "fixed_random",
    # "fixed_sine": "fixed_sine",
    # "fixed_onehot": "fixed_onehot",
    "throughout_learnable": "throughout_learnable",
}

DIFFERENT_SEEDS = {
    "0": 0,
    "42": 42,
    "1234": 1234,
}


def experiments_with_seeds(args):
    print(f"Running Experiments\n")
    start_time = time.time()
    results = {}
    avg_fwt, avg_bwt, avg_fgt, avg_cil, avg_til = [], [], [], [], []
    lr = copy.deepcopy(args.lr)
    for z in DIFFERENT_LAYER_EMBEDDINGS.keys():
        args.exp_name = z
        args.layer_embedding_type = DIFFERENT_LAYER_EMBEDDINGS[z]
        args.leave_out = DIFFERENT_LAYERS["all"]
        args.seed = DIFFERENT_SEEDS["42"]
        args.order = None
        seed_start_time = time.time()
        print(
            f"================================= Experiment : {z} ==================================="
        )
        results[z] = {}
        args.lr = lr
        cl_acc_matrix = main.main(args)
        args.lr = lr
        it_acc_matrix = main_fwt.main(args)
        args.pretrained_weights = (
            args.model_store_path + f"model_weights_{args.exp_name}.pth"
        )
        cil_avg_acc = (
            cil_inference.main(args) if not args.dil else dil_inference.main(args)
        )

        # Calculate Average Accuracy for TIL
        avg_acc = np.mean(cl_acc_matrix[:, -1])
        results[z]["til_avg_acc"] = avg_acc
        avg_til.append(avg_acc)
        # Forward Transfer Calculation
        fwt = np.mean(np.diag(cl_acc_matrix) - np.diag(it_acc_matrix))
        results[z]["fwt"] = fwt
        avg_fwt.append(fwt)
        # Backward Transfer Calculation
        bwt = np.mean(cl_acc_matrix[:, -1] - np.diag(cl_acc_matrix))
        results[z]["bwt"] = bwt
        avg_bwt.append(bwt)
        # Calculate Forgetting
        fgt = np.mean(np.max(cl_acc_matrix, axis=1) - cl_acc_matrix[:, -1])
        results[z]["fgt"] = fgt
        avg_fgt.append(fgt)
        # Calculate CIL Average Accuracy
        results[z]["cil_avg_acc"] = cil_avg_acc
        avg_cil.append(cil_avg_acc)
        print(
            "*************************************** RESULTS ***************************************"
        )
        print(json.dumps(results[z], indent=3))
        total_time = time.time() - seed_start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print(f"Total training time: {total_time_str}")
        print(
            "*****************************************************************************************"
        )

    print(
        "*************************************** FINAL RESULTS ***************************************"
    )
    # Calculate Final Results
    results["final"] = {}
    results["final"]["avg_til"] = np.mean(avg_til)
    results["final"]["std_til"] = np.std(avg_til)
    results["final"]["avg_fwt"] = np.mean(avg_fwt)
    results["final"]["std_fwt"] = np.std(avg_fwt)
    results["final"]["avg_bwt"] = np.mean(avg_bwt)
    results["final"]["std_bwt"] = np.std(avg_bwt)
    results["final"]["avg_fgt"] = np.mean(avg_fgt)
    results["final"]["std_fgt"] = np.std(avg_fgt)
    results["final"]["avg_cil"] = np.mean(avg_cil)
    results["final"]["std_cil"] = np.std(avg_cil)

    results = convert_to_serializable(results)
    with open(args.output_dir + f"results_order.json", "w") as f:
        json.dump(results, f, indent=4)

    # Printing the final results
    print(json.dumps(results["final"], indent=3))
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Total training time: {total_time_str}")
    print(
        "*********************************************************************************************"
    )
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser("AdaPrefix++ Arguments Parser")

    config = parser.parse_known_args()[-1][0]
    subparser = parser.add_subparsers(dest="subparser_name")

    if config == "cifar100":
        from parsers.cifar100 import get_args_parser

        config_parser = subparser.add_parser("cifar100", help="Split-CIFAR100 configs")
    elif config == "five_datasets":
        from parsers.five_datasets import get_args_parser

        config_parser = subparser.add_parser("five_datasets", help="5-Datasets configs")
    elif config == "imagenet_r":
        from parsers.imagenet_r import get_args_parser

        config_parser = subparser.add_parser(
            "imagenet_r", help="Split-ImageNet-R configs"
        )
    elif config == "cddb":
        from parsers.cddb import get_args_parser

        config_parser = subparser.add_parser("cddb", help="DIL CDBB configs")
    else:
        raise NotImplementedError

    get_args_parser(config_parser)
    print("###########################################################################")
    print(
        f"###################### EXPERIMENT CONFIG: {config.upper()} ########################"
    )
    print(
        "###########################################################################\n"
    )
    # Required arguments updated using yaml file
    args = parser.parse_args()
    opt = yaml.load(open(args.config), Loader=yaml.FullLoader)
    args = vars(args)
    args.update(opt)

    args = argparse.Namespace(**args)

    args.model_store_path = args.output_dir.replace("outputs", "weights")
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    if args.model_store_path:
        Path(args.model_store_path).mkdir(parents=True, exist_ok=True)

    _ = experiments_with_seeds(args)

    sys.exit(0)
