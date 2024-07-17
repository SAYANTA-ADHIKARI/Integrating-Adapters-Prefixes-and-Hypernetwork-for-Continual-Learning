import main
import cil_inference
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


EXPERIMENTS = {
    "10": 10,
    "1": 1,
    "42": 42,
}


def experiments_with_seeds(args):
    # Running Expewriments with seeds 10, 1, 42
    print(f"Running Experiments\n")
    start_time = time.time()
    results = {}
    avg_cil, avg_til = [], []
    lr = copy.deepcopy(args.lr)
    for z in EXPERIMENTS.keys():
        args.exp_name = z
        args.prefix_mlp = True
        args.has_atten_adapter = False
        args.has_output_adapter = False
        seed_start_time = time.time()
        print(
            f"================================= Experiment: {z} ==================================="
        )
        results[z] = {}
        args.seed = EXPERIMENTS[z]
        args.lr = lr
        cl_acc_matrix = main.main(args)
        args.pretrained_weights = (
            args.model_store_path + f"model_weights_{args.exp_name}.pth"
        )
        cil_avg_acc = cil_inference.main(args)

        # Calculate Average Accuracy for TIL
        avg_acc = np.mean(cl_acc_matrix[:, -1])
        results[z]["til_avg_acc"] = avg_acc
        avg_til.append(avg_acc)
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
    results["final"]["avg_cil"] = np.mean(avg_cil)
    results["final"]["std_cil"] = np.std(avg_cil)

    results = convert_to_serializable(results)
    with open(args.output_dir + f"results_{args.exp_name}.json", "w") as f:
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
