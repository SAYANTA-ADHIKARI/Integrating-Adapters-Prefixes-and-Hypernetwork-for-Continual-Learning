import argparse


def get_args_parser(subparsers):
    subparsers.add_argument(
        "--config",
        help="configuration file *.yml",
        type=str,
        required=False,
        default="config.yml",
    )
    subparsers.add_argument(
        "--batch-size", default=16, type=int, help="Batch size per device"
    )
    subparsers.add_argument("--epochs", default=5, type=int)

    # Model parameters
    subparsers.add_argument(
        "--model-name",
        default="vit_base_patch16_224",
        type=str,
        metavar="MODEL",
        help="Name of model to train",
    )
    subparsers.add_argument(
        "--input-size", default=224, type=int, help="images input size"
    )
    subparsers.add_argument(
        "--pretrained", default=True, help="Load pretrained model or not"
    )
    subparsers.add_argument(
        "--pretrained-weights",
        default="./pretrained_models_weights/",
        type=str,
        metavar="MODEL-PATH",
        help="Path of the pretrained model that needs to loaded",
    )
    subparsers.add_argument(
        "--drop",
        type=float,
        default=0.0,
        metavar="PCT",
        help="Dropout rate (default: 0.)",
    )
    subparsers.add_argument(
        "--drop-path",
        type=float,
        default=0.0,
        metavar="PCT",
        help="Drop path rate (default: 0.)",
    )

    # Optimizer parameters
    subparsers.add_argument(
        "--opt",
        default="adam",
        type=str,
        metavar="OPTIMIZER",
        help='Optimizer (default: "adam")',
    )
    subparsers.add_argument(
        "--opt-eps",
        default=1e-8,
        type=float,
        metavar="EPSILON",
        help="Optimizer Epsilon (default: 1e-8)",
    )
    subparsers.add_argument(
        "--opt-betas",
        default=(0.9, 0.999),
        type=float,
        nargs="+",
        metavar="BETA",
        help="Optimizer Betas (default: (0.9, 0.999), use opt default)",
    )
    subparsers.add_argument(
        "--clip-grad",
        type=float,
        default=1.0,
        metavar="NORM",
        help="Clip gradient norm (default: None, no clipping)",
    )
    subparsers.add_argument(
        "--opt-momentum",
        type=float,
        default=0.9,
        metavar="M",
        help="SGD momentum (default: 0.9)",
    )
    subparsers.add_argument(
        "--opt-nev",
        action="store_true",
        help="SGD Nesterov momentum",
    )
    subparsers.add_argument(
        "--weight-decay", type=float, default=0.0, help="weight decay (default: 0.0)"
    )

    # Learning rate schedule parameters
    subparsers.add_argument(
        "--sched",
        default="None",
        type=str,
        metavar="SCHEDULER",
        help='LR scheduler (default: "constant")',
    )
    subparsers.add_argument(
        "--lr",
        type=float,
        default=0.03,
        metavar="LR",
        help="learning rate (default: 0.03)",
    )
    subparsers.add_argument(
        "--step-size",
        type=int,
        default=30,
        metavar="N",
        help="epoch interval to decay LR",
    )
    subparsers.add_argument(
        "--step-mult",
        type=int,
        default=1,
        metavar="N",
        help="epoch interval multiplier to decay LR",
    )
    subparsers.add_argument(
        "--scale",
        type=float,
        default=0.1,
        metavar="RATE",
        help="LR scale / decay rate (default: 0.1)",
    )
    subparsers.add_argument(
        "--eta-min",
        type=float,
        default=0.0001,
        metavar="Min LR",
        help="minimum learning rate (default: 0.0001)",
    )
    subparsers.add_argument(
        "--verbose",
        action="store_true",
        help="Print details of LR",
    )

    # Augmentation parameters
    subparsers.add_argument(
        "--color-jitter",
        type=float,
        default=None,
        metavar="PCT",
        help="Color jitter factor (default: 0.3)",
    )
    subparsers.add_argument(
        "--aa",
        type=str,
        default=None,
        metavar="NAME",
        help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m9-mstd0.5-inc1)',
    ),
    subparsers.add_argument(
        "--smoothing", type=float, default=0.1, help="Label smoothing (default: 0.1)"
    )
    subparsers.add_argument(
        "--train-interpolation",
        type=str,
        default="bicubic",
        help='Training interpolation (random, bilinear, bicubic default: "bicubic")',
    )

    # * Random Erase params
    subparsers.add_argument(
        "--reprob",
        type=float,
        default=0.0,
        metavar="PCT",
        help="Random erase prob (default: 0.25)",
    )
    subparsers.add_argument(
        "--remode",
        type=str,
        default="pixel",
        help='Random erase mode (default: "pixel")',
    )
    subparsers.add_argument(
        "--recount", type=int, default=1, help="Random erase count (default: 1)"
    )

    # Data parameters
    subparsers.add_argument(
        "--data-path", default="./datasets/", type=str, help="dataset path"
    )
    subparsers.add_argument("--dataset", default="CDDB", type=str, help="dataset name")
    subparsers.add_argument("--shuffle", default=False, help="shuffle the data order")
    subparsers.add_argument(
        "--output_dir",
        default="./output/",
        help="path where to save, empty for no saving",
    )
    subparsers.add_argument(
        "--device", default="cuda", help="device to use for training / testing"
    )
    subparsers.add_argument("--seed", default=42, type=int)
    subparsers.add_argument(
        "--eval", action="store_true", help="Perform evaluation only"
    )
    subparsers.add_argument("--num_workers", default=4, type=int)
    subparsers.add_argument(
        "--pin-mem",
        action="store_true",
        help="Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.",
    )
    subparsers.add_argument(
        "--no-pin-mem", action="store_false", dest="pin_mem", help=""
    )
    subparsers.set_defaults(pin_mem=True)

    # distributed training parameters
    subparsers.add_argument(
        "--world_size", default=1, type=int, help="number of distributed processes"
    )
    subparsers.add_argument(
        "--dist_url", default="env://", help="url used to set up distributed training"
    )

    # Continual learning parameters
    subparsers.add_argument(
        "--num_tasks", default=10, type=int, help="number of sequential tasks"
    )
    subparsers.add_argument(
        "--train_mask",
        default=True,
        type=bool,
        help="if using the class mask at training",
    )
    subparsers.add_argument(
        "--task_inc", default=False, type=bool, help="if doing task incremental"
    )

    # Adapter-Prefix Parameters
    subparsers.add_argument(
        "--has-atten-adapter",
        action="store_true",
        help="add adapter after attention",
    )
    subparsers.add_argument(
        "--has-output-adapter",
        action="store_true",
        help="add adapter after attention",
    )
    subparsers.add_argument(
        "--reduction-factor",
        default=10,
        type=int,
        help="Reduction Factor for Adapter",
    )
    subparsers.add_argument(
        "--has-prefix",
        action="store_true",
        help="add prefix in attention",
    )
    subparsers.add_argument(
        "--prefix-length",
        default=10,
        type=int,
        help="Number of prefix to add to both key AND value",
    )
    subparsers.add_argument(
        "--bottleneck",
        default=200,
        type=int,
        help="Number of prefix to add to both key AND value",
    )
    subparsers.add_argument(
        "--leave-out",
        default=[],
        nargs="*",
        type=list,
        help="ViT Layer Numbers where u don't want adapter to be added",
    )

    # Hypernetwork Parameters
    subparsers.add_argument(
        "--l-embeddings",
        default=64,
        type=int,
        help="Layer Embedding Dimension",
    )
    subparsers.add_argument(
        "--t-embeddings",
        default=64,
        type=int,
        help="Task Embedding Dimension",
    )
    subparsers.add_argument(
        "--bottleneck-factor",
        default=4,
        type=int,
        help="HNet Bottleneck Dimension reduction factor",
    )
    subparsers.add_argument(
        "--h-reg",
        default=0.0,
        type=float,
        help="Regularization Constant for L2 regularization on Hypernetwork Output",
    )

    # Misc parameters
    subparsers.add_argument(
        "--print_freq", type=int, default=10, help="The frequency of printing"
    )
    subparsers.add_argument(
        "--model-store-path",
        default="./test_models/",
        help="path where to save, empty for no saving",
    )
    subparsers.add_argument(
        "--validaton",
        action="store_true",
        help="Validation Split",
    )
