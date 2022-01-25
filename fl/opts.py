import argparse
import time
from datetime import datetime
import os
import torch


def parse_args(args):
    parser = initialise_arg_parser(args, 'FL Simulated Experiments.')

    parser.add_argument(
        '--fl-algorithm',
        type=str,
        choices=['fedavg', 'fednova', 'fedshuffle'],
        default='fedavg',
        help='FL algorithm used in training (default: fedavg)'
    )

    parser.add_argument(
        "--sum-one",
        default=False,
        action='store_true',
        help="Whether to force aggregation to have sum one"
    )

    parser.add_argument(
        "--unbiased-grad",
        default=False,
        action='store_true',
        help="Whether to use unbiased gradient or data shuffling"
    )

    parser.add_argument(
        "--clip-grad",
        default=False,
        action='store_true',
        help="Whether to use gradient cipping"
    )

    # SERVER OPTIMIZATION PARAMS
    parser.add_argument(
        "--rounds",
        type=int,
        default=1,
        help="Number of communication rounds",
    )
    parser.add_argument(
        "--client-distribution",
        type=str,
        default='uniform_1',
        help="Client Selection Distribution,"
             "Second part stands for expected number of sampled clients.",
    )
    parser.add_argument(
        '--global-lr',
        type=float,
        default=1,
        help='Global initial local learning rate (default: 1)'
    )
    parser.add_argument(
        '--global-lr-type',
        type=str,
        choices=['cosine', 'cifar_1', 'cifar_2', 'static'],
        default='static',
        help='Global learning rate strategy (default: static)'
    )
    parser.add_argument(
        "--global-optimiser",
        type=str,
        choices=['sgd'],
        default='sgd',
        help='Global optimiser to use (default: SGD)'
    )
    parser.add_argument(
        '--global-momentum',
        type=float,
        default=0.,
        help='Global momentum (default: 0.)'
    )
    parser.add_argument(
        '--server-momentum',
        type=float,
        default=0.,
        help='Server momentum (default: 0.)'
    )
    parser.add_argument(
        '--global-weight-decay',
        type=float,
        default=0.,
        help='Global weight decay (default: 0.)'
    )

    # LOCAL OPTIMISATION PARAMETERS
    parser.add_argument(
        "--run-local-steps",
        action="store_true",
        default=False,
        help="Run local epochs or local iterations, "
             "if 'True',"
             " then each worker runs '--number-of-local-iters' local steps,"
             " else each worker runs '--number-of-local-iters' local epochs."
    )
    parser.add_argument(
        "-li", "--local-iters-sampler",
        type=str,
        default='static_1',
        help="Sampler for the number of local steps/epochs to run."
    )
    parser.add_argument(
        "-b", "--batch-size",
        type=int,
        default=32,
        help="Static batch size for local runs"
    )
    parser.add_argument(
        '--local-lr',
        type=float,
        default=0.1,
        help='initial local learning rate (default: 0.1)'
    )
    parser.add_argument(
        '--local-lr-type',
        type=str,
        choices=['cosine', 'cifar_1', 'cifar_2', 'static'],
        default='cifar_1',
        help='Local learning rate strategy (default: cifar_1)'
    )
    parser.add_argument(
        "--local-optimiser",
        type=str,
        choices=['sgd'],
        default='sgd',
        help='Local optimiser to use (default: SGD)'
    )
    parser.add_argument(
        '--local-momentum',
        type=float,
        default=0.,
        help='Momentum (default: 0.)'
    )
    parser.add_argument(
        '--local-weight-decay',
        type=float,
        default=0.,
        help='Local weight decay (default: 0.)'
    )

    # MODEL and DATA PARAMETERS
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=[
            "cifar10", "cifar100", "emnist", "full_shakespeare",
            "cifar10_fl", "cifar100_fl", "shakespeare", "femnist",
            "synth_0.0_0.0", "synth_0.5_0.5", "synth_1.0_1.0",
            "mushrooms", "ijcnn1", "w8a", "a9a", "phishing", 'full_mushrooms',
            'full_ijcnn1', 'full_w8a', 'full_a9a', 'full_phishing'],
        help="Define which dataset to load"
    )
    parser.add_argument(
        "--metric",
        type=str,
        default='top_1_acc',
        choices=["top_1_acc", "top_5_acc", "neg_perplexity"],
        help="Define which metric to optimize."
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Define which model to load"
    )

    # SETUP ARGUMENTS
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default='../check_points',
        help="Directory to persist run meta data_preprocess,"
             " e.g. best/last models."
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        help="Resume from checkpoint."
    )
    parser.add_argument(
        "--load-best",
        default=False,
        action='store_true',
        help="Load best from checkpoint"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="../data/",
        help="Base root directory for the dataset."
    )

    parser.add_argument(
        "--gpu",
        type=str,
        default="0",
        help="Define on which GPU to run the model"
             " (comma-separated for multiple). If -1, use CPU."
    )
    parser.add_argument(
        "-n", "--num-workers",
        type=int,
        default=2,
        help="Num workers for dataset loading"
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        default=False,
        help="Run deterministically for reproducibility."
    )
    parser.add_argument(
        "--manual-seed",
        type=int,
        default=123,
        help="Random seed to use."
    )
    parser.add_argument(
        "--eval-every",
        type=int,
        default=5,
        help="How often to do validation."
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=str(time.time()),
        help="Identifier for the current job"
    )
    parser.add_argument(
        "--loglevel",
        type=str,
        choices=["DEBUG", "INFO", "WARN", "ERROR", "CRITICAL"],
        default="INFO"
    )
    now = datetime.now()
    now = now.strftime("%Y%m%d%H%M%S")
    os.makedirs("../logs/", exist_ok=True)
    parser.add_argument(
        "--logfile",
        type=str,
        default=f"../logs/log_{now}.txt"
    )

    # Evaluation mode, do not run training
    parser.add_argument("--evaluate", action='store_true', default=False,
                        help="Evaluation or Training mode")

    # run setup for cluster
    parser.add_argument("--cluster", action='store_true', default=False,
                        help="Run on cluster.")
    parser.add_argument(
        "--cluster-partition",
        type=str,
        choices=["learnlab", "prioritylab", "devlab"],
        default="learnlab"
    )
    parser.add_argument(
        "--cluster-time-limit",
        type=int,
        default=120,
        help="Maximum run time for the job in minutes."
    )
    parser.add_argument(
        "--slurm-folder",
        type=str,
        default=f'/checkpoint/{os.getenv("USER", "..")}/fedshuffle/slurm'
    )

    args = parser.parse_args()
    transform_gpu_args(args)

    return args


def get_available_gpus():
    """
    Get list of available gpus in the system
    """
    gpus = []
    for i in range(torch.cuda.device_count()):
        gpus.append(torch.cuda.get_device_properties(i))


def initialise_arg_parser(args, description):
    parser = argparse.ArgumentParser(args, description=description)
    return parser


def transform_gpu_args(args):
    if args.gpu == "-1":
        args.gpu = "cpu"
    else:
        gpu_str_arg = args.gpu.split(',')
        if len(gpu_str_arg) > 1:
            args.gpu = sorted([int(card) for card in gpu_str_arg])
        else:
            args.gpu = f"cuda:{args.gpu}"
