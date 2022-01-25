import sys
import os
import json
import torch
import numpy as np
import time
# import wandb
import submitit

from opts import parse_args
from utils.logger import Logger
from data_funcs.data_loader import load_data, get_test_batch_size
from utils.model_funcs import get_training_elements, evaluate_model, \
     get_lr_scheduler, get_optimiser, run_one_communication_round
from utils.checkpointing import save_checkpoint
from utils.utils import create_model_dir, create_metrics_dict, \
    metric_to_dict, init_metrics_meter, extend_metrics_dict
from utils.fl_funcs import get_sampled_clients, get_sampled_local_epochs

from models import RNN_MODELS

# os.environ["WANDB_API_KEY"] = '205924dacff241a772a053e251a37bc15ceb90b4'


def main(args):
    # system setup
    global CUDA_SUPPORT

    Logger.setup_logging(args.loglevel, logfile=args.logfile)
    logger = Logger()

    logger.debug(f"CLI args: {args}")

    if torch.cuda.device_count():
        CUDA_SUPPORT = True
    else:
        logger.warning('CUDA unsupported!!')
        CUDA_SUPPORT = False

    if not CUDA_SUPPORT:
        args.gpu = "cpu"

    if args.deterministic:
        # import torch.backends.cudnn as cudnn
        import os
        import random

        if CUDA_SUPPORT:
            # cudnn.deterministic = args.deterministic
            # cudnn.benchmark = not args.deterministic
            torch.cuda.manual_seed(args.manual_seed)
            torch.cuda.manual_seed_all(args.manual_seed)

        random.seed(args.manual_seed)
        np.random.seed(args.manual_seed)
        os.environ['PYTHONHASHSEED'] = str(args.manual_seed)
        torch.manual_seed(args.manual_seed)

    logger.info(f"Model: {args.model}, Dataset:{args.dataset}")

    # In case of DataParallel for .to() to work
    args.device = args.gpu[0] if type(args.gpu) == list else args.gpu

    # Load data sets
    trainsets, testset = load_data(args.data_path, args.dataset,
                                   load_trainset=True, download=True)

    test_batch_size = get_test_batch_size(args.dataset, args.batch_size)
    testloader = torch.utils.data.DataLoader(testset,
                                             batch_size=test_batch_size,
                                             num_workers=4,
                                             shuffle=False,
                                             persistent_workers=True)

    if not args.evaluate:  # Training mode
        # trainloader = torch.utils.data.DataLoader(
        #     trainset,
        #     batch_size=args.batch_size,
        #     num_workers=args.num_workers,
        #     shuffle=True,
        #     **loader_kwargs
        # )

        init_and_train_model(args, trainsets, testloader)

    else:  # Evaluation mode
        model, criterion, round = get_training_elements(
            args.model_name, args.dataset, args.resume_from,
            args.load_best, args.gpu)

        metrics = evaluate_model(
            model, testloader, criterion, args.device, round,
            print_freq=10, metric_to_optim=args.metric,
            is_rnn=args.model in RNN_MODELS)

        metrics_dict = create_metrics_dict(metrics)
        logger.info(f'Validation metrics: {metrics_dict}')


def init_and_train_model(args, trainsets, testloader):
    full_metrics = init_metrics_meter()
    model_dir = create_model_dir(args)
    # don't train if setup already exists
    if os.path.isdir(model_dir):
        Logger.get().info(f"{model_dir} already exists.")
        Logger.get().info("Skipping this setup.")
        return
    # create model directory
    os.makedirs(model_dir, exist_ok=True)
    # init wandb tracking
    # wandb.init(
    #     project="fedshuffle", entity="samuelhovath", config=vars(args),
    #     name=str(create_model_dir(args, lr=False)), reinit=True)
    # save used args as json to experiment directory
    with open(os.path.join(create_model_dir(args), 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)

    is_rnn = args.model in RNN_MODELS
    model, criterion, current_round = get_training_elements(
        args.model, args.dataset, args.resume_from, args.load_best, args.gpu)

    local_optimiser = get_optimiser(
        model.parameters(), args.local_optimiser, args.local_lr,
        args.local_momentum, args.local_weight_decay, args.server_momentum)

    local_scheduler = get_lr_scheduler(
        local_optimiser, args.rounds, args.local_lr_type)

    global_optimiser = get_optimiser(
        model.parameters(), args.global_optimiser, args.global_lr,
        args.global_momentum, args.global_weight_decay)

    global_scheduler = get_lr_scheduler(
        global_optimiser, args.rounds, args.global_lr_type)

    metric_to_optim = args.metric
    best_metric = -np.inf
    train_time_meter = 0

    # trainloader.dataset.num_clients -> len(trainloaders)
    sampled_clients, prob_clients = get_sampled_clients(
        args.client_distribution, trainsets,
        args.rounds, args.manual_seed)

    sampled_steps = get_sampled_local_epochs(
        args.local_iters_sampler, sampled_clients,
        args.rounds, args.manual_seed)

    for i in range(args.rounds):
        start = time.time()
        metrics_meter = run_one_communication_round(
            args.fl_algorithm, args.sum_one, model, trainsets,
            criterion, local_optimiser, global_optimiser, args.device,
            current_round, args.run_local_steps, sampled_clients,
            sampled_steps, prob_clients, is_rnn, args.unbiased_grad,
            args.clip_grad, args.batch_size, args.num_workers)
        extend_metrics_dict(
            full_metrics, metric_to_dict(metrics_meter, i+1, 'train'))
        # wandb.log(metric_to_dict(metrics_meter, i+1, 'train', False))
        train_time = time.time() - start
        train_time_meter += train_time
        # Track timings across epochs
        Logger.get().debug(f'Epoch train time: {train_time}')

        if i % args.eval_every == 0 or i == (args.rounds - 1):
            metrics = evaluate_model(
                model, testloader, criterion, args.device, current_round,
                print_freq=10, is_rnn=is_rnn, metric_to_optim=metric_to_optim)
            extend_metrics_dict(
                full_metrics, metric_to_dict(metrics, i+1, 'test'))
            # wandb.log(metric_to_dict(metrics, i+1, 'test', False))
            avg_metric = metrics[metric_to_optim].get_avg()
            # Save model checkpoint
            model_filename = (f"{args.model}_{args.run_id}_checkpoint"
                              f"_{current_round:0>2d}.pth.tar")
            is_best = avg_metric > best_metric
            save_checkpoint(model, model_filename, is_best=is_best, args=args,
                            metrics=metrics, metric_to_optim=metric_to_optim)
            if is_best:
                best_metric = avg_metric

            if np.isnan(metrics['loss'].get_avg()):
                Logger.get().info(
                    'NaN loss detected, aborting training procedure.')
                return

        Logger.get().info(
            f'Current lrs: local:{local_scheduler.get_last_lr()},'
            f'global:{global_scheduler.get_last_lr()}')
        local_scheduler.step()
        global_scheduler.step()
        current_round += 1
        torch.cuda.empty_cache()

    Logger.get().debug(
        f'Average epoch train time: {train_time_meter / args.rounds}')
    #  store the run
    with open(os.path.join(
            create_model_dir(args), 'full_metrics.json'), 'w') as f:
        json.dump(full_metrics, f, indent=4)


if __name__ == "__main__":
    args = parse_args(sys.argv)
    if args.cluster:
        # Submit a job to the SLURM queue, batch mode only. Find logs
        # in the conf.slurm_folder.
        gpus_per_node = 1
        nodes = 1

        executor = submitit.AutoExecutor(folder=args.slurm_folder)
        executor.update_parameters(
            name=f"{args.run_id}_{args.dataset}_{args.manual_seed}",
            slurm_partition=args.cluster_partition,
            nodes=nodes,
            tasks_per_node=1,
            slurm_gpus_per_task=1,
            cpus_per_task=24,
            slurm_mem=f"{gpus_per_node * 50}G",
            slurm_time=args.cluster_time_limit,
        )
        with executor.batch():
            job = executor.submit(
                main, args
            )
    else:
        # run locally
        main(args)
        torch.cuda.empty_cache()
        assert torch.cuda.memory_allocated() == 0
