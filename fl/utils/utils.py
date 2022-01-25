import numpy as np
import os
import json
import glob


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else None

    def get_val(self):
        return self.val

    def get_avg(self):
        return self.avg


def init_metrics_meter(round=None):
    if round is not None:
        metrics_meter = {
            'round': round,
            'loss': AverageMeter(),
            'top_1_acc': AverageMeter(),
            'top_5_acc': AverageMeter(),
            'neq_perplexity': AverageMeter(),
        }
    else:
        metrics_meter = {
            'train_round': [], 'train_loss': [], 'train_top_1_acc': [],
            'train_top_5_acc': [], 'train_neq_perplexity': [],
            'test_round': [], 'test_loss': [], 'test_top_1_acc': [],
            'test_top_5_acc': [], 'test_neq_perplexity': [],
        }
    return metrics_meter


def neg_perplexity_from_loss(loss):
    return - np.exp(loss)


def get_model_str_from_obj(model):
    return str(list(model.modules())[0]).split("\n")[0][:-1]


def create_metrics_dict(metrics):
    metrics_dict = {'round': metrics['round']}
    for k in metrics:
        if k == 'round':
            continue
        metrics_dict[k] = metrics[k].get_avg()
    return metrics_dict


def create_model_dir(args, lr=True):
    model_dataset = '_'.join([args.model, args.dataset])
    run_id = f'id={args.run_id}'
    sum_one_suffix = '_so' if args.sum_one else ''
    unbiased_suffix = '_ug' if args.unbiased_grad else ''
    clip_grad_suffix = '_cg' if args.clip_grad else ''

    model_dir = os.path.join(
        args.checkpoint_dir, model_dataset, run_id,
        args.fl_algorithm + unbiased_suffix + sum_one_suffix + 
        clip_grad_suffix)
    if lr:
        run_hp = os.path.join(
            f"lr=l_{str(args.local_lr)}_g_{str(args.global_lr)}",
            f"seed={str(args.manual_seed)}")
        model_dir = os.path.join(model_dir, run_hp)

    return model_dir


def log_epoch_info(Logger, i, total_iters, metrics_meter, dataload_duration,
                   inference_duration, backprop_duration, train=True):
    mode_str = 'Train' if train else 'Test'
    Logger.get().info("{mode_str} [{round}][{current_batch}/{total_batches}]\t"
                      "DataLoad time {dataload_duration:.3f}\t"
                      "F/W time {inference_duration:.3f}\t"
                      "B/W time {backprop_duration:.3f}\t"
                      "Loss {loss:.4f}\t"
                      "Prec@1 {prec1:.3f}\t"
                      "Prec@5 {prec5:.3f}\t".format(
                        mode_str=mode_str,
                        round=metrics_meter['round'],
                        current_batch=i,
                        total_batches=total_iters,
                        dataload_duration=dataload_duration,
                        inference_duration=inference_duration,
                        backprop_duration=backprop_duration,
                        loss=metrics_meter['loss'].get_avg(),
                        prec1=metrics_meter['top_1_acc'].get_avg(),
                        prec5=metrics_meter['top_5_acc'].get_avg()))


def metric_to_dict(metrics_meter, round, preffix='', all_prefix=True):
    round_preffix = preffix + '_round' if all_prefix else 'round'
    out = {
        round_preffix: round,
        preffix + '_loss': metrics_meter['loss'].get_avg(),
        preffix + '_top_1_acc': metrics_meter['top_1_acc'].get_avg(),
        preffix + '_top_5_acc': metrics_meter['top_5_acc'].get_avg(),
    }
    return out


def extend_metrics_dict(full_metrics, last_metrics):
    for k in last_metrics:
        if last_metrics[k] is not None:
            full_metrics[k].append(last_metrics[k])


def get_key(train=True):
    return 'train_' if train else 'test_'


def get_best_lr_and_metric(args, last=True):
    best_arg, best_lookup = (np.nanargmin, np.nanmin) \
        if args.metric in ['loss'] else (np.nanargmax, np.nanmax)
    key = get_key(args.train_metric)
    model_dir_no_lr = create_model_dir(args, lr=False)
    lr_dirs = [lr_dir for lr_dir in os.listdir(model_dir_no_lr)
               if os.path.isdir(os.path.join(model_dir_no_lr, lr_dir))
               and not lr_dir.startswith('.')]
    runs_metric = list()
    for lr_dir in lr_dirs:
        # /*/ for different seeds
        lr_metric_dirs = glob.glob(
            model_dir_no_lr + '/' + lr_dir + '/*/full_metrics.json')
        if len(lr_metric_dirs) == 0:
            runs_metric.append(np.nan)
        else:
            lr_metric = list()
            for lr_metric_dir in lr_metric_dirs:
                with open(lr_metric_dir) as json_file:
                    metrics = json.load(json_file)
                metric_values = metrics[key + args.metric]
                metric = metric_values[-1] if last else \
                    best_lookup(metric_values)
                lr_metric.append(metric)
            runs_metric.append(np.mean(lr_metric))

    i_best_lr = best_arg(runs_metric)
    best_metric = runs_metric[i_best_lr]
    best_lr = lr_dirs[i_best_lr]
    return best_lr, best_metric, lr_dirs


def get_best_runs(args_exp, last=True):
    model_dir_no_lr = create_model_dir(args_exp, lr=False)
    best_lr, _, _ = get_best_lr_and_metric(args_exp, last=last)
    model_dir_lr = os.path.join(model_dir_no_lr, best_lr)
    json_dir = 'full_metrics.json'
    metric_dirs = glob.glob(model_dir_lr + '/*/' + json_dir)

    print(f'Best_lr: {best_lr}')
    with open(metric_dirs[0]) as json_file:
        metric = json.load(json_file)
    runs = [metric]

    for metric_dir in metric_dirs[1:]:
        with open(metric_dir) as json_file:
            metric = json.load(json_file)
        # ignores failed runs
        if not np.isnan(metric[get_key(train=True) + 'loss']).any():
            runs.append(metric)

    return runs
