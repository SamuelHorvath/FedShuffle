import torch
from torch import nn
from torch.nn import DataParallel
import time
from collections import OrderedDict
import numpy as np
from copy import deepcopy

from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR

import models
from optimizers.sgd_fl import FLSGD
from utils.logger import Logger
from .utils import init_metrics_meter, neg_perplexity_from_loss, log_epoch_info
from data_funcs.data_loader import get_num_classes
from .fl_funcs import update_train_dicts
from utils.checkpointing import load_checkpoint

CIFAR_MODELS = {
    'resnet': [18, 34, 50, 101, 152],
    'vgg': [11, 13, 16, 19],
    'wideresnet': [282, 284, 288]
}

CLIP_RNN_GRAD = 5


def get_training_elements(model_name, dataset, resume_from, load_best, gpu):
    # Define the model
    model, current_round = initialise_model(
        model_name, dataset, resume_from, load_best, use_pretrained=False)

    model = model_to_device(model, gpu)

    # TODO: we might consider other losses passed through args
    criterion = nn.CrossEntropyLoss()

    return model, criterion, current_round


def initialise_model(model_name, dataset, resume_from=None, load_best=None,
                     use_pretrained=False):

    model_prefix = [prefix for prefix in CIFAR_MODELS
                    if model_name.startswith(prefix)]
    if len(model_prefix) == 1 and dataset.startswith('cifar') and \
            int(model_name.split(model_prefix[0])[1]) \
            in CIFAR_MODELS[model_prefix[0]]:
        Logger.get().debug("Loading cifar version of model")
        model_name = model_name + '_cifar'

    model = getattr(models, model_name)(pretrained=use_pretrained,
                                        num_classes=get_num_classes(dataset))

    current_round = 0
    if resume_from:
        model, current_round = load_checkpoint(resume_from, load_best)

    return model, current_round


def model_to_device(model, device):
    if type(device) == list:  # if to allocate on more than one GPU
        model = model.to(device[0])
        model = DataParallel(model, device_ids=device)
    else:
        model = model.to(device)
    return model


def set_model_weights(model, weights, strict=True):
    """
    Sets the weight models in-place.
    To be used to integrate new updated weights.
    :param model: The model to be updated
    :param weights: (fl.common.Weights)
    List of np ndarrays representing the model weights
    :param strict: To require 1-to-1 parameter to weights association.s
    """
    state_dict = OrderedDict(
        {
            k: torch.Tensor(np.atleast_1d(v))
            for k, v in zip(model.state_dict().keys(), weights)
        }
    )
    model.load_state_dict(state_dict, strict=strict)


def get_model_weights(model):
    """Get model weights as a list of NumPy ndarrays."""

    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def get_lr_scheduler(optimiser, total_epochs, method='static'):
    """
    Implement learning rate scheduler.
    :param optimiser: A reference to the optimiser being used
    :param total_epochs: The total number of epochs (from the args)
    :param method: The strategy to adjust the learning rate
    (multistep, cosine or static)
    :returns: scheduler on current step/epoch/policy
    """
    if method == 'cosine':
        return CosineAnnealingLR(optimiser, total_epochs)
    elif method == 'static':
        return MultiStepLR(optimiser, [total_epochs + 1])
    if method == 'cifar_1':  # VGGs + ResNets
        return MultiStepLR(optimiser, [int(0.5 * total_epochs),
                           int(0.75 * total_epochs)], gamma=0.1)
    if method == 'cifar_2':  # WideResNets
        return MultiStepLR(optimiser, [int(0.3 * total_epochs),
                           int(0.6 * total_epochs), int(0.8 * total_epochs)],
                           gamma=0.2)
    raise ValueError(f"{method} is not defined as scheduler name.")


def run_one_communication_round(
        algo, sum_one, model, trainsets, criterion, local_optimiser,
        global_optimiser, device, round, run_local_iters,
        sampled_clients, sampled_steps, prob_clients,
        is_rnn=False, unbiased=False, clip_grad=False,
        batch_size=32, num_workers=0, print_freq=10):
    metrics_meter = None
    # fed_dataset = train_loader.dataset
    model_dict_original = deepcopy(model.state_dict())
    optimiser_dict_original = deepcopy(local_optimiser.state_dict())
    sampled_clients_round = sampled_clients[round]
    sampled_steps_round = sampled_steps[round]

    Logger.get().info(f"Sampled FL clients: {len(sampled_clients_round)}")
    # Logger.get().info(f"Sampled FL steps: {sampled_steps_round}")

    clients_num_data = list()
    for client_id in sampled_clients_round:
        # fed_dataset.set_client(client_id)
        # clients_num_data.append(len(fed_dataset))
        clients_num_data.append(
            len(trainsets[client_id]))
    clients_num_data = np.array(clients_num_data)

    train_loaders = [
        torch.utils.data.DataLoader(
            trainsets[i],
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True,
            persistent_workers=num_workers > 0
        ) for i in sampled_clients_round
    ]
    # Step size/ update scaling for FedNova/ FedShuffle
    local_iters_s = np.zeros(len(sampled_clients_round), dtype=int)
    for i, client_id in enumerate(sampled_clients_round):
        # fed_dataset.set_client(client_id)
        # local_iters = sampled_epochs_round[i] * len(train_loader)
        if run_local_iters:
            local_iters_s[i] = sampled_steps_round[i]
        else:
            local_epochs = sampled_steps_round[i]
            local_iters = local_epochs * len(train_loaders[i])
            local_iters_s[i] = int(local_iters)
    # extra step size scaling by min epochs and dataset size is applied
    # so FedShuffle step sizes are easier to compare to FedAvg (FedNova)
    # min_epochs = np.min([np.min(steps_i) for steps_i in sampled_steps
    #                      if len(steps_i) > 0])
    max_epochs = np.max([np.max(steps_i) for steps_i in sampled_steps
                         if len(steps_i) > 0])

    if run_local_iters:
        # min_local_iters = min_epochs
        max_local_iters = max_epochs
    else:
        # min_steps_per_client = np.min([len(tl) for tl in train_loaders])
        # min_local_iters = min_steps_per_client * min_epochs
        max_steps_per_client = np.max([np.ceil(len(d) / batch_size)
                                       for d in trainsets])
        max_local_iters = max_steps_per_client * max_epochs
    # print('Max epochs:', max_epochs)
    # print('Max/Min ratio:', max_local_iters / min_local_iters)

    state_dicts = list()
    for i, client_id in enumerate(sampled_clients_round):
        Logger.get().info(
            f'Running local epoch for client: {client_id},'
            f' [{i + 1}/{len(sampled_clients_round)}]')
        # fed_dataset.set_client(client_id)
        train_loader = train_loaders[i]
        # beginning of the round
        model.load_state_dict(model_dict_original)
        local_optimiser.load_state_dict(optimiser_dict_original)
        local_optimiser.start_local_run()

        # fedshuffle scales stepsize to balance out progress
        if algo == 'fedshuffle':
            # lr_scaling = local_iters_s[i] / min_local_iters
            # assert lr_scaling >= 1
            lr_scaling = local_iters_s[i] / max_local_iters
            assert lr_scaling <= 1
            for g in local_optimiser.param_groups:
                g['lr'] = g['lr'] / lr_scaling

        metrics_meter, total_loc_iters = train_model(
            local_iters_s[i], model, train_loader, criterion, local_optimiser,
            device, round, is_rnn, print_freq, unbiased, clip_grad)
        assert local_optimiser.local_steps == total_loc_iters

        # fedshuffle scale back step size
        if algo == 'fedshuffle':
            for g in local_optimiser.param_groups:
                # lr_scaling = local_iters_s[i] / min_local_iters
                lr_scaling = local_iters_s[i] / max_local_iters
                g['lr'] = g['lr'] * lr_scaling

        # save current local model
        local_model_dict = deepcopy(model.state_dict())
        local_optimiser_dict = deepcopy(local_optimiser.state_dict())
        state_dicts.append(
            {'model': local_model_dict, 'optimiser': local_optimiser_dict}
        )
    # Aggregation
    if len(sampled_clients_round) >= 1:
        # weights based on the number of local data
        # fed_dataset.set_client(None)  # full dataset
        # weights = torch.tensor(
        #     clients_num_data / len(fed_dataset)).to(device)
        if len(trainsets) == 1:
            num_samples = len(trainsets[0])
        else:
            num_samples = len(trainsets[0].fl_dataset)
        # sum([len(tl.dataset) for tl in train_loaders])
        weights_data = torch.tensor(
            clients_num_data / num_samples).to(device)

        # fednova scales updates to balance out progress
        if algo == 'fednova':
            scaling = torch.from_numpy(local_iters_s).float().to(device)
            weights_actual = weights_data / scaling
            # rescale to preserve sum of weights
            weights_actual *= sum(weights_data) / sum(weights_actual)
        else:
            weights_actual = weights_data

        # weights adjusted by probability
        probs = torch.from_numpy(
            prob_clients[sampled_clients_round]).float().to(device)
        weights_agg = weights_actual / probs

        # normalize weights to sum one
        if sum_one:
            weights_agg /= sum(weights_agg)

        # SERVER MOMENTUM  UPDATE
        if local_optimiser.server_momentum > 0:
            weights_mom = weights_data / probs
            # normalize weights to sum one
            if sum_one:
                weights_mom /= sum(weights_mom)
            if algo in ['fedavg', 'fednova']:
                weights_mom /= torch.from_numpy(
                    local_iters_s).float().to(device)
            else:  # fedshuffle
                # weights_mom /= min_local_iters
                weights_mom /= max_local_iters
            # aggregate local models to compute momentum update
            model_dict_mom, optimiser_dict_mom = update_train_dicts(
                state_dicts, weights_mom)
            # load weighted average of model and criterion state dictionaries
            local_optimiser.load_state_dict(optimiser_dict_mom)
            model.load_state_dict(model_dict_mom)
            with torch.no_grad():
                for name, param in model.named_parameters():
                    # compute mom update with learning rate
                    param.mom_w_lr = model_dict_original[name] * \
                        sum(weights_mom) - param.data
            local_optimiser.assign_new_momentum()

        # GLOBAL MODEL UPDATE
        # aggregate local models to global model
        model_dict_new, optimiser_dict_new = update_train_dicts(
            state_dicts, weights_agg)
        # load weighted average of model and criterion state dictionaries
        model.load_state_dict(model_dict_new)
        # if positive, the local optimizer wes already updated
        if local_optimiser.server_momentum == 0:
            local_optimiser.load_state_dict(optimiser_dict_new)
        # assign grad from aggregation as model gradient
        global_optimiser.zero_grad(set_to_none=True)
        with torch.no_grad():
            for name, param in model.named_parameters():
                # set data to original value and adjust gradient as difference
                diff = model_dict_original[name] * sum(weights_agg) -  \
                    param.data
                param.data = torch.clone(model_dict_original[name]).detach()
                param.grad = torch.clone(diff).detach()
        global_optimiser.step()
        global_optimiser.zero_grad(set_to_none=True)
    return metrics_meter


def train_model(iters_to_take, model, train_loader, criterion, local_optimiser,
                device, round, is_rnn=False, print_freq=10,
                unbiased=False, clip_grad=False,):
    metrics_meter = init_metrics_meter(round)
    model.train()

    n = len(train_loader)
    iters = 0
    epochs = np.ceil(iters_to_take / n).astype(int)
    if not unbiased:
        for _ in range(epochs):
            for data, label in train_loader:
                iters += 1
                process_batch(
                    model, criterion, data, label, device, local_optimiser,
                    is_rnn, clip_grad, metrics_meter,
                    print_freq, iters, iters_to_take)
                if iters >= iters_to_take:
                    return metrics_meter, iters_to_take
    else:
        for _ in range(iters_to_take):
            data, label = next(iter(train_loader))
            iters += 1
            process_batch(
                model, criterion, data, label, device, local_optimiser,
                is_rnn, clip_grad, metrics_meter, print_freq,
                iters, iters_to_take)
    return metrics_meter, iters_to_take


def process_batch(model, criterion, data, label, device, local_optimiser,
                  is_rnn, clip_grad, metrics_meter,
                  print_freq, i, total_iters):
    start_ts = time.time()
    batch_size = data.shape[0]
    data, label = data.to(device), label.to(device)
    dataload_duration = time.time() - start_ts

    inference_duration = 0.
    backprop_duration = 0.

    local_optimiser.zero_grad(set_to_none=True)

    input, label = get_train_inputs(
        data, label, model, batch_size, device, is_rnn)
    inference_duration, backprop_duration, _, _ = \
        forward_backward(
            model, criterion, input, label, inference_duration,
            backprop_duration, batch_size, metrics_meter, is_rnn)
    if is_rnn or clip_grad:
        nn.utils.clip_grad_norm_(model.parameters(), CLIP_RNN_GRAD)
    local_optimiser.step()
    local_optimiser.zero_grad(set_to_none=True)
    if i % print_freq == 0:
        log_epoch_info(
            Logger, i, total_iters, metrics_meter, dataload_duration,
            inference_duration, backprop_duration, train=True)


def get_train_inputs(data, label, model, batch_size, device, is_rnn):
    if not is_rnn:
        input = (data,)
    else:
        hidden = model.init_hidden(batch_size, device)
        input = (data, hidden)
        label = label.reshape(-1)
    return input, label


def evaluate_model(model, val_loader, criterion, device, round,
                   print_freq=10, metric_to_optim='top_1', is_rnn=False):
    metrics_meter = init_metrics_meter(round)
    if is_rnn:
        hidden = model.init_hidden(val_loader.batch_size, device)

    model.eval()
    with torch.no_grad():
        for i, (data, label) in enumerate(val_loader):
            batch_size = data.shape[0]
            start_ts = time.time()

            data = data.to(device)
            label = label.to(device)
            if is_rnn:
                label = label.reshape(-1)

            dataload_duration = time.time() - start_ts
            if is_rnn:
                output, hidden = model(data, hidden)
            else:
                output = model(data)
            inference_duration = time.time() - (start_ts + dataload_duration)

            loss = compute_loss(model, criterion, output, label)
            update_metrics(metrics_meter, loss, batch_size, output, label)
            if i % print_freq == 0:
                log_epoch_info(
                    Logger, i, len(val_loader), metrics_meter,
                    dataload_duration, inference_duration,
                    backprop_duration=0., train=False)

    # Metric for avg/single model(s)
    Logger.get().info(f'{metric_to_optim}:'
                      f' {metrics_meter[metric_to_optim].get_avg()}')
    return metrics_meter


def accuracy(output, label, topk=(1,)):
    """
    Extract the accuracy of the model.
    :param output: The output of the model
    :param label: The correct target label
    :param topk: Which accuracies to return (e.g. top1, top5)
    :return: The accuracies requested
    """
    maxk = max(topk)
    if maxk > output.shape[-1]:
        maxk = output.shape[-1]
        topk = (np.min([maxk, k]) for k in topk)
    batch_size = label.size(0)

    if len(output.size()) == 1:
        _, pred = output.topk(maxk, 0, True, True)
    else:
        _, pred = output.topk(maxk, 1, True, True)
    if pred.size(0) != 1:
        pred = pred.t()

    if pred.size() == (1,):
        correct = pred.eq(label)
    else:
        correct = pred.eq(label.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size).item())
    return res


def forward_backward(model, criterion, input, label, inference_duration,
                     backprop_duration, batch_size, metrics_meter, is_rnn):
    start_ts = time.time()
    outputs = model(*input)

    if not is_rnn:
        hidden = None
        output = outputs
    else:
        output, hidden = outputs

    single_inference = time.time() - start_ts
    inference_duration += single_inference

    loss = compute_loss(model, criterion, output, label)
    loss.backward()
    backprop_duration += time.time() - (start_ts + single_inference)

    update_metrics(metrics_meter, loss, batch_size, output, label)
    return inference_duration, backprop_duration, output, hidden


def compute_loss(model, criterion, output, label):

    if type(output) == list and len(output) > 1:
        if type(model).__name__.lower() == 'inception3':
            loss = criterion(
                output[0], label) + 0.4 * criterion(output[1], label)
        else:
            loss = sum([criterion(out, label) for out in output])
    else:
        loss = criterion(output, label)
    return loss


def update_metrics(metrics_meter, loss, batch_size, output, label):
    metrics_meter['loss'].update(loss.item(), batch_size)
    metrics_meter['neq_perplexity'].update(
        neg_perplexity_from_loss(loss.item()), batch_size)
    acc = accuracy(output, label, (1, 5))
    metrics_meter['top_1_acc'].update(acc[0], batch_size)
    metrics_meter['top_5_acc'].update(acc[1], batch_size)


def get_optimiser(params_to_update, optimiser_name, lr, momentum,
                  weight_decay, server_momentum=0):
    if optimiser_name == 'sgd':
        optimiser = FLSGD(
            params_to_update, lr, momentum=momentum,
            weight_decay=weight_decay, server_momentum=server_momentum)
    else:
        raise ValueError("optimiser not supported")

    return optimiser
