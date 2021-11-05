import pandas as pd
import dgl
from dgllife.utils import ScaffoldSplitter, Meter
import torch.nn as nn
import torch
import numpy as np
from dgllife.utils import EarlyStopping
from torch.optim import Adam, lr_scheduler
from torch.utils.data import DataLoader
import wandb
import os
import json
import copy
from collections import defaultdict
from data_loader import make_timestamp, GraphCancerMolecules, split_subset_data_loader, SelfiesCancerMolecules
from data_loader import get_datasets, split_data, kfold_split_data
from arguments import get_args
from models import get_model
from plot_utils import viz_mols



def to_device(data_dict, device):

    new_data_dict = {}
    for k, v in data_dict.items():
        if isinstance(v, torch.Tensor):
            new_data_dict[k] = v.to(device, non_blocking=True)
        elif isinstance(v, dgl.DGLGraph):
            new_data_dict[k] = v.to(device, non_blocking=True)
        else:
            new_data_dict[k] = v
    return new_data_dict


def run_a_train_epoch(args, epoch, model, data_loader, mut_loss_criterion, carc_loss_criterion, optimizer):
    model.train()
    train_meter_carc = Meter()
    train_meter_mut = Meter()
    losses = []
    mut_losses = []
    carc_losses = []

    if type(data_loader.dataset) == dgl.data.utils.Subset:
        use_carc_prob = data_loader.dataset.dataset.use_carc_prob
    elif type(data_loader.dataset) == GraphCancerMolecules or type(data_loader.dataset) == SelfiesCancerMolecules:
        use_carc_prob = data_loader.dataset.use_carc_prob
    else:
        raise ValueError

    for batch_id, batch_data in enumerate(data_loader):
        batch_data = to_device(batch_data, args["device"])
        logits = model(batch_data)
        # Mask non-existing labels

        # get carcinogenic logits + labels based on the loss function
        if type(carc_loss_criterion) == torch.nn.modules.loss.BCEWithLogitsLoss:
            carc_logits = torch.masked_select(logits[:, 1], batch_data['carc_mask'])

            if use_carc_prob:
                carc_labels = torch.masked_select(batch_data["carc_prob"], batch_data['carc_mask'])
            else:
                carc_labels = torch.masked_select(batch_data["carc_label"], batch_data['carc_mask'])
            carc_logging = torch.masked_select(batch_data["carc_label"], batch_data['carc_mask'])

        elif type(carc_loss_criterion) == torch.nn.modules.loss.MSELoss:
            carc_logits = torch.masked_select(logits[:, 1], batch_data['carc_mask_continuous'])
            carc_labels = torch.masked_select(batch_data["carc_continuous"], batch_data['carc_mask_continuous'])
            carc_logging = carc_labels

        elif type(carc_loss_criterion) == torch.nn.modules.loss.CrossEntropyLoss:
            new_mask = batch_data['carc_mask_continuous'].view(-1, 1).expand(logits[:, 1:6].shape)

            carc_logits = torch.masked_select(logits[:, 1:6], new_mask).view(-1, 5)
            carc_labels = torch.masked_select(batch_data["carc_label_multi"], batch_data['carc_mask_multi'])
            carc_logging = carc_labels

        else:
            raise ValueError

        # Get mutagenic logits
        mut_logits = torch.masked_select(logits[:, 0], batch_data['mut_mask'])
        mut_labels = torch.masked_select(batch_data["mut_label"], batch_data['mut_mask'])

        # In case batch does not contain any carcinogenic labels set loss manually to 0
        if args['use_carc_loss'] and len(carc_logits) > 0:
            carc_loss = carc_loss_criterion(carc_logits, carc_labels).mean()
            train_meter_carc.update(carc_logits.view(-1, 1), carc_logging.view(-1, 1))
        else:
            carc_loss = torch.tensor(0)

        if args['use_mut_loss'] and len(mut_logits) > 0:
            mut_loss = mut_loss_criterion(mut_logits, mut_labels).mean()
            train_meter_mut.update(mut_logits.view(-1, 1), mut_labels.view(-1, 1))
        else:
            mut_loss = torch.tensor(0)

        # Take weighted average of mut_loss and carc loss
        loss = carc_loss * (1 - args['mut_loss_ratio']) + mut_loss * args['mut_loss_ratio']

        # Zero out the optimizer
        optimizer.zero_grad()

        # Backpropagate loss
        loss.backward()

        # Clip loss
        torch.nn.utils.clip_grad_norm_(model.parameters(), args["gradient_clip_norm"])

        # apply changes to weights
        optimizer.step()

        losses.append(loss.item())
        carc_losses.append(carc_loss.item())
        mut_losses.append(mut_loss.item())

        if batch_id + 1 % args['print_every'] == 0:
            print('epoch {:d}/{:d}, batch {:d}/{:d}'.format(
                epoch + 1, args['num_epochs'], batch_id + 1, len(data_loader)))

        if args['use_carc_loss']:
            train_carc_metric, train_carc_metric_name, train_carc_metric2, train_carc_metric_name2 = perform_eval(
                carc_loss_criterion, train_meter_carc)
        else:
            train_carc_metric = np.nan
            train_carc_metric_name = np.nan
            train_carc_metric2 = np.nan
            train_carc_metric_name2 = np.nan

        if args['use_mut_loss']:
            train_mut_metric, train_mut_metric_name, train_mut_metric2, train_mut_metric_name2 = perform_eval(
                mut_loss_criterion, train_meter_mut)
        else:
            train_mut_metric = np.nan
            train_mut_metric_name = np.nan
            train_mut_metric2 = np.nan
            train_mut_metric_name2 = np.nan

        train_loss = np.nanmean(losses)
        train_mut_loss = np.nanmean(mut_losses)
        train_carc_loss = np.nanmean(carc_losses)

        return train_loss, train_carc_loss, train_mut_loss,\
               train_mut_metric, train_mut_metric_name, train_mut_metric2, train_mut_metric_name2, \
               train_carc_metric, train_carc_metric_name, train_carc_metric2, train_carc_metric_name2


def run_an_eval_epoch(args, model, data_loader, mut_loss_criterion, carc_loss_criterion):
    model.eval()
    eval_carc_meter = Meter()
    eval_mut_meter = Meter()
    losses, mut_losses, carc_losses, ys = [], [], [], []

    eval_dataset = defaultdict(list)
    if type(data_loader.dataset) == dgl.data.utils.Subset:
        use_carc_prob = data_loader.dataset.dataset.use_carc_prob
    elif type(data_loader.dataset) == GraphCancerMolecules or type(data_loader.dataset) == SelfiesCancerMolecules:
        use_carc_prob = data_loader.dataset.use_carc_prob
    else:
        raise ValueError

    with torch.no_grad():
        for batch_id, batch_data in enumerate(data_loader):
            batch_data = to_device(batch_data, args["device"])
            logits = model(batch_data)

            # save eval_dataset
            mut_logits = logits[:,0]
            carc_logits = logits[:,1:6]
            eval_dataset['smiles'].extend(batch_data['smiles'])
            eval_dataset['carc_continuous'].extend(batch_data['carc_continuous'].cpu().numpy())
            eval_dataset['carc_label'].extend(batch_data['carc_label'].cpu().numpy())
            eval_dataset['carc_mask'].extend(batch_data['carc_mask'].cpu().numpy())
            eval_dataset['carc_logits'].extend(carc_logits.cpu().numpy())
            eval_dataset['mut_label'].extend(batch_data['mut_label'].cpu().numpy())
            eval_dataset['mut_mask'].extend(batch_data['mut_mask'].cpu().numpy())
            eval_dataset['mut_logits'].extend(mut_logits.cpu().numpy())

            # get carc labels
            if type(carc_loss_criterion) == torch.nn.modules.loss.BCEWithLogitsLoss:
                carc_mask = batch_data['carc_mask']
                if use_carc_prob:
                    carc_labels = torch.masked_select(batch_data["carc_prob"], carc_mask)
                else:
                    carc_labels = torch.masked_select(batch_data["carc_label"], carc_mask)
                carc_logging = torch.masked_select(batch_data["carc_label"], carc_mask)

            elif type(carc_loss_criterion) == torch.nn.modules.loss.MSELoss:
                carc_mask = batch_data['carc_mask_continuous']
                carc_labels = torch.masked_select(batch_data["carc_continuous"], carc_mask)
                carc_logging = carc_labels

            elif type(carc_loss_criterion) == torch.nn.modules.loss.CrossEntropyLoss:
                carc_mask = batch_data['carc_mask_multi']
                carc_labels = torch.masked_select(batch_data["carc_label_multi"], carc_mask)
                carc_logging = carc_labels

            else:
                raise ValueError

            # get carc logits
            if args['train_carc_loss_fnc'] == 'CE':
                # Add predictions to dataset
                eval_dataset['carc_pred'].extend(logits[:, 1:6].argmax(dim=1).cpu().numpy())
                new_mask = carc_mask.view(-1, 1).expand(logits[:, 1:6].shape)
                carc_logits = torch.masked_select(logits[:, 1:6], new_mask).view(-1, 5)
                # carc_pred = (nn.functional.softmax(carc_logits) * torch.arange(1, 6).to(args['device'])).sum(axis=1)
                carc_pred = carc_logits.argmax(dim=1)

            elif args['train_carc_loss_fnc'] == 'MSE':
                eval_dataset['carc_pred'].extend(logits[:, 1].cpu().numpy())
                carc_logits = torch.masked_select(logits[:, 1], carc_mask)
                carc_pred = carc_logits

            elif args['train_carc_loss_fnc'] == 'BCE':
                eval_dataset['carc_pred'].extend(logits[:, 1].cpu().numpy())
                carc_logits = torch.masked_select(logits[:, 1], carc_mask)
                carc_pred = carc_logits

            else:
                raise ValueError

            # Always update carc_meter
            if len(carc_logits) > 0:
                eval_carc_meter.update(carc_pred.view(-1, 1), carc_logging.view(-1, 1))

            # XNOR: calculate it either when both are CE or when none are CE
            CE_train_xnor_CE_eval = ((args['train_carc_loss_fnc'] == "CE") ==
                                    (type(carc_loss_criterion) == nn.CrossEntropyLoss))
            if len(carc_logits) > 0 and CE_train_xnor_CE_eval:
                carc_loss = carc_loss_criterion(carc_logits, carc_labels).mean()
            else:
                carc_loss = torch.tensor(0)

            mut_logits = torch.masked_select(logits[:, 0], batch_data['mut_mask'])
            mut_labels = torch.masked_select(batch_data["mut_label"], batch_data['mut_mask'])

            if len(mut_logits) > 0:
                mut_loss = mut_loss_criterion(mut_logits, mut_labels).mean()
                eval_mut_meter.update(mut_logits.view(-1, 1), mut_labels.view(-1, 1))
            else:
                mut_loss = torch.tensor(0)

            loss = carc_loss * (1 - args['mut_loss_ratio']) + mut_loss * args['mut_loss_ratio']
            losses.append(loss.item())
            carc_losses.append(carc_loss.item())
            mut_losses.append(mut_loss.item())
            ys.append(batch_data["mut_label"])

    # TODO fixme
    # ys = torch.cat(ys, dim=0).cpu()
    # viz_mols(args, epoch, model, losses, ys, data_loader.dataset)

    eval_carc_metric, carc_metric_name, eval_carc_metric2, carc_metric_name2 = perform_eval(
        carc_loss_criterion, eval_carc_meter)

    eval_mut_metric, mut_metric_name, eval_mut_metric2, mut_metric_name2 = perform_eval(
        mut_loss_criterion, eval_mut_meter)

    val_loss = np.nanmean(losses)
    val_carc_loss = np.nanmean(carc_losses)
    val_mut_loss = np.nanmean(mut_losses)

    return eval_carc_metric, carc_metric_name, eval_carc_metric2, carc_metric_name2,\
        eval_mut_metric, mut_metric_name, eval_mut_metric2, mut_metric_name2, \
        val_loss, val_carc_loss, val_mut_loss, pd.DataFrame(eval_dataset)


def perform_eval(loss_criterion, meter):

    if type(loss_criterion) == torch.nn.modules.loss.BCEWithLogitsLoss:
        eval_metric = np.nanmean(meter.compute_metric('roc_auc_score'))
        eval_metric2 = np.nanmean(meter.compute_metric('pr_auc_score'))
        metric_name = 'roc_auc_score'
        metric_name2 = 'pr_auc_score'

    elif type(loss_criterion) == torch.nn.modules.loss.MSELoss:
        eval_metric = np.nanmean(meter.compute_metric('r2'))
        eval_metric2 = np.nanmean(meter.compute_metric('rmse'))
        metric_name = 'pearson_r2'
        metric_name2 = 'rmse'

    elif type(loss_criterion) == torch.nn.modules.loss.CrossEntropyLoss:
        # TODO compute F1?
        eval_metric = np.nan
        eval_metric2 = np.nan
        metric_name = np.nan
        metric_name2 = np.nan

    else:
        raise ValueError
    return eval_metric, metric_name, eval_metric2, metric_name2


def perform_inference_and_log(
        args, model, data_loader, mut_loss_criterion, carc_loss_criterion, summary_dict, dataset_name, note=''
):
    if not data_loader:
        return summary_dict

    carc_metric_value, carc_metric_name, carc_metric_value2, carc_metric_name2, \
    mut_metric_value, mut_metric_name, mut_metric_value2, mut_metric_name2, \
    model_loss, model_carc_loss, model_mut_loss,\
    perfromance_df = run_an_eval_epoch(
        args, model, data_loader, mut_loss_criterion, carc_loss_criterion
    )

    summary_dict[f"{dataset_name}_mut_{mut_metric_name}{note}"] = mut_metric_value
    summary_dict[f"{dataset_name}_mut_{mut_metric_name2}{note}"] = mut_metric_value2
    summary_dict[f"{dataset_name}_carc_{carc_metric_name}{note}"] = carc_metric_value
    summary_dict[f"{dataset_name}_carc_{carc_metric_name2}{note}"] = carc_metric_value2
    summary_dict[f"{dataset_name}_mut_loss{note}"] = model_mut_loss
    summary_dict[f"{dataset_name}_carc_loss{note}"] = model_carc_loss
    summary_dict[f"{dataset_name}_loss{note}"] = model_loss

    return summary_dict, perfromance_df


def end_of_training_evaluation(
        model, args, mut_loss_criterion, carc_loss_criterion, held_out_test_carc_loss_criterion,
        val_loader, test_loader, held_out_test_data_loader, note='', save_note='', save_data=True
):
    """
    does inference on the validation dataset and logs it
    does inference on the test dataset and logs it
    does inference on the held out test set and logs it
    dumps the model and the config file

    :param model:
    :param args:
    :param mut_loss_criterion:
    :param carc_loss_criterion:
    :param held_out_test_carc_loss_criterion:
    :param val_loader:
    :param test_loader:
    :param held_out_test_data_loader:
    :param note
    :param save_data
    :return:
    """
    summary_dict = {}

    summary_dict, val_df = perform_inference_and_log(
        args, model, val_loader, mut_loss_criterion, carc_loss_criterion, summary_dict, "final_valid",
        note=note,
    )

    summary_dict, test_df = perform_inference_and_log(
        args, model, test_loader, mut_loss_criterion, carc_loss_criterion, summary_dict, "test",
        note=note,
    )

    summary_dict, held_out_test_df = perform_inference_and_log(
        args, model, held_out_test_data_loader, mut_loss_criterion,
        held_out_test_carc_loss_criterion, summary_dict, "held_out_test",
        note=note,
    )

    print(summary_dict)
    if args["use_wandb"] and save_data:
        checkpoint_fp = os.path.join(wandb.run.dir, f"checkpoint{save_note}.pkl")
        model.to("cpu")
        print(model)
        torch.save(model.state_dict(), checkpoint_fp)
        # Save config
        device = args['device']
        del args['device']
        config_path = os.path.join(wandb.run.dir, f'config{save_note}.json')
        with open(config_path, 'w') as config_file:
            json.dump(args, config_file)

        summary_dict_path = os.path.join(wandb.run.dir, f'summary{save_note}.json')
        with open(summary_dict_path, 'w') as summary_file:
            json.dump(summary_dict, summary_file)

        val_fp = os.path.join(wandb.run.dir, f"val{save_note}.csv")
        val_df.to_csv(val_fp, index=False)

        test_fp = os.path.join(wandb.run.dir, f"test{save_note}.csv")
        test_df.to_csv(test_fp, index=False)

        held_out_test_fp = os.path.join(wandb.run.dir, f'held_out_test{save_note}.csv')
        held_out_test_df.to_csv(held_out_test_fp, index=False)

        args['device'] = device
        model.to(device)

    return summary_dict


def training_loop(model, args, mut_loss_criterion, carc_loss_criterion, train_loader, val_loader, note=''):
    stopper = construct_stopper(args)

    optimizer = Adam(model.parameters(), lr=args['lr'], weight_decay=args['network_weight_decay'])
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=args['lr_decay_factor'])

    for epoch in range(args['num_epochs']):
        train_loss, train_carc_loss, train_mut_loss,\
        train_mut_metric, train_mut_metric_name, train_mut_metric2, train_mut_metric_name2, \
        train_carc_metric, train_carc_metric_name, train_carc_metric2, train_carc_metric_name2 = run_a_train_epoch(
            args, epoch, model, train_loader, mut_loss_criterion, carc_loss_criterion, optimizer)

        # Validation and early stop
        val_carc_metric, val_carc_metric_name, val_carc_metric2, val_carc_metric_name2,\
        val_mut_metric, val_mut_metric_name, val_mut_metric2, val_mut_metric_name2, \
        val_loss, val_carc_loss, val_mut_loss, \
        performance_df = run_an_eval_epoch(
            args, model, val_loader, mut_loss_criterion, carc_loss_criterion
        )

        if args['early_stopping_metric'][1] == 'carc':
            if args['early_stopping_metric'][0] == 'roc_auc_score':
                val_score = val_carc_metric
                if val_carc_metric_name != 'roc_auc_score':
                    raise ValueError

            elif args['early_stopping_metric'][0] == 'pearson_r2':
                val_score = val_carc_metric
                if val_carc_metric_name != 'pearson_r2':
                    raise ValueError

            elif args['early_stopping_metric'][0] == 'rmse':
                val_score = val_carc_metric2
                if val_carc_metric_name2 != 'rmse':
                    raise ValueError

            elif args['early_stopping_metric'][0] == 'validation_loss':
                val_score = val_loss

            else:
                raise ValueError
        elif args['early_stopping_metric'][1] == 'mut':
            if args['early_stopping_metric'][0] == 'roc_auc_score':
                val_score = val_mut_metric
                if val_mut_metric_name != 'roc_auc_score':
                    raise ValueError
            else:
                raise ValueError
        else:
            raise ValueError

        scheduler.step(val_score)
        early_stop = stopper.step(val_score, model)

        print(f"Training: epoch   {epoch + 1:d}/{args['num_epochs']:d}, "
              f"training loss     {train_loss:.3f}, "
              f"mut_{train_mut_metric_name} {train_mut_metric:.3f} "
              f"mut_{train_mut_metric_name2} {train_mut_metric2:.3f} "
              f"carc_{train_carc_metric_name} {train_carc_metric:.3f} "
              f"carc_{train_carc_metric_name2} {train_carc_metric2:.3f} "
              )
        print(f"Validation: epoch {epoch + 1:d}/{args['num_epochs']:d}, "
              f"validation loss   {val_loss:.3f}, "
              f"mut_{val_mut_metric_name} {val_mut_metric:.3f} "
              f"mut_{val_mut_metric_name2} {val_mut_metric2:.3f} "
              f"carc_{val_carc_metric_name} {val_carc_metric:.3f} "
              f"carc_{val_carc_metric_name2} {val_carc_metric2:.3f} \n"
              )

        if args["use_wandb"]:
            wandb.log({
                f"epoch{note}": epoch + 1,
                f"training_carcinogenic_loss{note}": train_carc_loss,
                f"training_mutagenic_loss{note}": train_mut_loss,
                f"training_loss{note}": train_loss,
                f"training_mut_{train_mut_metric_name}{note}": train_mut_metric,
                f"training_mut_{train_mut_metric_name2}{note}": train_mut_metric2,
                f"training_carc_{train_carc_metric_name}{note}": train_carc_metric,
                f"training_carc_{train_carc_metric_name2}{note}": train_carc_metric2,

                f"validation_loss{note}": val_loss,
                f"validation_carc_loss{note}": val_carc_loss,
                f"validation_mut_loss{note}": val_mut_loss,
                f"validation_carc_{val_carc_metric_name}{note}": val_carc_metric,
                f"validation_carc_{val_carc_metric_name2}{note}": val_carc_metric2,
                f"validation_mut_{val_mut_metric_name}{note}": val_mut_metric,
                f"validation_mut_{val_mut_metric_name2}{note}": val_mut_metric2,
            })

        if early_stop:
            break
    stopper.load_checkpoint(model)
    return model


def construct_stopper(args):
    if args["use_wandb"]:
        es_filename = os.path.join(wandb.run.dir, "early_stopping")
    else:
        es_filename = "/tmp/es.pth"

    if args['early_stopping_metric'][0] == 'validation_loss':
        # ugly hack because we're using a different loss
        metric_for_stopper = 'rmse'
    else:
        metric_for_stopper = args['early_stopping_metric'][0]

    stopper = EarlyStopping(patience=args['patience'],
                            filename=es_filename,
                            metric=metric_for_stopper)

    return stopper


def construct_data_loader(dataset, args, collate_fn, shuffle=True, drop_last=False):
    if len(dataset) == 0:
        return None
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=args['batch_size'],
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=args['num_workers'],
        drop_last=drop_last
    )
    if type(dataset) == dgl.data.utils.Subset:
        mut_labels = data_loader.dataset.dataset.mut_labels.flatten().numpy()[data_loader.dataset.indices]
        carc_values = data_loader.dataset.dataset.carc_labels.flatten().numpy()[data_loader.dataset.indices]
        print(f"Number of mutagenic molecules: \n{pd.value_counts(mut_labels)}")
        print(f"Number of carciongenic molecules: {pd.notnull(carc_values).sum()}")

    elif type(dataset) == GraphCancerMolecules:
        mut_labels = data_loader.dataset.mut_labels.flatten().numpy()
        carc_values = data_loader.dataset.carc_labels.flatten().numpy()
        print(f"Number of mutagenic molecules: \n{pd.value_counts(mut_labels)}")
        print(f"Number of carciongenic molecules: {pd.notnull(carc_values).sum()}")
    elif type(dataset) == SelfiesCancerMolecules:
        mut_labels = data_loader.dataset.mut_labels.flatten().numpy()
        carc_values = data_loader.dataset.carc_labels.flatten().numpy()
        print(f"Number of mutagenic molecules: \n{pd.value_counts(mut_labels)}")
        print(f"Number of carciongenic molecules: {pd.notnull(carc_values).sum()}")

    else:
        raise ValueError

    return data_loader


def set_device_and_set_seed(args):
    if torch.cuda.is_available():
        args['device'] = torch.device('cuda:0')
    else:
        args['device'] = torch.device('cpu')

    # set random seeds
    torch.manual_seed(args["random_seed"])
    np.random.seed(args["random_seed"]*2)
    return args


def load_data(args):
    data, held_out_test_data = get_datasets(args)
    train, val, test = split_data(data)
    data_feats = data.get_data_feats()
    collate_fn = data.get_collate_fn()

    train_loader = construct_data_loader(train, args, collate_fn)
    val_loader = construct_data_loader(val, args, collate_fn)
    test_loader = construct_data_loader(test, args, collate_fn)
    held_out_test_data_loader = construct_data_loader(held_out_test_data, args, collate_fn)

    return train_loader, val_loader, test_loader, held_out_test_data_loader, data_feats


def load_kfold_cross_validation_data(args):
    data, held_out_test_data = get_datasets(args)
    list_of_data_tuples = kfold_split_data(data, k=3)
    data_feats = data.get_data_feats()
    collate_fn = data.get_collate_fn()

    list_of_data_loader_tuples = []
    for i, (train, val) in enumerate(list_of_data_tuples):
        # split into validation and test
        val, test, _ = split_subset_data_loader(val, train_fraction=0.5, val_fraction=0.5, test_fraction=0)
        print(f"fold {i}:")
        # a subset of a subset gets created so for val and test have to get the sub-object
        val = construct_data_loader(val, args, collate_fn)

        test = construct_data_loader(test, args, collate_fn)
        train = construct_data_loader(train, args, collate_fn)
        assert len(train) > 0
        assert len(val) > 0
        assert len(test) > 0

        list_of_data_loader_tuples.append((train, val, test))
    # don't forget the held out test loader
    held_out_test_data_loader = construct_data_loader(held_out_test_data, args, collate_fn)

    return list_of_data_loader_tuples, held_out_test_data_loader, data_feats


def split_loader_into_carc_and_mut(data_loader, args):
    assert type(data_loader) == torch.utils.data.dataloader.DataLoader
    assert type(data_loader.dataset) == dgl.data.utils.Subset
    data_loader_idx = np.array(data_loader.dataset.indices)
    og_dataset = data_loader.dataset.dataset
    collate_fn = og_dataset.get_collate_fn()

    carc_mask_data_loader = og_dataset.carc_mask[data_loader_idx].numpy()
    idx_carc_mask_data_loader = np.argwhere(carc_mask_data_loader).flatten()
    idx_carc_mask = data_loader_idx[idx_carc_mask_data_loader]
    carc_dataset = dgl.data.utils.Subset(og_dataset, idx_carc_mask)

    mut_mask_data_loader = og_dataset.mut_mask[data_loader_idx].numpy()
    idx_mut_mask_data_loader = np.argwhere(mut_mask_data_loader).flatten()
    idx_mut_mask = data_loader_idx[idx_mut_mask_data_loader]
    mut_dataset = dgl.data.utils.Subset(og_dataset, idx_mut_mask)

    mut_data_loader = construct_data_loader(
        mut_dataset, args, collate_fn, shuffle=True, drop_last=False)
    carc_data_loader = construct_data_loader(
        carc_dataset, args, collate_fn, shuffle=True, drop_last=False)
    return mut_data_loader, carc_data_loader


def mutagenicity_pre_training(args, train_loader, val_loader, model, mut_loss_criterion, carc_loss_criterion, note=''):
    train_mut_loader, train_carc_loader = split_loader_into_carc_and_mut(train_loader, args)

    # mut loss only
    args['use_carc_loss'] = False
    args['use_mut_loss'] = True
    args['early_stopping_metric'] = ('roc_auc_score', 'mut')
    model = training_loop(
        model, args, mut_loss_criterion, carc_loss_criterion, train_mut_loader, val_loader, note=note)

    # carc loss only
    args['use_carc_loss'] = True
    args['use_mut_loss'] = False

    if args['train_carc_loss_fnc'] == 'MSE':
        args['early_stopping_metric'] = ('rmse', 'carc')
    elif args['train_carc_loss_fnc'] == 'BCE':
        args['early_stopping_metric'] = ('roc_auc_score', 'carc')
    elif args['train_carc_loss_fnc'] == 'CE':
        args['early_stopping_metric'] = ('validation_loss', 'carc')
    else:
        raise ValueError

    model = training_loop(
        model, args, mut_loss_criterion, carc_loss_criterion, train_carc_loader, val_loader, note=note)
    # will do this K times
    args['use_carc_loss'] = True
    args['use_mut_loss'] = True

    return model


def kfold_cross_validation_mode(args, mut_loss_criterion, carc_loss_criterion, held_out_test_carc_loss_criterion):
    # load the datasets for every fold
    list_of_data_tuples, held_out_test_data_loader, data_feats = load_kfold_cross_validation_data(args)

    job_type = make_timestamp()
    summary_dicts = []

    for i in range(len(list_of_data_tuples)):
        model = get_model(args, data_feats)
        model.to(args['device'])

        wandb_run = None

        if wandb_run:
            wandb_run.finish()

        wandb_run = set_up_wandb(
            model, args, reinit=True, name=args["run"] + f'_{i}', job_type=job_type, group=args['group_name']
        )

        train_loader = list_of_data_tuples[i][0]
        val_loader = list_of_data_tuples[i][1]
        test_loader = list_of_data_tuples[i][2]

        if not args['mut_pre_training']:
            model = training_loop(
                model, args, mut_loss_criterion, carc_loss_criterion, train_loader, val_loader, note=f"_{i}")

        else:

            for j in range(args['num_mut_pre_training_loop']):
                model = mutagenicity_pre_training(
                    args, train_loader, val_loader, model, mut_loss_criterion, carc_loss_criterion, note=f"_{i}")

                if j < args['num_mut_pre_training_loop'] - 1:
                    summary_dict = end_of_training_evaluation(
                        model, args, mut_loss_criterion, carc_loss_criterion, held_out_test_carc_loss_criterion,
                        val_loader, test_loader, held_out_test_data_loader, note=f"_{j}", save_data=False
                    )

        # save notes are different but logging will be the same
        summary_dict = end_of_training_evaluation(
            model, args, mut_loss_criterion, carc_loss_criterion, held_out_test_carc_loss_criterion,
            val_loader, test_loader, held_out_test_data_loader, save_note=f"_{i}"
        )
        summary_dicts.append(summary_dict)

    final_summary_dict = {}
    assert len(summary_dicts) > 0
    for i, key in enumerate(summary_dicts[0].keys()):
        final_summary_dict[key] = np.mean([s_dict[key] for s_dict in summary_dicts])

    wandb.log(final_summary_dict)
    # Save dataset prediction values for easier ensembling afterwards
    # Should I retrain on the whole dataset? What would be my external validation data?
    # Should I generate an ensemble?
    return summary_dicts


def train_val_test_mode(args, mut_loss_criterion, carc_loss_criterion, held_out_test_carc_loss_criterion):

    train_loader, val_loader, test_loader, held_out_test_data_loader, data_feats = load_data(args)

    model = get_model(args, data_feats)
    model.to(args['device'])

    wandb_run = set_up_wandb(model, args, name=args['run'], group=args['group_name'])
    print(data_feats)
    print(model)

    if not args['mut_pre_training']:
        model = training_loop(model, args, mut_loss_criterion, carc_loss_criterion, train_loader, val_loader)

    else:
        for j in range(args['num_mut_pre_training_loop']):
            model = mutagenicity_pre_training(
                args, train_loader, val_loader, model, mut_loss_criterion, carc_loss_criterion)

            if j < args['num_mut_pre_training_loop'] - 1:
                summary_dict = end_of_training_evaluation(
                    model, args, mut_loss_criterion, carc_loss_criterion, held_out_test_carc_loss_criterion,
                    val_loader, test_loader, held_out_test_data_loader, note=f"_{j}", save_data=False
                )

    summary_dict = end_of_training_evaluation(
        model, args, mut_loss_criterion, carc_loss_criterion, held_out_test_carc_loss_criterion,
        val_loader, test_loader, held_out_test_data_loader
    )

    wandb.log(summary_dict)
    return summary_dict


def set_up_wandb(model, args, job_type=None, name=None, reinit=False, group=None):
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    args['number_of_parameters'] = pytorch_total_params
    print(args)

    if args["use_wandb"]:
        wandb_config = copy.deepcopy(args)
        del wandb_config["run"]
        wandb_run = wandb.init(
            project=f"CONCERTO_v2{args['project_note']}", name=name, config=wandb_config, job_type=job_type,
            reinit=reinit, dir=args['wandb_run_dir'], group=group
        )
    else:
        wandb_run = None

    return wandb_run


def get_loss_criteria(args):

    if args['train_carc_loss_fnc'] == 'MSE':
        carc_loss_criterion = nn.MSELoss()
    elif args['train_carc_loss_fnc'] == "CE":
        carc_loss_criterion = nn.CrossEntropyLoss(reduction='none')
        assert args['out_feats'] == 6
    elif args['train_carc_loss_fnc'] == 'BCE':
        carc_loss_criterion = nn.BCEWithLogitsLoss(reduction='none')
        # set it in cases with BCE
        args['early_stopping_metric'] = ("roc_auc_score", 'carc')
    else:
        raise ValueError

    mut_loss_criterion = nn.BCEWithLogitsLoss(reduction='none')
    held_out_test_carc_loss_criterion = nn.BCEWithLogitsLoss(reduction='none')

    return carc_loss_criterion, mut_loss_criterion, held_out_test_carc_loss_criterion


def main():

    args = get_args()

    # Yaml sweeps don't take in tuples so need to hack it
    butchered_data = (
        'c', 'a', 'r', 'c', '_', 'c', 'a', 'p', 's', '_', 'p', 'r', 'e', 'd', '_', 'e', 'l')
    if args['training_carc_datasets'] == butchered_data:
        args['training_carc_datasets'] = ('carc_caps_pred_el', )
    # find device
    args = set_device_and_set_seed(args)

    carc_loss_criterion, mut_loss_criterion, held_out_test_carc_loss_criterion = get_loss_criteria(args)

    if not args['cross_validation']:
        training_procedure = train_val_test_mode
    else:
        training_procedure = kfold_cross_validation_mode

    summary = training_procedure(args, mut_loss_criterion, carc_loss_criterion, held_out_test_carc_loss_criterion)


if __name__ == "__main__":
    main()
