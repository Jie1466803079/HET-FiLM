import json
import os.path as osp
import torch
import torch.nn.functional as F


def load_trainer(args):
    task = getattr(args, 'task', 'regression').lower()
    if task == 'link':
        from .lpred import train_till_end as trainer
        criterion = torch.nn.BCEWithLogitsLoss()
        return trainer, criterion

    if args.dataset in "Aminer Ecomm".split():
        from .lpred import train_till_end as trainer
    elif args.dataset in "Yelp-nc".split():
        from .nclf import train_till_end as trainer
    elif args.dataset in "covid Funds".split():
        from .nreg import train_till_end as trainer

    if args.dataset in "Aminer Ecomm".split():
        criterion = torch.nn.BCEWithLogitsLoss()
    elif args.dataset in "Yelp-nc".split():
        criterion = F.cross_entropy
    elif args.dataset in "covid Funds".split():
        # Choose criterion based on args.loss_mode for Funds
        loss_mode = getattr(args, 'loss_mode', 'mse')
        if loss_mode == 'huber':
            criterion = torch.nn.HuberLoss(delta=1.0)
        elif loss_mode == 'ic':
            # Negative Pearson correlation (maximize IC)
            def ic_loss(pred, target):
                pred = pred.view(-1)
                target = target.view(-1)
                pred = pred - pred.mean()
                target = target - target.mean()
                denom = (pred.std(unbiased=False) * target.std(unbiased=False) + 1e-8)
                corr = (pred * target).mean() / denom
                return -corr
            criterion = ic_loss
        elif loss_mode == 'bce':
            criterion = torch.nn.BCEWithLogitsLoss()
        else:
            criterion = F.mse_loss  # default
    return trainer, criterion


def load_train_test(args):
    task = getattr(args, 'task', 'regression').lower()
    if task == 'link':
        from .lpred import train, test
        return train, test
    if args.dataset in "Aminer Ecomm".split():
        from .lpred import train, test
    elif args.dataset in "Yelp-nc".split():
        from .nclf import train, test
    elif args.dataset in "covid Funds".split():
        from .nreg import train, test
    return train, test


def log_train(log_dir, args, train_dict, writer, **kwargs):

    info_dict = args.__dict__
    measure_dict = {}
    # Handle different metric names based on task type
    if args.dataset in "covid Funds".split():
        # For regression tasks, prefer Rank-IC metrics if present
        for name in "test_rank_ic val_rank_ic train_rank_ic".split():
            if name in train_dict:
                measure_dict[name] = train_dict[name]
                del train_dict[name]
        # Also record MAE for reference
        for name in "test_mae val_mae train_mae".split():
            if name in train_dict:
                measure_dict[name] = train_dict[name]
                del train_dict[name]
    else:
        # For classification tasks, use AUC metrics
        for name in "test_auc val_auc train_auc".split():
            if name in train_dict:
                measure_dict[name] = train_dict[name]
                del train_dict[name]

    info_dict.update(train_dict)
    info_dict.update(kwargs)
    if writer:
        writer.add_hparams(info_dict, measure_dict)

    info_dict.update(measure_dict)

    json.dump(
        info_dict, open(osp.join(log_dir, "info.json"), "w"), indent=4, sort_keys=True
    )
