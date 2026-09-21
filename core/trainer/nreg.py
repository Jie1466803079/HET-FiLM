import json
import torch
import os.path as osp
import sys
import os

import numpy as np
from core.utils import EarlyStopping
from tqdm import tqdm
import time
from torch import nn
from torch.nn import functional as F
from core.utils import setup_seed

# Add path to evaluation metrics
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from evaluation_metrics import calculate_comprehensive_metrics, get_empty_metrics


def train(
    model, optimizer, criterion, train_data, culmulate=1, grad_clip=0, device="cpu", args=None
):
    model.train()

    losses = []
    aux_lambda = getattr(args, 'aux_lambda', 0.2) if args is not None else 0.0
    diff_sampling = os.environ.get('MG_DIFF_SAMPLING', '0').strip().lower() in ('1','true','yes','y')

    for support, query in train_data:
        # Set edge features if model supports it (for EdgeGRU)
        if hasattr(model, 'set_edge_features') and hasattr(support, 'edge_features'):
            model.set_edge_features(support.edge_features)
        
        if diff_sampling and hasattr(model, 'set_target_batch'):
            # Mini-batch over target fund nodes
            y = query.y.squeeze()
            valid = torch.isfinite(y)
            idx_all = torch.nonzero(valid, as_tuple=True)[0]
            bs = int(os.environ.get('MG_BATCH_SIZE', '1024'))
            for i in range(0, idx_all.numel(), bs):
                idx = idx_all[i:i+bs]
                try:
                    model.set_target_batch(idx)
                except Exception:
                    pass
                encode_out = model.encode(support)
                z = encode_out[0] if (isinstance(encode_out, tuple) and len(encode_out)==2) else encode_out
                out = model.decode_nclf(z).squeeze()
                pred = out.index_select(0, idx)
                targ = y.index_select(0, idx)
                mask = torch.isfinite(pred) & torch.isfinite(targ)
                if mask.sum() == 0:
                    continue
                main_loss = criterion(pred[mask], targ[mask])
                optimizer.zero_grad()
                main_loss.backward()
                if grad_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                losses.append(main_loss.item())
            try:
                model.set_target_batch(None)
            except Exception:
                pass
        else:
            # Full-batch training
            encode_out = model.encode(support)
            if isinstance(encode_out, tuple) and len(encode_out) == 2:
                z, aux_outputs = encode_out
            else:
                z = encode_out
                aux_outputs = None
            out = model.decode_nclf(z)
            targets = query.y
            pred = out.squeeze()
            targ = targets.squeeze()
            # Align to labeled nodes if present_mask exists
            if hasattr(query, "present_mask"):
                pm = query.present_mask.bool()
                # If pred is longer (full graph) and targ already masked, slice pred by pm
                if pm.numel() == pred.shape[0] and pred.shape[0] != targ.shape[0]:
                    pred = pred[pm]
                # If shapes match pm, optionally mask both to labeled nodes
                elif pm.numel() == pred.shape[0] == targ.shape[0]:
                    pred = pred[pm]
                    targ = targ[pm]
            # mask invalid
            mask = torch.isfinite(pred) & torch.isfinite(targ)
            if mask.sum() == 0:
                continue
            main_loss = criterion(pred[mask], targ[mask])
            if aux_outputs and hasattr(model, 'auxiliary_losses') and aux_lambda > 0:
                edge_features = getattr(support, 'edge_features', None)
                if edge_features:
                    aux_loss = model.auxiliary_losses(aux_outputs, edge_features)
                    main_loss = main_loss + aux_lambda * aux_loss
            optimizer.zero_grad()
            main_loss.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            losses.append(main_loss.item())

    # Avoid propagating NaN when no batches contributed (e.g., all-masked windows)
    # Returning 0.0 keeps the training loop numerically stable and lets evaluation proceed.
    return float(np.mean(losses)) if len(losses) > 0 else 0.0


def train_xgboost_model(model, dataset, args, device="cpu"):
    """
    Special training procedure for XGBoost model.
    """
    import time
    print("\n" + "="*80)
    print("TRAINING XGBOOST MODEL")
    print("="*80)

    start_time = time.time()

    # Train XGBoost using the wrapper's method
    train_mae, val_mae = model.train_xgboost(
        dataset.time_dataset['train'],
        dataset.time_dataset['val']
    )

    print(f"\n✓ XGBoost training complete!")
    print(f"  Train MAE: {train_mae:.4f}")
    print(f"  Val MAE:   {val_mae:.4f}")

    # Evaluate on test set
    test_mae = test(model, dataset.test_dataset, device=device)

    print(f"  Test MAE:  {test_mae:.4f}")
    print(f"\n  Training time: {time.time() - start_time:.2f}s")
    print("="*80 + "\n")

    return {
        "test_mae": float(test_mae),
        "val_mae": float(val_mae),
        "train_mae": float(train_mae),
        "epoch": 0,  # XGBoost doesn't have epochs in the same sense
        "best_epoch": 0,
        "time": float(time.time() - start_time),
    }


@torch.no_grad()
def test(model, data, device="cpu", return_comprehensive=False):
    def to_raw(query, preds):
        # If loader applied log-shift, inverse-transform to raw units
        offset = getattr(query, 'target_offset', None)
        if hasattr(query, 'y_raw_target') and offset is not None:
            y_true = query.y_raw_target
            y_pred = torch.exp(preds) - float(offset)
            return y_true, y_pred
        return query.y, preds

    def test_one(model, data):
        support, query = data
        model.eval()
        
        # Fast path for non-torch regressors (e.g., XGBoost)
        if hasattr(model, 'predict_tensor') and hasattr(query, 'x'):
            preds = model.predict_tensor(query.x)
            targets, preds = to_raw(query, preds.squeeze())
            if return_comprehensive:
                return calculate_comprehensive_metrics(preds.squeeze(), targets.squeeze())
            else:
                mae = F.l1_loss(preds.squeeze(), targets.squeeze()).item()
                return mae

        # Set edge features if model supports it (for EdgeGRU)
        if hasattr(model, 'set_edge_features') and hasattr(support, 'edge_features'):
            model.set_edge_features(support.edge_features)
        
        diff_sampling = os.environ.get('MG_DIFF_SAMPLING', '0').strip().lower() in ('1','true','yes','y')
        if diff_sampling and hasattr(model, 'set_target_batch') and not return_comprehensive:
            # Batched MAE over targets
            y = query.y.squeeze()
            valid = torch.isfinite(y)
            idx_all = torch.nonzero(valid, as_tuple=True)[0]
            bs = int(os.environ.get('MG_EVAL_BATCH_SIZE', os.environ.get('MG_BATCH_SIZE', '2048')))
            maes = []
            for i in range(0, idx_all.numel(), bs):
                idx = idx_all[i:i+bs]
                try:
                    model.set_target_batch(idx)
                except Exception:
                    pass
                encode_out = model.encode(support)
                z = encode_out[0] if (isinstance(encode_out, tuple) and len(encode_out)==2) else encode_out
                out = model.decode_nclf(z).squeeze()
                pred = out.index_select(0, idx)
                targ = y.index_select(0, idx)
                t_raw, p_raw = to_raw(query, pred)
                mask = torch.isfinite(p_raw) & torch.isfinite(t_raw)
                if mask.sum() > 0:
                    maes.append(F.l1_loss(p_raw[mask], t_raw[mask]).item())
            try:
                model.set_target_batch(None)
            except Exception:
                pass
            return float(np.mean(maes)) if maes else float('nan')
        else:
            # Full-batch
            encode_out = model.encode(support)
            if isinstance(encode_out, tuple) and len(encode_out) == 2:
                z, _ = encode_out  # Ignore aux_outputs during evaluation
            else:
                z = encode_out
            out = model.decode_nclf(z)
            targets, out = to_raw(query, out.squeeze())
            # Align to labeled nodes if present_mask exists
            if hasattr(query, "present_mask"):
                pm = query.present_mask.bool()
                if pm.numel() == out.shape[0] and out.shape[0] != targets.shape[0]:
                    out = out[pm]
                elif pm.numel() == out.shape[0] == targets.shape[0]:
                    out = out[pm]
                    targets = targets[pm]
            mask = torch.isfinite(out) & torch.isfinite(targets)
            if mask.sum() == 0:
                return float('nan') if not return_comprehensive else get_empty_metrics()
            out = out[mask]
            targets = targets[mask]
            if return_comprehensive:
                return calculate_comprehensive_metrics(out.squeeze(), targets.squeeze())
            else:
                return F.l1_loss(out.squeeze(), targets.squeeze()).item()

    if isinstance(data, list):
        if return_comprehensive:
            # Aggregate comprehensive metrics across multiple data points
            all_predictions = []
            all_targets = []
            for d in data:
                support, query = d
                model.eval()
                
                # Set edge features if model supports it (for EdgeGRU)
                if hasattr(model, 'set_edge_features') and hasattr(support, 'edge_features'):
                    model.set_edge_features(support.edge_features)
                
                # Handle potential tuple return from encode
                encode_out = model.encode(support)
                if isinstance(encode_out, tuple) and len(encode_out) == 2:
                    z, _ = encode_out  # Ignore aux_outputs during evaluation
                else:
                    z = encode_out
                
                out = model.decode_nclf(z)
                targets, out = to_raw(query, out.squeeze())
                # Align to labeled nodes if present_mask exists
                if hasattr(query, "present_mask"):
                    pm = query.present_mask.bool()
                    if pm.numel() == out.shape[0] and out.shape[0] != targets.shape[0]:
                        out = out[pm]
                    elif pm.numel() == out.shape[0] == targets.shape[0]:
                        out = out[pm]
                        targets = targets[pm]
                mask = torch.isfinite(out) & torch.isfinite(targets)
                if mask.sum() == 0:
                    continue
                out = out[mask]
                targets = targets[mask]
                all_predictions.append(out.squeeze())
                all_targets.append(targets.squeeze())

            # Concatenate all predictions and targets (guard empty datasets)
            if len(all_predictions) == 0 or len(all_targets) == 0:
                return get_empty_metrics()
            all_predictions = torch.cat(all_predictions)
            all_targets = torch.cat(all_targets)
            return calculate_comprehensive_metrics(all_predictions, all_targets)
        else:
            maes = [test_one(model, d) for d in data]
            return np.mean(maes)
    return test_one(model, data)


def train_till_end(
    model,
    optimizer,
    criterion,
    dataset,
    args,
    max_epochs,
    patience,
    disable_progress=False,
    writer=None,
    grad_clip=0,
    device="cpu",
):
    # Special handling for XGBoost
    if hasattr(model, 'train_xgboost'):
        return train_xgboost_model(model, dataset, args, device)

    # procedure
    setup_seed(args.seed)
    start_time = time.time()
    # Decide primary selection metric
    loss_mode = getattr(args, 'loss_mode', 'mse')
    target_mode = getattr(args, 'target_mode', 'value')
    use_rank_ic_primary = (loss_mode == 'ic') or (str(target_mode).lower() == 'rank')

    best_epoch = 0
    best_model_state = None
    # Track the best score for early stopping (maximize Rank-IC or minimize MAE)
    if use_rank_ic_primary:
        best_val_score = -1e8  # higher is better for Rank-IC
    else:
        best_val_score = 1e8   # lower is better for MAE

    # Track complete training history
    history = {
        'train_mae': [],
        'val_mae': [],
        'test_mae': [],
        'train_rank_ic': [],
        'val_rank_ic': [],
        'test_rank_ic': [],
        'loss': [],
        'overfitting_gap': [],  # val_mae - train_mae
        'generalization_gap': []  # test_mae - val_mae
    }

    earlystop = EarlyStopping(mode=("max" if use_rank_ic_primary else "min"), patience=patience)

    with tqdm(range(max_epochs), disable=disable_progress) as bar:
        for epoch in bar:
            loss = train(
                model,
                optimizer,
                criterion,
                dataset.train_dataset,
                grad_clip=grad_clip,
                device=device,
                args=args,
            )

            # Compute comprehensive metrics for all splits
            train_metrics = test(model, dataset.train_dataset, device=device, return_comprehensive=True)
            val_metrics = test(model, dataset.val_dataset, device=device, return_comprehensive=True)
            test_metrics = test(model, dataset.test_dataset, device=device, return_comprehensive=True)

            # Extract MAE and Rank-IC
            train_mae = float(train_metrics.get('MAE', float('nan')))
            val_mae = float(val_metrics.get('MAE', float('nan')))
            test_mae = float(test_metrics.get('MAE', float('nan')))
            train_rank_ic = float(train_metrics.get('Rank_IC', float('nan')))
            val_rank_ic = float(val_metrics.get('Rank_IC', float('nan')))
            test_rank_ic = float(test_metrics.get('Rank_IC', float('nan')))

            # Calculate gaps for overfitting/underfitting analysis
            overfitting_gap = val_mae - train_mae  # Positive = overfitting
            generalization_gap = test_mae - val_mae  # Large positive = poor generalization

            # Store history
            history['train_mae'].append(float(train_mae))
            history['val_mae'].append(float(val_mae))
            history['test_mae'].append(float(test_mae))
            history['loss'].append(float(loss))
            history['train_rank_ic'].append(float(train_rank_ic))
            history['val_rank_ic'].append(float(val_rank_ic))
            history['test_rank_ic'].append(float(test_rank_ic))
            history['overfitting_gap'].append(float(overfitting_gap))
            history['generalization_gap'].append(float(generalization_gap))

            # Save best model based on primary selection metric
            current_val_score = val_rank_ic if use_rank_ic_primary else val_mae
            is_better = (current_val_score > best_val_score) if use_rank_ic_primary else (current_val_score < best_val_score)
            if is_better:
                best_val_score = current_val_score
                best_epoch = epoch
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

                # Save checkpoint to disk for reproducibility
                if hasattr(args, 'log_dir') and args.log_dir:
                    checkpoint_path = osp.join(args.log_dir, 'best_model.pt')
                    os.makedirs(args.log_dir, exist_ok=True)
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': best_model_state,
                        'val_mae': val_mae,
                        'train_mae': train_mae,
                        'test_mae': test_mae,
                        'val_rank_ic': val_rank_ic,
                        'train_rank_ic': train_rank_ic,
                        'test_rank_ic': test_rank_ic,
                        'optimizer_state_dict': optimizer.state_dict(),
                        'args': vars(args) if hasattr(args, '__dict__') else {}
                    }, checkpoint_path)

            # Detect overfitting/underfitting status
            status = ""
            if overfitting_gap > 0.05:  # Significant overfitting threshold
                status = "⚠️OVERFIT"
            elif train_mae > 0.15:  # Underfitting threshold (high train error)
                status = "⚠️UNDERFIT"
            elif overfitting_gap < 0.02:  # Good fit
                status = "✓"

            # Enhanced progress bar with overfitting indicators
            bar.set_postfix(
                loss=f"{loss:.4f}",
                train=f"{train_mae:.4f}",
                val=f"{val_mae:.4f}",
                test=f"{test_mae:.4f}",
                gap=f"{overfitting_gap:.4f}",
                status=status,
                best_ep=best_epoch
            )

            if writer:
                writer.add_scalar("Model/train_loss", loss, epoch)
                writer.add_scalar("Model/train_mae", train_mae, epoch)
                writer.add_scalar("Model/val_mae", val_mae, epoch)
                writer.add_scalar("Model/test_mae", test_mae, epoch)
                writer.add_scalar("Model/overfitting_gap", overfitting_gap, epoch)
                writer.add_scalar("Model/generalization_gap", generalization_gap, epoch)

            if earlystop.step(current_val_score):
                print(f"\n⏹️  Early stopping triggered at epoch {epoch}")
                break

    # Load best model and compute final test performance
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"\n✓ Loaded best model from epoch {best_epoch}")

    # Recompute final metrics on best model
    final_train_metrics = test(model, dataset.train_dataset, device=device, return_comprehensive=True)
    final_val_metrics = test(model, dataset.val_dataset, device=device, return_comprehensive=True)
    final_test_metrics = test(model, dataset.test_dataset, device=device, return_comprehensive=True)

    final_train_mae = float(final_train_metrics.get('MAE', float('nan')))
    final_val_mae = float(final_val_metrics.get('MAE', float('nan')))
    final_test_mae = float(final_test_metrics.get('MAE', float('nan')))

    final_train_rank_ic = float(final_train_metrics.get('Rank_IC', float('nan')))
    final_val_rank_ic = float(final_val_metrics.get('Rank_IC', float('nan')))
    final_test_rank_ic = float(final_test_metrics.get('Rank_IC', float('nan')))

    # Analyze overfitting/underfitting patterns
    analysis = analyze_training_behavior(history, best_epoch)

    # Print training analysis
    print("\n" + "="*80)
    print("TRAINING BEHAVIOR ANALYSIS")
    print("="*80)
    print(f"Best Epoch: {best_epoch}")
    # Primary: Rank-IC first
    print(f"Final Train Rank-IC: {final_train_rank_ic:.4f}")
    print(f"Final Val Rank-IC:   {final_val_rank_ic:.4f}")
    print(f"Final Test Rank-IC:  {final_test_rank_ic:.4f}")
    # Rank-IC gaps (positive train-val indicates overfitting)
    print(f"Rank-IC Overfit Gap (Train-Val): {final_train_rank_ic - final_val_rank_ic:.4f}")
    print(f"Rank-IC Gen Drop (Val-Test):     {final_val_rank_ic - final_test_rank_ic:.4f}")
    # Then MAE as secondary
    print(f"Final Train MAE: {final_train_mae:.4f}")
    print(f"Final Val MAE:   {final_val_mae:.4f}")
    print(f"Final Test MAE:  {final_test_mae:.4f}")
    print(f"Overfitting Gap (Val-Train): {final_val_mae - final_train_mae:.4f}")
    print(f"Generalization Gap (Test-Val): {final_test_mae - final_val_mae:.4f}")
    print(f"\n{analysis['summary']}")
    print("="*80 + "\n")

    return {
        "test_mae": final_test_mae,
        "val_mae": final_val_mae,
        "train_mae": final_train_mae,
        "test_rank_ic": final_test_rank_ic,
        "val_rank_ic": final_val_rank_ic,
        "train_rank_ic": final_train_rank_ic,
        "epoch": epoch,
        "best_epoch": best_epoch,
        "history": history,
        "analysis": analysis,
        "time": time.time() - start_time,
        "time_per_epoch": (time.time() - start_time) / (epoch + 1),
    }


def analyze_training_behavior(history, best_epoch):
    """
    Analyze training history to detect overfitting, underfitting, and other patterns.

    Args:
        history: Dictionary containing training metrics history
        best_epoch: Epoch with best validation performance

    Returns:
        Dictionary with analysis results
    """
    train_maes = np.array(history['train_mae'])
    val_maes = np.array(history['val_mae'])
    test_maes = np.array(history['test_mae'])
    gaps = np.array(history['overfitting_gap'])

    analysis = {
        'best_train_mae': float(train_maes[best_epoch]),
        'best_val_mae': float(val_maes[best_epoch]),
        'best_test_mae': float(test_maes[best_epoch]),
        'best_gap': float(gaps[best_epoch]),
        'max_gap': float(np.max(gaps)),
        'final_gap': float(gaps[-1]),
    }

    # Determine overall training behavior
    best_gap = analysis['best_gap']
    final_train = train_maes[best_epoch]

    if best_gap > 0.08:
        status = "SEVERE OVERFITTING"
        recommendation = "⚠️  Model is overfitting significantly. Consider:\n" \
                        "   - Increase weight decay (wd)\n" \
                        "   - Add dropout\n" \
                        "   - Reduce model complexity\n" \
                        "   - Increase training data"
    elif best_gap > 0.04:
        status = "MODERATE OVERFITTING"
        recommendation = "⚡ Some overfitting detected. Consider:\n" \
                        "   - Slightly increase regularization\n" \
                        "   - Monitor for a few more epochs"
    elif final_train > 0.15:
        status = "UNDERFITTING"
        recommendation = "📈 Model is underfitting. Consider:\n" \
                        "   - Increase model capacity (hid_dim, n_layers)\n" \
                        "   - Train for more epochs\n" \
                        "   - Reduce regularization\n" \
                        "   - Increase learning rate"
    elif best_gap < 0.02:
        status = "GOOD FIT"
        recommendation = "✓ Model shows good generalization!"
    else:
        status = "ACCEPTABLE FIT"
        recommendation = "✓ Model performance is acceptable."

    # Check if training was improving when stopped
    if len(val_maes) > 5:
        recent_trend = val_maes[-5:] - val_maes[-6:-1]
        if np.mean(recent_trend) < 0:  # Still improving
            recommendation += "\n   ℹ️  Note: Validation was still improving. Consider longer patience."

    analysis['status'] = status
    analysis['summary'] = f"Status: {status}\n{recommendation}"

    return analysis


class NodePredictor(nn.Module):
    def __init__(self, n_inp: int, n_classes: int):
        """

        :param n_inp      : int, input dimension
        :param n_classes  : int, number of classes
        """
        super().__init__()

        self.fc1 = nn.Linear(n_inp, n_inp)
        self.fc2 = nn.Linear(n_inp, n_classes)
        self.input_adapter = None

    def forward(self, node_feat: torch.tensor):
        """

        :param node_feat: torch.tensor
        """

        # Adapt input dim to expected hidden size if they differ
        if node_feat.size(-1) != self.fc1.in_features:
            if (
                self.input_adapter is None
                or self.input_adapter.in_features != node_feat.size(-1)
                or self.input_adapter.out_features != self.fc1.in_features
            ):
                self.input_adapter = nn.Linear(
                    node_feat.size(-1), self.fc1.in_features
                ).to(node_feat.device)
            node_feat = self.input_adapter(node_feat)

        node_feat = F.relu(self.fc1(node_feat))
        pred = self.fc2(node_feat)  # Remove ReLU for regression - allow negative predictions

        return pred
