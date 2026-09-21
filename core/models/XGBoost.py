from torch import nn
import torch
import os


class XGBoostWrapper(nn.Module):
    """
    Lightweight wrapper that plugs an XGBoost-style regressor into the
    CORE training/evaluation pipeline for node regression on Funds.

    Interface:
    - train_xgboost(train_list, val_list): special training path used by nreg.train_till_end
    - predict_tensor(X): predict from a torch.Tensor of features
    - encode(support): returns feature tensor for downstream decode (fallback path)
    - decode_nclf(z): returns predictions given features tensor z
    """

    def __init__(
        self,
        n_inp,
        n_hid,
        n_layers,
        n_heads,
        time_window,
        norm,
        metadata,
        device,
        predict_type,
        featemb=None,
        nclf_linear=None,
        **xgb_params,
    ):
        super().__init__()
        self.metadata = metadata
        self.device = device
        self.predict_type = predict_type
        self._featemb = featemb  # unused for XGB, kept for compatibility
        self._head = nclf_linear  # unused

        # Require xgboost; no fallback
        import xgboost as xgb  # type: ignore
        self._use_xgb = True
        # Defaults chosen to be reasonably regularized for Funds:
        #  - shallower trees
        #  - fewer estimators
        #  - non-zero L1/L2 and min_child_weight
        params = dict(
            objective="reg:squarederror",
            max_depth=3,
            learning_rate=0.1,
            n_estimators=200,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            gamma=0.1,
            reg_alpha=0.1,
            reg_lambda=1.0,
            tree_method="hist",
            n_jobs=16,
        )
        # Allow CLI / tuning scripts to override any of the above.
        params.update({k: v for k, v in xgb_params.items() if v is not None})
        self._xgb = xgb.XGBRegressor(**params)

    # No learnable torch parameters; keep optimizer creation harmless
    def parameters(self, recurse: bool = True):  # type: ignore[override]
        return []

    # ============ Feature extraction helpers ============
    @staticmethod
    def _to_numpy(X: torch.Tensor):
        if X is None:
            return None
        if torch.is_tensor(X):
            return X.detach().cpu().numpy()
        return X

    def _gather_train_xy(self, time_list):
        """time_list: list of (support, eval_data) for consecutive times.
        Use eval_data.x as features and eval_data.y as labels (already scaled/logged by loader).
        """
        import numpy as np
        feats = []
        labels = []
        for _, eval_data in time_list:
            if not (hasattr(eval_data, 'x') and hasattr(eval_data, 'y')):
                continue
            X = self._to_numpy(eval_data.x)
            y = self._to_numpy(eval_data.y)
            if X is None or y is None:
                continue
            # Flatten to 1D targets
            y = y.reshape(-1)
            # Guard length mismatch
            n = min(len(y), X.shape[0])
            feats.append(X[:n])
            labels.append(y[:n])
        if len(feats) == 0:
            return np.zeros((0, 1), dtype='float32'), np.zeros((0,), dtype='float32')
        X_all = np.concatenate(feats, axis=0)
        y_all = np.concatenate(labels, axis=0)
        # Replace infs; allow NaNs (XGBoost can handle missing)
        X_all = np.nan_to_num(X_all, nan=np.nan, posinf=np.finfo('float32').max/2, neginf=-np.finfo('float32').max/2)
        return X_all, y_all

    # ============ Public API used by trainer ============
    def train_xgboost(self, train_list, val_list):
        X_train, y_train = self._gather_train_xy(train_list)
        X_val, y_val = self._gather_train_xy(val_list)
        if X_train.shape[0] == 0:
            return float('nan'), float('nan')
        # Fit
        if self._use_xgb:
            es = int(os.environ.get("XGB_EARLY_STOPPING_ROUNDS", "20"))
            # Track per-iteration MAE on train/val.
            evals_result = None
            fit_kwargs = {"verbose": False, "eval_set": [(X_train, y_train)]}
            # Set eval metric on the estimator (sklearn wrapper). Use RMSE to align with squared-error objective.
            self._xgb.set_params(eval_metric="rmse")
            if X_val.shape[0]:
                fit_kwargs["eval_set"].append((X_val, y_val))
                if es > 0:
                    self._xgb.set_params(early_stopping_rounds=es)
            self._xgb.fit(X_train, y_train, **fit_kwargs)
            # Pull history from the fitted model
            try:
                evals_result = self._xgb.evals_result()
            except Exception:
                evals_result = None
        else:
            # sklearn GBDT has no eval_set / callbacks
            self._xgb.fit(X_train, y_train)
        # Simple MAE on train/val
        import numpy as np
        train_mae = float(np.mean(np.abs(self._xgb.predict(X_train) - y_train)))
        val_mae = float(np.mean(np.abs(self._xgb.predict(X_val) - y_val))) if X_val.shape[0] else float('nan')

        # Log a compact view of the learning curves if available
        try:
            # XGBoost names eval sets as validation_0, validation_1, ...
            hist_train = evals_result.get("validation_0", {}).get("mae", [])  # type: ignore[name-defined]
            hist_val = evals_result.get("validation_1", {}).get("mae", [])
            if hist_train:
                best_iter = len(hist_train) - 1
                if hist_val:
                    best_iter = int(np.argmin(hist_val))
                print(
                    f"[XGB] rounds={len(hist_train)} "
                    f"train_last={hist_train[-1]:.4f} "
                    f"val_last={(hist_val[-1] if hist_val else float('nan')):.4f} "
                    f"val_best={(min(hist_val) if hist_val else float('nan')):.4f}@{best_iter}"
                )
        except Exception:
            pass
        return train_mae, val_mae

    @torch.no_grad()
    def predict_tensor(self, X: torch.Tensor) -> torch.Tensor:
        if X is None:
            return torch.empty(0, device=self.device)
        x_np = self._to_numpy(X)
        # Predict; convert back to torch
        y = self._xgb.predict(x_np)
        return torch.from_numpy(y).to(X.device).float().view(-1)

    # ============ Compatibility with generic pipeline ============
    def encode(self, support):
        """Fallback path when generic test() calls encode() for predictions.
        Extract fund-node features from the most recent snapshot if a list; else from support directly.
        """
        try:
            if isinstance(support, (list, tuple)) and len(support) > 0:
                g = support[-1]
            else:
                g = support
            if hasattr(g, 'x_dict') and 'fund' in g.x_dict:
                X = g['fund'].x
            elif hasattr(g, 'x'):
                X = g.x
            else:
                X = None
        except Exception:
            X = None
        return X if X is not None else torch.empty(0)

    @torch.no_grad()
    def decode_nclf(self, z):
        # z is a torch.Tensor of features when called via generic path
        if torch.is_tensor(z):
            return self.predict_tensor(z)
        # If some code calls decode_nclf without prior encode features
        return torch.empty(0, device=self.device)
