#!/opt/homebrew/bin/python3.12
import os, sys

import numpy as np
import pandas as pd
from scipy.spatial import Delaunay
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator, RBFInterpolator
from scipy.optimize import root_scalar, minimize
from scipy.spatial import cKDTree
from sklearn.ensemble import GradientBoostingRegressor
import time
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, mean_absolute_percentage_error
import cloudpickle as pickle
# import matplotlib.pyplot as plt

# REGRESSOR = "GBR"
REGRESSOR = "XGBOOST"
# REGRESSOR = "LightGBM"
# REGRESSOR = "CATBOOST"
# REGRESSOR = "RANDOMFOREST"
# REGRESSOR = "EXTRATREES"
def build_regressor(reg_class):
    if reg_class=="GBR":
        from sklearn.ensemble import GradientBoostingRegressor
        return GradientBoostingRegressor(n_estimators=200, max_depth=4, learning_rate=0.1)
    elif reg_class=="XGBOOST":
        from xgboost import XGBRegressor
        return XGBRegressor(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.1,
                objective='reg:squarederror'
            )
    elif reg_class == "LightGBM": 
        from lightgbm import LGBMRegressor
        return LGBMRegressor(n_estimators=200, max_depth=4, learning_rate=0.1, verbose=-1)
    elif reg_class=="CATBOOST":
        from catboost import CatBoostRegressor
        return CatBoostRegressor(iterations=200, depth=4, learning_rate=0.1, verbose=0)
    elif reg_class=="RANDOMFOREST":
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(n_estimators=200, max_depth=4)
    elif reg_class=="EXTRATREES":
        from sklearn.ensemble import ExtraTreesRegressor
        return ExtraTreesRegressor(n_estimators=200, max_depth=4)
    else: 
        sys.exit(f"Unknow regressor class {reg_class}")


class RoundDurationModel:
    """
    Analytical + Gradient Boosting Regressor (GBR) model for predicting LLM serving round duration.

    Analytical model:
        round_duration_pred = max(
            decode_time,
            prefill_time
        ) + overhead
    
    - decode_time = c * N_decode^d * KV^e
    - prefill_time = c0 * N_prefill + c1 * T_prefill
    - N_decode: number of decode requests
    - N_prefill: number of prefill requests
    - T_prefill: total number of tokens in prefill
    - KV: KV cache per request
    - overhead: fixed scheduler overhead

    After fitting the analytical parameters, a Gradient Boosting Regressor
    is trained to learn residuals (differences between actual data and analytical predictions).
    """

    def __init__(self, max_requests, c=1.0, d=0.5, e=0.5, c0=1.0, c1=1.0, overhead=1.0, scheduler_overhead = 8.689/30000):
        self.c = c
        self.d = d
        self.e = e
        self.c0 = c0
        self.c1 = c1
        self.overhead = overhead
        self.regressor = None
        self.median_KV = 0
        self.prefill_tps = None
        self.decode_tps = None
        self.scheduler_overhead = scheduler_overhead
        self.max_requests = max_requests
    def fit(self, df):
        """
        Fit analytical model parameters to the data, then train GBR on residuals.
        
        Parameters
        ----------
        df : pandas.DataFrame
            Must contain columns:
            ['num_decode_requests','num_prefill_requests','prefill_tokens',
             'kv_cache_per_req','round_duration']
        """
        X = df[['num_decode_requests', 'num_prefill_requests', 'prefill_tokens', 'kv_cache_per_req']].to_numpy()
        y = df['round_duration'].to_numpy()

        # ------------------------
        # Step 1: Fit analytical model
        # ------------------------
        def loss(params):
            c, d, e, c0, c1, overhead = params
            N_decode, N_prefill, T_prefill, KV = X.T

            # Compute decode and prefill times
            pred_decode = c * N_decode**d * KV**e
            pred_prefill = c0 * N_prefill + c1 * T_prefill

            # Use max for parallel execution
            pred_total = np.maximum(pred_decode, pred_prefill) + overhead
            return np.mean((y - pred_total)**2)

        print("Fitting analytical model...")
        start_time = time.monotonic()
        res = minimize(
            loss,
            x0=[self.c, self.d, self.e, self.c0, self.c1, self.overhead],
            bounds=[(0, None), (0, None), (0, None), (0, None), (0, None), (0, None)]
        )
        fit_duration = time.monotonic() - start_time
        print(f"Analytical model fitting time: {fit_duration:.2f}s")

        # Save fitted parameters
        self.c, self.d, self.e, self.c0, self.c1, self.overhead = res.x
        print("Fitted analytical parameters:", res.x)

        # ------------------------
        # Step 2: Fit GBR on residuals
        # ------------------------
        pred_analytical = self.predict_analytical(*X.T)
        residuals = y - pred_analytical

        # Compute error metrics
        mae = mean_absolute_error(y, pred_analytical)
        rmse = root_mean_squared_error(y, pred_analytical)
        print(f"MAE of analytical model: {mae:.6f}")
        print(f"RMSE of analytical model: {rmse:.6f}")

        # Train GBR on residuals
        self.regressor = build_regressor(REGRESSOR)
        print(f"Training {REGRESSOR} on residuals...")
        start_time = time.monotonic()
        self.regressor.fit(X, residuals)
        gbr_duration = time.monotonic() - start_time
        print(f"{REGRESSOR} training time: {gbr_duration:.2f}s")
        print(f"{REGRESSOR} trained on residuals")

        # save the median of KV values
        _, _, _, KV = X.T 
        self.median_KV = np.median(KV)

        # ------------------------
        # Step 3: Measure GBR contribution
        # ------------------------

        pred_analytical = self.predict_analytical(*X.T)
        pred_full = pred_analytical + self.regressor.predict(X)

        mae_analytical = mean_absolute_error(y, pred_analytical)
        mae_full = mean_absolute_error(y, pred_full)

        rmse_analytical = root_mean_squared_error(y, pred_analytical)
        rmse_full = root_mean_squared_error(y, pred_full)

        # variance du résidu
        residuals = y - pred_analytical
        residual_pred = self.regressor.predict(X)

        residual_var = np.var(residuals)
        residual_error = np.var(residuals - residual_pred)

        r2_residual = 1 - residual_error / residual_var

        print(f"\n===== {REGRESSOR} Contribution Analysis =====")
        print(f"MAE analytical      \t: {mae_analytical:.6f}")
        print(f"MAE analytical+{REGRESSOR}\t: {mae_full:.6f}")
        print(f"MAE reduction       \t: {(1 - mae_full/mae_analytical)*100:.2f}%")

        print()

        print(f"RMSE analytical     \t: {rmse_analytical:.6f}")
        print(f"RMSE analytical+{REGRESSOR}\t: {rmse_full:.6f}")
        print(f"RMSE reduction      \t: {(1 - rmse_full/rmse_analytical)*100:.2f}%")

        print()

        print(f"Residual variance explained by {REGRESSOR} (R²): {r2_residual:.3f}")
        print("=======================================")

    def predict_analytical(self, N_decode, N_prefill, T_prefill, KV):
        """
        Compute round duration using analytical model.
        Supports both scalars and arrays.
        """
        pred_decode = self.c * N_decode**self.d * KV**self.e
        pred_prefill = self.c0 * N_prefill + self.c1 * T_prefill
        return np.maximum(pred_decode, pred_prefill) + self.overhead
    
    def predict(self, N_decode, N_prefill, T_prefill, KV=None):
        """
        Combined analytical + GBR prediction.
        Can handle scalars or arrays.
        """
      
        # Detect scalar input
        is_scalar = np.isscalar(N_decode)
     
        N_decode = np.atleast_1d(N_decode)
        N_prefill = np.atleast_1d(N_prefill)
        T_prefill = np.atleast_1d(T_prefill)
        if KV is None:
            KV = np.full_like(N_decode, self.median_KV)
        KV = np.atleast_1d(KV)


        X = np.column_stack([N_decode, N_prefill, T_prefill, KV])
        pred_analytical = self.predict_analytical(N_decode, N_prefill, T_prefill, KV)
        correction = self.regressor.predict(X) if self.regressor is not None else 0.0
        result = pred_analytical + correction
        result += self.scheduler_overhead

        # Return scalar if scalar input
        return result[0] if is_scalar else result

    def compute_tps(self, force = False):
        if not force and self.prefill_tps is not None and self.decode_tps is not None:
            return self.prefill_tps, self.decode_tps

        N_decode  = np.zeros(5)
        T_prefill = np.array([128, 256, 512, 1024, 2048])
        N_prefill = np.ones(5)
        KV        = np.full(5, self.median_KV) 


        duration = self.predict(
            N_decode, N_prefill, T_prefill, KV
        )
        self.prefill_tps = float(np.max(T_prefill/duration))

        print(f"{self.prefill_tps=:1.2f} tokens/s")
       
        duration = self.predict(
            1, 0, 0, self.median_KV
        )
        self.decode_tps= float(1/duration)

        print(f"{self.decode_tps=:1.2f} tokens/s")

        return self.prefill_tps, self.decode_tps
    # Compute TPS per decode request:
    # Each decode request corresponds to 1 token, so the total tokens k = N_decode.
    # Throughput per second (TPS) is total tokens / total time: tps = k / t
    # Since decode_tps_per_req = tps / N_decode = k / (t * N_decode) and k = N_decode
    # This simplifies to decode_tps_per_req = 1 / t
    def decode_tps_per_req(self, N_decode, N_prefill, T_prefill, KV=None):
        if N_decode == 0: 
            return None
        else :
            return 1/self.predict(N_decode, N_prefill, T_prefill, KV)
        
    
    def num_decode_requests_vectorized(self, N_prefill, T_prefill, tps_target, N_max=512):
        """
        Compute the required N_decode to reach tps_target, vectorized over arrays.
        
        Parameters
        ----------
        model : callable
            Function predicting round duration: model(N_decode, N_prefill, T_prefill)
        N_prefill : scalar or array-like
        T_prefill : scalar or array-like
        tps_target : scalar or array-like
        N_max : int
            Maximum N_decode to consider

        Returns
        -------
        np.array of required N_decode (first value where 1/round_duration >= tps_target)
        """
        N_prefill = np.atleast_1d(N_prefill)
        T_prefill = np.atleast_1d(T_prefill)
        tps_target = np.atleast_1d(tps_target)
        
        results = []
        N_decode_vals = np.arange(0, N_max+1)

        for npf, tpf, tps in zip(N_prefill, T_prefill, tps_target):
            # Predict round duration for all candidate N_decode values
            N_array = N_decode_vals
            P_array = np.full_like(N_array, npf, dtype=float)
            T_array = np.full_like(N_array, tpf, dtype=float)
            durations = self.predict(N_array, P_array, T_array)

            

            # Find the largest N_decode that achieves target TPS
            idx = np.where(1 / durations >= tps)[0]
            # print(f"{tps_target=} {durations=} {idx=}")
            if len(idx) > 0:
                val= N_array[idx[-1]]
                results.append(val)
            else:
                # If none satisfy target, return -1
                results.append(-100)
        
        return np.array(results)
    

    # Evaluate the concurrency requirement for each TPS value and convert it to integers.
    def max_concurrency_lookup_table(self, model_nb_requests_max=None, step = 0.1):
        
        if model_nb_requests_max is None:
            model_nb_requests_max = self.max_requests
        _, decode_tps = self.compute_tps()
        max_decode_tps = decode_tps + 10
        
        tps_array = np.arange(max_decode_tps, 0, -step)
        
        max_concurrency_lookup_table=np.floor(self.num_decode_requests_vectorized(
            np.zeros(len(tps_array)),
            np.zeros(len(tps_array)),
            tps_array, 
            N_max= model_nb_requests_max
        )).astype(int)
    
        # create a mask of valid (non-None) entries
        valid_mask = max_concurrency_lookup_table >0

        return max_concurrency_lookup_table[valid_mask], tps_array[valid_mask]
    def save_model(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        print(f"Model saved to {path}")

    @staticmethod
    def load_model(path):
        with open(path, 'rb') as f:
            model = pickle.load(f)
        print(f"Model loaded from {path}")
        return model
    
class RoundDurationModelDumb:
    """
    Analytical + Gradient Boosting Regressor (GBR) model for predicting LLM serving round duration.

    Analytical model:
        round_duration_pred = max(
            decode_time,
            prefill_time
        ) + overhead
    
    - decode_time = c * N_decode^d * KV^e
    - prefill_time = c0 * N_prefill + c1 * T_prefill
    - N_decode: number of decode requests
    - N_prefill: number of prefill requests
    - T_prefill: total number of tokens in prefill
    - KV: KV cache per request
    - overhead: fixed scheduler overhead

    After fitting the analytical parameters, a Gradient Boosting Regressor
    is trained to learn residuals (differences between actual data and analytical predictions).
    """

    def __init__(self, c=1.0, d=0.5, e=0.5, c0=1.0, c1=1.0, overhead=1.0):
        self.c = c
        self.d = d
        self.e = e
        self.c0 = c0
        self.c1 = c1
        self.overhead = overhead
        self.regressor = None

    def fit(self, df):
        """
        Fit analytical model parameters to the data, then train GBR on residuals.
        
        Parameters
        ----------
        df : pandas.DataFrame
            Must contain columns:
            ['num_decode_requests','num_prefill_requests','prefill_tokens',
             'kv_cache_per_req','round_duration']
        """
        X = df[['num_decode_requests', 'num_prefill_requests', 'prefill_tokens', 'kv_cache_per_req']].to_numpy()
        y = df['round_duration'].to_numpy()

        # ------------------------
        # Step 1: Fit analytical model
        # ------------------------
        def loss(params):
            c, d, e, c0, c1, overhead = params
            N_decode, N_prefill, T_prefill, KV = X.T

            # Compute decode and prefill times
            pred_decode = 0*N_decode
            pred_prefill =  0*N_prefill

            # Use max for parallel execution
            pred_total = np.maximum(pred_decode, pred_prefill) + overhead
            return np.mean((y - pred_total)**2)

        print("Fitting analytical DUMB model...")
        start_time = time.monotonic()
        res = minimize(
            loss,
            x0=[self.c, self.d, self.e, self.c0, self.c1, self.overhead],
            bounds=[(0, None), (0, None), (0, None), (0, None), (0, None), (0, None)]
        )
        fit_duration = time.monotonic() - start_time
        print(f"Analytical model fitting time: {fit_duration:.2f}s")

        # Save fitted parameters
        self.c, self.d, self.e, self.c0, self.c1, self.overhead = res.x
        print("Fitted analytical parameters:", res.x)

        # ------------------------
        # Step 2: Fit GBR on residuals
        # ------------------------
        pred_analytical = self.predict_analytical(*X.T)
        residuals = y - pred_analytical

        # Compute error metrics
        mae = mean_absolute_error(y, pred_analytical)
        rmse = root_mean_squared_error(y, pred_analytical)
        print(f"MAE of analytical model: {mae:.6f}")
        print(f"RMSE of analytical model: {rmse:.6f}")

        # Train GBR on residuals
        self.regressor = build_regressor(REGRESSOR)
        print(f"Training {REGRESSOR} on residuals...")
        start_time = time.monotonic()
        self.regressor.fit(X, residuals)
        gbr_duration = time.monotonic() - start_time
        print(f"{REGRESSOR} training time: {gbr_duration:.2f}s")
        print(f"{REGRESSOR} trained on residuals")

        # ------------------------
        # Step 3: Measure GBR contribution
        # ------------------------

        pred_analytical = self.predict_analytical(*X.T)
        pred_full = pred_analytical + self.regressor.predict(X)

        mae_analytical = mean_absolute_error(y, pred_analytical)
        mae_full = mean_absolute_error(y, pred_full)

        rmse_analytical = root_mean_squared_error(y, pred_analytical)
        rmse_full = root_mean_squared_error(y, pred_full)

        # variance du résidu
        residuals = y - pred_analytical
        residual_pred = self.regressor.predict(X)

        residual_var = np.var(residuals)
        residual_error = np.var(residuals - residual_pred)

        r2_residual = 1 - residual_error / residual_var

        print(f"\n===== {REGRESSOR} Contribution Analysis =====")
        print(f"MAE analytical      \t: {mae_analytical:.6f}")
        print(f"MAE analytical+{REGRESSOR}\t: {mae_full:.6f}")
        print(f"MAE reduction       \t: {(1 - mae_full/mae_analytical)*100:.2f}%")

        print()

        print(f"RMSE analytical     \t: {rmse_analytical:.6f}")
        print(f"RMSE analytical+{REGRESSOR}\t: {rmse_full:.6f}")
        print(f"RMSE reduction      \t: {(1 - rmse_full/rmse_analytical)*100:.2f}%")

        print()

        print(f"Residual variance explained by {REGRESSOR} (R²): {r2_residual:.3f}")
        print("=======================================")


    def predict_analytical(self, N_decode, N_prefill, T_prefill, KV):
        """
        Compute round duration using analytical model.
        Supports both scalars and arrays.
        """
        pred_decode = 0*N_decode
        pred_prefill = 0*N_prefill
        return np.maximum(pred_decode, pred_prefill) + self.overhead
    
    def predict(self, N_decode, N_prefill, T_prefill, KV):
        """
        Combined analytical + GBR prediction.
        Can handle scalars or arrays.
        """
    
         # Detect scalar input
        is_scalar = np.isscalar(N_decode)

        N_decode = np.atleast_1d(N_decode)
        N_prefill = np.atleast_1d(N_prefill)
        T_prefill = np.atleast_1d(T_prefill)
        KV = np.atleast_1d(KV)

        X = np.column_stack([N_decode, N_prefill, T_prefill, KV])
        pred_analytical = self.predict_analytical(N_decode, N_prefill, T_prefill, KV)
        correction = self.regressor.predict(X) if self.regressor is not None else 0.0
        result = pred_analytical + correction

        # Return scalar if scalar input
        return result[0] if is_scalar else result

        
    def save_model(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        print(f"Model saved to {path}")

    @staticmethod
    def load_model(path):
        with open(path, 'rb') as f:
            model = pickle.load(f)
        print(f"Model loaded from {path}")
        return model
    


class RoundDurationModelNoKV:
    """
    Analytical + Gradient Boosting Regressor (GBR) model for predicting LLM serving round duration
    WITHOUT KV cache as a parameter.

    Analytical model:
        round_duration_pred = max(
            decode_time,
            prefill_time
        ) + overhead
    
    - decode_time = c * N_decode^d
    - prefill_time = c0 * N_prefill + c1 * T_prefill
    - N_decode: number of decode requests
    - N_prefill: number of prefill requests
    - T_prefill: total number of tokens in prefill
    - overhead: fixed scheduler overhead
    """

    def __init__(self, c=1.0, d=0.5, c0=1.0, c1=1.0, overhead=1.0):
        self.c = c
        self.d = d
        self.c0 = c0
        self.c1 = c1
        self.overhead = overhead
        self.regressor = None

    def fit(self, df):
        """
        Fit analytical model parameters, then train GBR on residuals.

        Expected columns:
        ['num_decode_requests','num_prefill_requests','prefill_tokens','round_duration']
        """
        X = df[['num_decode_requests', 'num_prefill_requests', 'prefill_tokens']].to_numpy()
        y = df['round_duration'].to_numpy()

        # ------------------------
        # Step 1: Fit analytical model
        # ------------------------
        def loss(params):
            c, d, c0, c1, overhead = params
            N_decode, N_prefill, T_prefill = X.T

            pred_decode = c * N_decode**d
            pred_prefill = c0 * N_prefill + c1 * T_prefill

            pred_total = np.maximum(pred_decode, pred_prefill) + overhead
            return np.mean((y - pred_total)**2)

        print("Fitting analytical model (no KV)...")
        start_time = time.monotonic()

        res = minimize(
            loss,
            x0=[self.c, self.d, self.c0, self.c1, self.overhead],
            bounds=[(0, None)] * 5
        )

        fit_duration = time.monotonic() - start_time
        print(f"Analytical fit time: {fit_duration:.2f}s")

        self.c, self.d, self.c0, self.c1, self.overhead = res.x
        print("Fitted parameters:", res.x)

        # ------------------------
        # Step 2: Fit GBR on residuals
        # ------------------------
        pred_analytical = self.predict_analytical(*X.T)
        residuals = y - pred_analytical

        mae = mean_absolute_error(y, pred_analytical)
        rmse = root_mean_squared_error(y, pred_analytical)

        print(f"MAE (analytical): {mae:.6f}")
        print(f"RMSE (analytical): {rmse:.6f}")

        self.regressor = build_regressor(REGRESSOR)

        print(f"Training {REGRESSOR}on residuals...")
        start_time = time.monotonic()
        self.regressor.fit(X, residuals)
        print(f"{REGRESSOR} training time: {time.monotonic() - start_time:.2f}s")

        # ------------------------
        # Step 3: Measure GBR contribution
        # ------------------------

        pred_analytical = self.predict_analytical(*X.T)
        pred_full = pred_analytical + self.regressor.predict(X)

        mae_analytical = mean_absolute_error(y, pred_analytical)
        mae_full = mean_absolute_error(y, pred_full)

        rmse_analytical = root_mean_squared_error(y, pred_analytical)
        rmse_full = root_mean_squared_error(y, pred_full)

        # variance du résidu
        residuals = y - pred_analytical
        residual_pred = self.regressor.predict(X)

        residual_var = np.var(residuals)
        residual_error = np.var(residuals - residual_pred)

        r2_residual = 1 - residual_error / residual_var

        print(f"\n===== {REGRESSOR} Contribution Analysis =====")
        print(f"MAE analytical      \t: {mae_analytical:.6f}")
        print(f"MAE analytical+{REGRESSOR}\t: {mae_full:.6f}")
        print(f"MAE reduction       \t: {(1 - mae_full/mae_analytical)*100:.2f}%")

        print()

        print(f"RMSE analytical     \t: {rmse_analytical:.6f}")
        print(f"RMSE analytical+{REGRESSOR}\t: {rmse_full:.6f}")
        print(f"RMSE reduction      \t: {(1 - rmse_full/rmse_analytical)*100:.2f}%")

        print()

        print(f"Residual variance explained by {REGRESSOR} (R²): {r2_residual:.3f}")
        print("=======================================")

    def predict_analytical(self, N_decode, N_prefill, T_prefill):
        """
        Analytical prediction (no KV).
        """
        pred_decode = self.c * N_decode**self.d
        pred_prefill = self.c0 * N_prefill + self.c1 * T_prefill
        return np.maximum(pred_decode, pred_prefill) + self.overhead

    def predict(self, N_decode, N_prefill, T_prefill):
        """
        Analytical + GBR prediction.
        """


        # Detect scalar input
        is_scalar = np.isscalar(N_decode)

        N_decode = np.atleast_1d(N_decode)
        N_prefill = np.atleast_1d(N_prefill)
        T_prefill = np.atleast_1d(T_prefill)
  

        X = np.column_stack([N_decode, N_prefill, T_prefill])
        pred_analytical = self.predict_analytical(N_decode, N_prefill, T_prefill)
        correction = self.regressor.predict(X) if self.regressor is not None else 0.0
        result = pred_analytical + correction

        # Return scalar if scalar input
        return result[0] if is_scalar else result

    def save_model(self, path):
        """Save model to disk."""
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        print(f"Model saved to {path}")

    @staticmethod
    def load_model(path):
        """Load model from disk."""
        with open(path, 'rb') as f:
            model = pickle.load(f)
        print(f"Model loaded from {path}")
        return model
    
    # def save_model(self, filepath):
    #     """Save the model (analytical params + trained GBR) to a file."""
    #     model_dict = {
    #         'c': self.c,
    #         'd': self.d,
    #         'e': self.e,
    #         'c0': self.c0,
    #         'c1': self.c1,
    #         'overhead': self.overhead,
    #         'gbr': self.regressor
    #     }
    #     joblib.dump(model_dict, filepath)
    #     print(f"Model saved to {filepath}")
    # @classmethod
    # def load_model(cls, filepath):
    #     """Load a RoundDurationModel from file."""
    #     model_dict = joblib.load(filepath)
    #     model = cls(
    #         c=model_dict['c'],
    #         d=model_dict['d'],
    #         e=model_dict['e'],
    #         c0=model_dict['c0'],
    #         c1=model_dict['c1'],
    #         overhead=model_dict['overhead']
    #     )
    #     model.gbr = model_dict['gbr']
    #     print(f"Model loaded from {filepath}")
    #     return model
    
class DecodeReqInterpolator4D:
    def __init__(self, df: pd.DataFrame):
        filtered_df = df.dropna(subset=['decode_tps_per_req', 'mean_num_decode_requests'])
        self.points = filtered_df[['mean_num_prefill_requests', 'avg_prefill_tokens', 'kv_cache_per_req', 'decode_tps_per_req']].to_numpy()
        self.values = filtered_df['mean_num_decode_requests'].to_numpy()
        self.linear = LinearNDInterpolator(self.points, self.values)
        self.nn = NearestNDInterpolator(self.points, self.values)
        self.hull = Delaunay(self.points)

    def __call__(self, num_prefill_requests, tot_prefill_tokens, kv_cache, target_decode_tps):
        query = np.c_[num_prefill_requests, tot_prefill_tokens, kv_cache, target_decode_tps]
        inside = self.hull.find_simplex(query) >= 0
        result = np.full(len(query), np.nan)
        result[inside] = self.linear(query[inside])
        outside = ~inside
        if np.any(outside):
            result[outside] = self.nn(query[outside])
        return np.maximum(2,result)

class DecodeReqInterpolator:
    """
    Interpolates mean_num_decode_requests based on:
      - mean_num_prefill_requests
      - avg_prefill_tokens
      - target decode_tps_per_req
    """
    def __init__(self, df: pd.DataFrame):
        # Filtrer les lignes valides
        filtered_df = df.dropna(subset=['decode_tps_per_req', 'mean_num_decode_requests'])
        # Points dans l'espace 3D (mean_num_prefill_requests, avg_prefill_tokens, decode_tps_per_req)
        self.points = filtered_df[['mean_num_prefill_requests', 'avg_prefill_tokens', 'decode_tps_per_req']].to_numpy()
        # Valeurs correspondantes: mean_num_decode_requests
        self.values = filtered_df['mean_num_decode_requests'].to_numpy()
        
        # Interpolateurs
        self.linear = LinearNDInterpolator(self.points, self.values)
        self.nn = NearestNDInterpolator(self.points, self.values)
        self.hull = Delaunay(self.points)
        
    def __call__(self, num_prefill_requests, tot_prefill_tokens, target_decode_tps):
        # Construire la requête 3D
        query = np.c_[num_prefill_requests, tot_prefill_tokens, target_decode_tps]
        
        # Identifier les points à l'intérieur du convex hull
        inside = self.hull.find_simplex(query) >= 0
        
        result = np.full(len(query), np.nan)
        
        # Interpolation linéaire pour les points à l'intérieur
        result[inside] = self.linear(query[inside])
        
        # Interpolation nearest neighbor pour les points à l'extérieur
        outside = ~inside
        if np.any(outside):
            result[outside] = self.nn(query[outside])
        
        return np.maximum(2,result) #element wise mininum
    
class PrefillReqInterpolator:
    """
    Interpolates mean_num_prefill_requests based on:
      - mean_num_decode_requests
      - avg_prefill_tokens
      - target prefill_tps_per_req
    """
    def __init__(self, df: pd.DataFrame):
        # Filtrer les lignes valides
        filtered_df = df.dropna(subset=['prefill_tps_per_req', 'mean_num_prefill_requests'])
        
        # Points dans l'espace 3D (mean_num_decode_requests, avg_prefill_tokens, prefill_tps_per_req)
        self.points = filtered_df[['mean_num_decode_requests', 'avg_prefill_tokens', 'prefill_tps_per_req']].to_numpy()
        # Valeurs correspondantes: mean_num_prefill_requests
        self.values = filtered_df['mean_num_prefill_requests'].to_numpy()
        
        # Interpolateurs
        self.linear = LinearNDInterpolator(self.points, self.values)
        self.nn = NearestNDInterpolator(self.points, self.values)
        self.hull = Delaunay(self.points)
        
    def __call__(self, num_decode_requests, tot_prefill_tokens, target_prefill_tps):
        # Construire la requête 3D
        query = np.c_[num_decode_requests, tot_prefill_tokens, target_prefill_tps]
        
        # Identifier les points à l'intérieur du convex hull
        inside = self.hull.find_simplex(query) >= 0
        
        result = np.full(len(query), np.nan)
        
        # Interpolation linéaire pour les points à l'intérieur
        result[inside] = self.linear(query[inside])
        
        # Interpolation nearest neighbor pour les points à l'extérieur
        outside = ~inside
        if np.any(outside):
            result[outside] = self.nn(query[outside])
        
        return result
class DecodeRequestSolver:
    def __init__(self, df):
        # points = [mean_num_decode_requests, mean_num_prefill_requests, avg_prefill_tokens]
        # values = decode_tps_per_req
        self.points = df[['mean_num_decode_requests', 'mean_num_prefill_requests', 'avg_prefill_tokens']].to_numpy()
        self.values = df['decode_tps_per_req'].to_numpy()
        # linear interpolation inside convex hull
        self.interp = LinearNDInterpolator(self.points, self.values)

    def solve(self, target_tps, mean_num_prefill, avg_prefill):
        """
        Returns estimated mean_num_decode_requests that yields target_tps for given
        mean_num_prefill_requests and avg_prefill_tokens
        """
        def func(mean_num_decode):
            point = np.array([[mean_num_decode, mean_num_prefill, avg_prefill]])
            tps = self.interp(point)[0]
            if np.isnan(tps):
                # fallback: nearest neighbor outside convex hull
                dist = np.sum((self.points - point)**2, axis=1)
                nearest_idx = np.argmin(dist)
                tps = self.values[nearest_idx]
            return tps - target_tps

        # bounds: at least 0 decode requests, max = e.g., 128
        sol = root_scalar(func, bracket=[0, 128], method='bisect')
        if sol.converged:
            return sol.root
        else:
            return np.nan
        
class PrefillRequestSolver:
    def __init__(self, df):
        # points = [mean_num_decode_requests, mean_num_prefill_requests, avg_prefill_tokens]
        # values = prefill_tps_per_req
        self.points = df[['mean_num_decode_requests', 'mean_num_prefill_requests', 'avg_prefill_tokens']].to_numpy()
        self.values = df['prefill_tps_per_req'].to_numpy()
        self.interp = LinearNDInterpolator(self.points, self.values)

    def solve(self, target_tps, mean_num_decode, avg_prefill):
        """
        Returns estimated mean_num_prefill_requests that yields target_tps for given
        mean_num_decode_requests and avg_prefill_tokens
        """
        def func(mean_num_prefill):
            point = np.array([[mean_num_decode, mean_num_prefill, avg_prefill]])
            tps = self.interp(point)[0]
            if np.isnan(tps):
                # fallback: nearest neighbor outside convex hull
                dist = np.sum((self.points - point)**2, axis=1)
                nearest_idx = np.argmin(dist)
                tps = self.values[nearest_idx]
            return tps - target_tps

        # bounds: at least 0 prefill requests, max = e.g., 128
        sol = root_scalar(func, bracket=[0, 128], method='bisect')
        if sol.converged:
            return sol.root
        else:
            return np.nan
        


class TPSInterpolator4D:
    def __init__(self, df, target_col: str):
        """
        df: DataFrame with columns 
            ['mean_num_decode_requests', 'mean_num_prefill_requests', 'avg_prefill_tokens', 'kv_cache_per_req', target_col]
        target_col: 'decode_tps_per_req', 'prefill_tps_per_req' or 'mean_round_duration'
        """
        df = df.dropna(subset=[target_col])
        self.points = df[['mean_num_decode_requests', 'mean_num_prefill_requests', 'avg_prefill_tokens', 'kv_cache_per_req']].to_numpy()
        self.values = df[target_col].to_numpy()
        self.target_col = target_col

        self.hull = Delaunay(self.points)
        self.lin_interp = LinearNDInterpolator(self.points, self.values)
        self.tree = cKDTree(self.points)  # fallback nearest neighbor

    def is_inside_hull(self, query_points: np.ndarray) -> np.ndarray:
        return self.hull.find_simplex(query_points) >= 0

    def __call__(self, num_decode_req, num_prefill_req, tot_prefill_tokens, kv_cache):
        scalar_input = np.isscalar(num_decode_req)
        num_decode_req = np.atleast_1d(num_decode_req)
        num_prefill_req = np.atleast_1d(num_prefill_req)
        tot_prefill_tokens = np.atleast_1d(tot_prefill_tokens)
        kv_cache = np.atleast_1d(kv_cache)

        query_points = np.column_stack([num_decode_req, num_prefill_req, tot_prefill_tokens, kv_cache])
        inside = self.is_inside_hull(query_points)
        # print(f"{query_points=} {inside=}")
        result = np.empty(len(query_points))

        if np.any(inside):
            result[inside] = self.lin_interp(query_points[inside])

        if np.any(~inside):
            closest_idx = self.tree.query(query_points[~inside])[1]
            result[~inside] = self.values[closest_idx]

        return result[0] if scalar_input else result

class TPSInterpolator3D:
    def __init__(self, df, target_col: str):
        """
        csv_file: path to CSV with columns 
                  ['mean_num_decode_requests', 'mean_num_prefill_requests', 'avg_prefill_tokens', target_col]
        target_col: 'decode_tps_per_req' or 'prefill_tps_per_req'
        """
        # Remove rows with NA in target_col
        df = df.dropna(subset=[target_col])
        self.points = df[['mean_num_decode_requests', 'mean_num_prefill_requests', 'avg_prefill_tokens']].to_numpy()
        self.values = df[target_col].to_numpy()
        self.target_col = target_col

        # Convex hull for inside/outside test
        self.hull = Delaunay(self.points)

        # Linear interpolator
        self.lin_interp = LinearNDInterpolator(self.points, self.values)

        # RBF interpolator
        self.rbf_interp = RBFInterpolator(self.points, self.values, neighbors=min(10, len(self.values)), degree=-1)

    def is_inside_hull(self, query_points: np.ndarray) -> np.ndarray:
        """Return a boolean array, True if point is inside convex hull"""
        return self.hull.find_simplex(query_points) >= 0

    def __call__(self, num_decode_req, num_prefill_req, tot_prefill_tokens):
        """Query interpolated values for arrays or scalars"""
        # Convert scalars to 1D arrays
        scalar_input = np.isscalar(num_decode_req)
        num_decode_req = np.atleast_1d(num_decode_req)
        num_prefill_req = np.atleast_1d(num_prefill_req)
        tot_prefill_tokens = np.atleast_1d(tot_prefill_tokens)

        query_points = np.column_stack([num_decode_req, num_prefill_req, tot_prefill_tokens])
        inside = self.is_inside_hull(query_points)
        result = np.empty(len(query_points))

        if np.any(inside):
            result[inside] = self.lin_interp(query_points[inside])
        # if np.any(~inside):
        #     result[~inside] = self.rbf_interp(query_points[~inside])

        if np.any(~inside):
                # Find closest point inside the hull
                tree = cKDTree(self.points)
                closest_idx = tree.query(query_points[~inside])[1]
                result[~inside] = self.values[closest_idx]

        # Return scalar if scalar input
        return result[0] if scalar_input else result


def scheduler_models(csv_file):
    if not os.path.exists(csv_file):
        sys.exit(f"Error: file '{csv_file}' does not exist. Exiting.")
    df = pd.read_csv(csv_file)
    
    # Interpolating decode TPS
    prefill_penalty3D = TPSInterpolator3D(df, "mean_round_duration")
    decode_req_interp = DecodeReqInterpolator(df)
    decode_tps_interp = TPSInterpolator3D(df, "decode_tps_per_req")

    return prefill_penalty3D, decode_req_interp, decode_tps_interp



def scheduler_models_all(csv_file, csv_train, save_path=None, save_path_no_KV=None):
    if not os.path.exists(csv_file):
        sys.exit(f"Error: file '{csv_file}' does not exist. Exiting.")
    df = pd.read_csv(csv_file)
    
    # Interpolating decode TPS
    prefill_penalty = TPSInterpolator4D(df, "mean_round_duration")
    prefill_penalty3D = TPSInterpolator3D(df, "mean_round_duration")
    decode_req_interp = DecodeReqInterpolator4D(df)
    decode_tps_interp = TPSInterpolator4D(df, "decode_tps_per_req")
    


    df_train = None
    if save_path is None or not os.path.exists(save_path):
        round_duration = RoundDurationModel()
        if not os.path.exists(csv_train):
            sys.exit(f"Error: file '{csv_train}' does not exist. Exiting.")
        df_train = pd.read_csv(csv_train)
        round_duration.fit(df_train)
        if save_path is not None:
            round_duration.save_model(save_path)
    else:
        round_duration = RoundDurationModel.load_model(save_path)
   

    # Compute TPS per decode request:
    # Each decode request corresponds to 1 token, so the total tokens k = N_decode.
    # Throughput per second (TPS) is total tokens / total time: tps = k / t
    # Since decode_tps_per_req = tps / N_decode = k / (t * N_decode) and k = N_decode
    # This simplifies to decode_tps_per_req = 1 / t
    def decode_tps_per_req(N_decode, N_prefill, T_prefill, KV):
        if N_decode == 0: 
            return None
        else :
            return 1/round_duration.predict(N_decode, N_prefill, T_prefill, KV)

    
    if save_path_no_KV is None or not os.path.exists(save_path_no_KV):
        round_duration_no_KV = RoundDurationModelNoKV()
        if not os.path.exists(csv_train):
            sys.exit(f"Error: file '{csv_train}' does not exist. Exiting.")
        df_train = df_train if df_train is not None else pd.read_csv(csv_train)
        round_duration_no_KV.fit(df_train)
        if save_path_no_KV is not None:
            round_duration_no_KV.save_model(save_path_no_KV)
    else:
        round_duration_no_KV = RoundDurationModelNoKV.load_model(save_path_no_KV)

    def make_num_decode_function(model):
        """
        Returns a function N_decode(round_duration_target, N_prefill, T_prefill, KV)
        that estimates the number of decode requests using the captured model.
        """
        
        def num_decode_requests_vectorized(N_prefill, T_prefill, tps_target, N_max=128):
            """
            Compute the required N_decode to reach tps_target, vectorized over arrays.
            
            Parameters
            ----------
            model : callable
                Function predicting round duration: model(N_decode, N_prefill, T_prefill)
            N_prefill : scalar or array-like
            T_prefill : scalar or array-like
            tps_target : scalar or array-like
            N_max : int
                Maximum N_decode to consider

            Returns
            -------
            np.array of required N_decode (first value where 1/round_duration >= tps_target)
            """
            N_prefill = np.atleast_1d(N_prefill)
            T_prefill = np.atleast_1d(T_prefill)
            tps_target = np.atleast_1d(tps_target)
            
            results = []
            N_decode_vals = np.arange(0, N_max+1)

            for npf, tpf, tps in zip(N_prefill, T_prefill, tps_target):
                # Predict round duration for all candidate N_decode values
                N_array = N_decode_vals
                P_array = np.full_like(N_array, npf, dtype=float)
                T_array = np.full_like(N_array, tpf, dtype=float)
                durations = model.predict(N_array, P_array, T_array)

                print(f"{tps_target=} {durations=}")

                # Find the largest N_decode that achieves target TPS
                idx = np.where(1 / durations >= tps)[0]
                if len(idx) > 0:
                    results.append(N_array[idx[-1]])
                else:
                    # If none satisfy target, return N_max
                    results.append(N_max)
            
            return np.array(results)
        return num_decode_requests_vectorized
    
    return prefill_penalty, prefill_penalty3D, decode_req_interp, decode_tps_interp, round_duration.predict, round_duration_no_KV.predict, round_duration.predict_analytical, decode_tps_per_req, make_num_decode_function(round_duration_no_KV)

def train_dumb_model(csv_train, save_path, fraction=1):
    if fraction !=1 :
            save_path = save_path.replace(".pkl", f"_{fraction}.pkl")
       
    if not os.path.exists(save_path):
        round_duration = RoundDurationModelDumb()
        if not os.path.exists(csv_train):
            sys.exit(f"Error: file '{csv_train}' does not exist. Exiting.")
        df_train = pd.read_csv(csv_train)
        df_train = df_train.sample(frac=fraction, random_state=42)
        round_duration.fit(df_train)    
        round_duration.save_model(save_path)
    else:
        round_duration = RoundDurationModel.load_model(save_path)

    return round_duration.predict, round_duration
def print_full(df):
    with pd.option_context(
        "display.max_rows", None,
        "display.max_columns", None,
        "display.width", None,
        "display.max_colwidth", None
    ):
        print(df)

def test_training_size(regressor, dumb=False):
    print(f"Using {regressor} {dumb=}")
    global REGRESSOR
    REGRESSOR = regressor
    df_test = pd.read_csv("monitoring-monotonic-test.csv")

    N_decode = df_test["num_decode_requests"].to_numpy()
    N_prefill = df_test["num_prefill_requests"].to_numpy()
    T_prefill = df_test["prefill_tokens"].to_numpy()
    KV = df_test["kv_cache_per_req"].to_numpy()

    y_true = df_test["round_duration"].to_numpy()

    csv_train = "monitoring-monotonic-train.csv"

    output_df = pd.DataFrame(
        columns=["regressor", "fraction", "nb_xp", "inference_duration", "MAE", "RMSE"]
    )

    fractions = np.logspace(np.log10(0.001), np.log10(1.0), 20)
    for fraction in fractions:
        fraction = round(fraction, 3) 
        save_path = f"pickled_model/duration_model_{regressor}_{fraction}.pkl"
        df_train = pd.read_csv(csv_train)
        subset = df_train.sample(frac=fraction, random_state=42)
        
        if dumb: 
            _, round_duration = train_dumb_model(csv_train, f"pickled_model/duration_model_dumb_{regressor}.pkl", fraction=fraction)
        else:
            if not os.path.exists(save_path):
                round_duration = RoundDurationModel()

                df_train = pd.read_csv(csv_train)
                subset = df_train.sample(frac=fraction, random_state=42)

                round_duration.fit(subset)
                round_duration.save_model(save_path)

            else:
                round_duration = RoundDurationModel.load_model(save_path)

        start = time.monotonic()

        y_pred_regressor = round_duration.predict(
            N_decode, N_prefill, T_prefill, KV
        )

        duration_regressor = time.monotonic() - start

        mae_regressor = mean_absolute_error(y_true, y_pred_regressor)
        rmse_regressor = root_mean_squared_error(y_true, y_pred_regressor)

        output_df.loc[len(output_df)] = [
            regressor,
            fraction,
            len(subset),
            duration_regressor,
            mae_regressor,
            rmse_regressor,
        ]

    print_full(output_df)

    if dumb :
        output_df.to_csv(f"fraction_regressor_csv/fraction_dumb_{regressor}.csv", index=False)
    else:
        output_df.to_csv(f"fraction_regressor_csv/fraction_{regressor}.csv", index=False)
    
   # MAE plot
    plt.figure(figsize=(6,4))
    plt.plot(output_df['fraction'], output_df['MAE'], marker='o', linestyle='-')
    plt.xlabel("Fraction of Training Data")
    plt.ylabel("MAE")
    plt.title(f"MAE vs Training Fraction ({regressor})")
    plt.grid(True)
    if dumb: 
        plt.savefig(f"figs/MAE_vs_fraction_dumb_{regressor}.pdf")
    else:
        plt.savefig(f"figs/MAE_vs_fraction_{regressor}.pdf")
    plt.close()

    # RMSE plot
    plt.figure(figsize=(6,4))
    plt.plot(output_df['fraction'], output_df['RMSE'], marker='o', linestyle='-')
    plt.xlabel("Fraction of Training Data")
    plt.ylabel("RMSE")
    plt.title(f"RMSE vs Training Fraction ({regressor})")
    plt.grid(True)
    if dumb:
        plt.savefig(f"figs/RMSE_vs_fraction_dumb_{regressor}.pdf")
    else:
        plt.savefig(f"figs/RMSE_vs_fraction_{regressor}.pdf")
    plt.close()
    

def test_model():
    prefill_penalty, prefill_penalty3D, decode_req_interp, decode_tps_interp, round_duration, round_duration_no_KV, round_duration_analytical, decode_tps_per_req, num_decode_req = scheduler_models_all("tps_data.csv", "monitoring-monotonic-train.csv", save_path="duration_model.pkl", save_path_no_KV="duration_model_no_KV.pkl")

    df_test = pd.read_csv("monitoring-monotonic-test.csv")

    N_decode = df_test["num_decode_requests"].to_numpy()
    N_prefill = df_test["num_prefill_requests"].to_numpy()
    T_prefill = df_test["prefill_tokens"]
    KV = df_test["kv_cache_per_req"].to_numpy()

    y_true = df_test["round_duration"].to_numpy()
    start = time.monotonic()
    y_pred_gbr = round_duration(N_decode, N_prefill, T_prefill, KV)
    duration_gbr = time.monotonic() - start
    mae_gbr = mean_absolute_error(y_true, y_pred_gbr)
    rmse_gbr = root_mean_squared_error(y_true, y_pred_gbr)


    start = time.monotonic()
    y_pred_gbr_no_KV = round_duration_no_KV(N_decode, N_prefill, T_prefill)
    duration_gbr_no_KV = time.monotonic() - start
    mae_gbr_no_KV = mean_absolute_error(y_true, y_pred_gbr_no_KV)
    rmse_gbr_no_KV= root_mean_squared_error(y_true, y_pred_gbr_no_KV)

    start = time.monotonic()
    y_pred_interp = prefill_penalty(N_decode, N_prefill, T_prefill, KV)
    duration_intrep = time.monotonic() - start
    mask = ~np.isnan(y_pred_interp)
    mae_interp = mean_absolute_error(y_true[mask], y_pred_interp[mask])
    rmse_interp = root_mean_squared_error(y_true[mask], y_pred_interp[mask])
    
    start = time.monotonic()
    y_pred_interp = prefill_penalty3D(N_decode, N_prefill, T_prefill)
    duration_intrep_no_KV = time.monotonic() - start
    mask = ~np.isnan(y_pred_interp)
    mae_interp_no_KV = mean_absolute_error(y_true[mask], y_pred_interp[mask])
    rmse_interp_no_KV = root_mean_squared_error(y_true[mask], y_pred_interp[mask])


    print(f"{duration_gbr=} {duration_gbr_no_KV=} {duration_intrep=} {duration_intrep_no_KV=}")

    print(f"{mae_gbr=} {mae_gbr_no_KV=} {mae_interp=} {mae_interp_no_KV=}")
    print(f"{rmse_gbr=} {rmse_gbr_no_KV=} {rmse_interp=} {rmse_interp_no_KV=}")

    dumb_model, _ = train_dumb_model("monitoring-monotonic-train.csv", "dumb.pkl")
    start = time.monotonic()
    y_pred_dumb = dumb_model(N_decode, N_prefill, T_prefill, KV)
    duration_dumb = time.monotonic() - start
    mae_dumb = mean_absolute_error(y_true, y_pred_dumb)
    rmse_dumb = root_mean_squared_error(y_true, y_pred_dumb)
    print(f"{duration_dumb=} {mae_dumb=} {rmse_dumb=}")


    max_decode_tps = 0
    for i in range(0, 2000):
        KV = i * 10_000
        for n in range(1,2):
            tps = decode_tps_per_req(n, 0, 0, KV)  # 1 decode, 0 prefill, 0 tokens
            if tps is not None:  # skip the None case
                max_decode_tps = max(max_decode_tps, tps)

    print(f"Max decode TPS per request = {max_decode_tps}")


    # Bornes connues
    N_decode_vals = np.arange(0, 129)        # 0 à 128
    N_prefill_vals = np.arange(0, 129)       # 0 à 128
    T_prefill_vals = np.arange(0, 2049, 64)  # 0 à 2048, pas de 64 tokens

    min_pred = float('inf')
    max_pred = float('-inf')
    min_point = None
    max_point = None

    from tqdm import tqdm
    for N in tqdm(N_decode_vals, desc="N_decode loop"):
        for P in N_prefill_vals:
            # Create arrays of the same size as T_prefill_vals
            N_array = np.full_like(T_prefill_vals, fill_value=N, dtype=float)
            P_array = np.full_like(T_prefill_vals, fill_value=P, dtype=float)

            # Call the model once for all T_prefill_vals
            preds = round_duration_no_KV(N_array, P_array, T_prefill_vals)
            preds = np.atleast_1d(preds)

            # Find min and max
            idx_min = np.argmin(preds)
            idx_max = np.argmax(preds)

            if preds[idx_min] < min_pred:
                min_pred = preds[idx_min]
                min_point = (N_array[idx_min], P_array[idx_min], T_prefill_vals[idx_min])

            if preds[idx_max] > max_pred:
                max_pred = preds[idx_max]
                max_point = (N_array[idx_max], P_array[idx_max], T_prefill_vals[idx_max])


    print(f"Min prediction: {min_pred} at {min_point}")
    print(f"Max prediction: {max_pred} at {max_point}")
    print(f"{1/round_duration_no_KV(1,0,0)=}")

    print(f"{1/round_duration_no_KV(2,0,0)=}")

    max_decode_tps = 1/round_duration_no_KV(2,0,0)

    step = 0.1
    tps_array = np.arange(108, 0, -step)

    print(tps_array)

    # Evaluate the concurrency requirement for each TPS value and convert it to integers.
    max_concurrency_lookup_table = np.floor(
        num_decode_req(
            np.zeros(len(tps_array)),
            np.zeros(len(tps_array)),
            tps_array
        )
    ).astype(int)

    print(f"{max_concurrency_lookup_table=}")
    # Convert to DataFrame
    df_lookup = pd.DataFrame({
        "TPS_target": tps_array,
        "Max_concurrency": max_concurrency_lookup_table
    })

    # Print the full table
    pd.set_option('display.max_rows', None)  # show all rows
    print(df_lookup)

    # Optionally save to CSV
    df_lookup.to_csv("max_concurrency_lookup_table.csv", index=False)
  
    import matplotlib.pyplot as plt

    # Plot TPS vs Max Concurrency
    plt.figure(figsize=(8, 5))
    plt.plot(df_lookup["TPS_target"], df_lookup["Max_concurrency"], marker='o', linestyle='-')
    plt.xlabel("Target Decode TPS")
    plt.ylabel("Max Concurrency (N_decode)")
    plt.title("Max Concurrency vs Target Decode TPS")
    plt.grid(True)
    plt.show()
    


    N_dec= 127
    pre_t= 256
    n_pref = 1
    print("KV,Duration,Analytic")
    for i in range(1,200):
        KV = i*100_000
        duration_gbr = round_duration(N_dec,n_pref, pre_t, KV)
        duration_analytic = round_duration_analytical(N_dec, pre_t, n_pref, KV)
        print(f"{KV},{duration_gbr},{duration_analytic}")


    df_test = pd.read_csv("monitoring-monotonic-test.csv")

    N_decode = df_test["num_decode_requests"].to_numpy()
    N_prefill = df_test["num_prefill_requests"].to_numpy()
    T_prefill = df_test["prefill_tokens"]
    KV = df_test["kv_cache_per_req"].to_numpy()

    y_true = df_test["round_duration"].to_numpy()
    start = time.monotonic()
    y_pred_gbr = round_duration(N_decode, N_prefill, T_prefill, KV)
    duration_gbr = time.monotonic() - start
    mae_gbr = mean_absolute_error(y_true, y_pred_gbr)
    rmse_gbr = root_mean_squared_error(y_true, y_pred_gbr)


    start = time.monotonic()
    y_pred_gbr_no_KV = round_duration_no_KV(N_decode, N_prefill, T_prefill)
    duration_gbr_no_KV = time.monotonic() - start
    mae_gbr_no_KV = mean_absolute_error(y_true, y_pred_gbr_no_KV)
    rmse_gbr_no_KV= root_mean_squared_error(y_true, y_pred_gbr_no_KV)

    start = time.monotonic()
    y_pred_interp = prefill_penalty(N_decode, N_prefill, T_prefill, KV)
    duration_intrep = time.monotonic() - start
    mask = ~np.isnan(y_pred_interp)
    mae_interp = mean_absolute_error(y_true[mask], y_pred_interp[mask])
    rmse_interp = root_mean_squared_error(y_true[mask], y_pred_interp[mask])
    
    start = time.monotonic()
    y_pred_interp = prefill_penalty3D(N_decode, N_prefill, T_prefill)
    duration_intrep_no_KV = time.monotonic() - start
    mask = ~np.isnan(y_pred_interp)
    mae_interp_no_KV = mean_absolute_error(y_true[mask], y_pred_interp[mask])
    rmse_interp_no_KV = root_mean_squared_error(y_true[mask], y_pred_interp[mask])


    print(f"{duration_gbr=} {duration_gbr_no_KV=} {duration_intrep=} {duration_intrep_no_KV=}")

    print(f"{mae_gbr=} {mae_gbr_no_KV=} {mae_interp=} {mae_interp_no_KV=}")
    print(f"{rmse_gbr=} {rmse_gbr_no_KV=} {rmse_interp=} {rmse_interp_no_KV=}")

    # Suppose N_decode, N_prefill, T_prefill, KV are numpy arrays of the same length


    n_samples = len(df_test)
    n_samples= 10000

    # --- Interpolator 4D (with KV) ---
    y_pred_interp = np.zeros(n_samples)
    start = time.monotonic()
    for i in range(n_samples):
        y_pred_interp[i] = prefill_penalty(
            N_decode[i],
            N_prefill[i],
            T_prefill[i],
            KV[i]
        )
    duration_interp = time.monotonic() - start
    # --- Interpolator 4D (with KV) ---
    y_pred_interp = np.zeros(n_samples)
    start = time.monotonic()
    for i in range(n_samples):
        y_pred_interp[i] = prefill_penalty(
            N_decode[i],
            N_prefill[i],
            T_prefill[i],
            KV[i]
        )
    duration_interp = time.monotonic() - start
    req_per_sec = n_samples/duration_interp
    print(f"Interpolator 4D inference duration (line by line): {duration_interp:.3f}s {req_per_sec=:.1f} Time per req: {1000/req_per_sec:.3}ms")

    # --- Interpolator 3D (no KV) ---
    y_pred_interp_no_KV = np.zeros(n_samples)
    start = time.monotonic()
    for i in range(n_samples):
        y_pred_interp_no_KV[i] = prefill_penalty3D(
            N_decode[i],
            N_prefill[i],
            T_prefill[i],
        )
        duration_interp_no_KV = time.monotonic() - start
        req_per_sec = n_samples / duration_interp_no_KV
    print(f"Interpolator 3D inference duration (line by line): {duration_interp_no_KV:.3f}s {req_per_sec=:.1f} Time per req: {1000/req_per_sec:.3}ms")


    # --- Hybrid with KV (analytical + GBR) ---
    y_pred_gbr = np.zeros(n_samples)
    start = time.monotonic()
    for i in range(n_samples):
        y_pred_gbr[i] = round_duration(
            N_decode[i],
            N_prefill[i],
            T_prefill[i],
            KV[i]
        )
    duration_gbr = time.monotonic() - start
    req_per_sec = n_samples / duration_gbr
    print(f"Hybrid + KV inference duration (line by line): {duration_gbr:.3f}s {req_per_sec=:.1f} Time per req: {1000/req_per_sec:.3}ms")

    # --- Hybrid without KV ---
    y_pred_gbr_no_KV = np.zeros(n_samples)
    start = time.monotonic()
    for i in range(n_samples):
        y_pred_gbr_no_KV[i] = round_duration_no_KV(
            N_decode[i],
            N_prefill[i],
            T_prefill[i],
        )
    duration_gbr_no_KV = time.monotonic() - start
    req_per_sec = n_samples /duration_gbr_no_KV
    print(f"Hybrid no KV inference duration (line by line): {duration_gbr_no_KV:.3f}s {req_per_sec=:.1f} Time per req: {1000/req_per_sec:.3}ms")


import pandas as pd
import numpy as np
from scipy.stats.qmc import LatinHypercube
from sklearn.neighbors import NearestNeighbors

def lhs_sampling(df, linear_cols=None, log_cols=None, n_bins=4, points_per_bin=1):
    """
    Generate a quasi-uniform sample from a multidimensional space using 
    Latin Hypercube Sampling (LHS) - optimal coverage guarantee.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the columns to sample from.
    linear_cols : list[str]
        Columns to sample from linearly.
    log_cols : list[str]
        Columns to sample from logarithmically.
    n_bins : int
        Controls grid density (total samples ≈ n_bins^dim * points_per_bin).
    points_per_bin : int
        Multiplicity per grid point (use 1 for single points).
    
    Returns
    -------
    pd.DataFrame
        Subsample with guaranteed uniform multidimensional coverage.
    """
    linear_cols = linear_cols or []
    log_cols = log_cols or []
    all_cols = linear_cols + log_cols
    
    if not all_cols:
        raise ValueError("At least one column must be specified.")
    if not all(col in df.columns for col in all_cols):
        raise ValueError("All specified columns must exist in df.")
    
    # Compute bounds in original/log space
    bounds = []
    col_names = []
    for col in all_cols:
        data = df[col].values
        if col in log_cols:
            data = np.maximum(data, 1e-6)  # Avoid log(0)
            data = np.log10(data)
            col_names.append(f"log10({col})")
        else:
            col_names.append(col)
        bounds.append([data.min(), data.max()])
    
    dim = len(all_cols)
    total_samples = min(n_bins ** dim * points_per_bin, len(df))
    
    # Latin Hypercube Sampling in transformed space
    sampler = LatinHypercube(d=dim, seed=42)
    lhs_points = sampler.random(n=total_samples)
    
    # Scale to data bounds
    for i, (low, high) in enumerate(bounds):
        lhs_points[:, i] *= (high - low)
        lhs_points[:, i] += low
    
    # Find nearest neighbors in original data
    n_neighbors = min(points_per_bin, len(df))
    knn = NearestNeighbors(n_neighbors=n_neighbors, algorithm='kd_tree')
    data_for_knn = df[linear_cols].copy()
    if log_cols:
        for col in log_cols:
            data_for_knn[col] = np.log10(np.maximum(df[col].values, 1e-6))
    knn.fit(data_for_knn.values)
    distances, indices = knn.kneighbors(lhs_points)
    
    # Flatten multi-neighbor results
    selected_indices = indices.flatten()
    
    print(len(selected_indices))

    # Deduplicate and return
    sample_df = df.iloc[selected_indices].drop_duplicates(subset=all_cols)
    
    return sample_df.reset_index(drop=True)


def grid_sampling(df, linear_cols=None, log_cols=None, n_bins=4, points_per_bin=1):
    """
    Generate a quasi-uniform sample from a multidimensional space using binning.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the columns to sample from.
    linear_cols : list[str]
        Columns to bin linearly.
    log_cols : list[str]
        Columns to bin logarithmically.
    n_bins : int
        Number of bins per column.
    points_per_bin : int
        Number of points to take from each combination of bins.

    Returns
    -------
    pd.DataFrame
        Subsample of the original DataFrame representing the space coverage.
    """

    linear_cols = linear_cols or []
    log_cols = log_cols or []
    all_cols = linear_cols + log_cols

    if not all_cols:
        raise ValueError("At least one column must be specified for binning.")

    binned_df = df.copy()
    bin_edges = {}

    # Create bins for linear columns
    for col in linear_cols:
        edges = np.linspace(df[col].min(), df[col].max(), n_bins + 1)
        bin_edges[col] = edges
        binned_df[col + "_bin"] = pd.cut(df[col], bins=edges, include_lowest=True, labels=False)

    # Create bins for logarithmic columns
    for col in log_cols:
        col_min = max(df[col].min(), 1e-6)  # avoid log(0)
        edges = np.logspace(np.log10(col_min), np.log10(df[col].max()), n_bins + 1)
        bin_edges[col] = edges
        binned_df[col + "_bin"] = pd.cut(df[col], bins=edges, include_lowest=True, labels=False)

    # Group by all binned columns
    bin_cols = [c + "_bin" for c in all_cols]
    sample_indices = []

    grouped = binned_df.groupby(bin_cols)
    for _, group in grouped:
        # Take `points_per_bin` random points from each bin
        n_take = min(len(group), points_per_bin)
        sample_indices.extend(group.sample(n_take).index)

    return df.loc[sample_indices].reset_index(drop=True)



def load_dataframe(data, nrows=None):
    """
    Returns a pandas DataFrame whether `data` is already a DataFrame
    or a string path to a CSV file.
    """
    if isinstance(data, pd.DataFrame):
        return data
    elif isinstance(data, str) and os.path.exists(data):
        return pd.read_csv(data, nrows=nrows)
    else:
        raise ValueError("`data` must be a Pandas DataFrame or a valid CSV file path")

best_rmse = None
best_mae = None
best_mape = None
best_nb_samples = None
patience_counter = 0
best_model = None

def perf_model_has_converged(data, nrows = None, target_error=0.01, patience=10, regressor = "XGBOOST", df_sampled_test = None, max_requests=512):
    global best_rmse, best_model, best_mae, best_mape, best_nb_samples, patience_counter, REGRESSOR
    REGRESSOR = regressor
    
    df_all = load_dataframe(data, nrows)
    if df_sampled_test is None:
        df_train = df_all.sample(frac=0.8)
        df_test = df_all.drop(df_train.index)
    else: 
        df_train = df_all
        df_test = df_sampled_test

    N_decode = df_test["num_decode_requests"].to_numpy()
    N_prefill = df_test["num_prefill_requests"].to_numpy()
    T_prefill = df_test["prefill_tokens"].to_numpy()
    KV = df_test["kv_cache_per_req"].to_numpy()

    y_true = df_test["round_duration"].to_numpy()
    round_duration = RoundDurationModel(max_requests)


    round_duration.fit(df_train)
    
    y_pred_regressor = round_duration.predict(
        N_decode, N_prefill, T_prefill, KV
    )

    
    print("================ DEBUG===============")
    abs_error = np.abs(y_pred_regressor - y_true)
    worst_idx = np.argmax(abs_error)

    print("Worst prediction error:")
    print("real:", y_true[worst_idx])
    print("pred:", y_pred_regressor[worst_idx])
    print("abs error:", abs_error[worst_idx])

    print("Worst case inputs:")
    print("N_decode:", N_decode[worst_idx])
    print("N_prefill:", N_prefill[worst_idx])
    print("T_prefill:", T_prefill[worst_idx])
    print("KV:", KV[worst_idx])

    print("====================================")
    mae_regressor = mean_absolute_error(y_true, y_pred_regressor)
    rmse_regressor = root_mean_squared_error(y_true, y_pred_regressor)
    mape_regressor = mean_absolute_percentage_error(y_true, y_pred_regressor)

    
    # print(f"{best_rmse}")

    if best_rmse is None or rmse_regressor < best_rmse:
        if best_rmse is None:
            best_rmse = rmse_regressor
            best_mae = mae_regressor
            best_mape = mape_regressor
            best_model = round_duration
            best_nb_samples = len(df_train)
            patience_counter = 0
        else:
            rel_improvement = (best_rmse - rmse_regressor) / best_rmse
            print (f"\t==========>{rel_improvement=:.3f}")
            # sys.exit()
            if rel_improvement > target_error:
                best_rmse = rmse_regressor
                best_mae = mae_regressor
                best_mape = mape_regressor
                best_model = round_duration
                best_nb_samples = len(df_train)
                patience_counter = 0
            else:
                patience_counter += 1
    else:
        patience_counter += 1
    
    print(f"Monitoring samples: {len(df_all)} - {mae_regressor=:.6f} {rmse_regressor=:.6f} {mape_regressor=:.6f}")


    return best_model, patience_counter >= patience, best_rmse, best_mae, best_mape, best_nb_samples

def convergence_test(csv_data):
    global best_rmse, best_model, best_mae, best_mape, best_nb_samples, patience_counter, REGRESSOR

    best_rmse = None
    best_mae = None
    best_mape = None
    best_nb_samples = None
    patience_counter = 0
    best_model = None

    df_all = pd.read_csv(csv_data)
    max_n_rows = len(df_all)
    print(max_n_rows)
    for n_rows in range(10000, max_n_rows, 10000):
        model, conv, best_rmse, best_mae, best_mape, best_nb_samples = perf_model_has_converged(csv_data, nrows=n_rows)
        if conv:
            print(f"Model converged at {best_nb_samples} samples\n{best_rmse=:.6f}, {best_mae=:.6f}, {best_mape=:.6f}")
            return model

def convergence_test_with_sampling(csv_data) -> RoundDurationModel:

    # Reset global variables for convergence
    global best_rmse, best_model, best_mae, best_mape, best_nb_samples, patience_counter, REGRESSOR

    best_rmse = None
    best_mae = None
    best_mape = None
    best_nb_samples = None
    patience_counter = 0
    best_model = None

    df_all = pd.read_csv(csv_data) 

    # print(df_all.loc[df_all["round_duration"].idxmax()])

    # sys.exit()
    max_requests = int((df_all["num_decode_requests"] + df_all["num_prefill_requests"]).max())
    print(f"{max_requests=}")
    df_train = df_all.sample(frac=0.8, random_state=40)
    df_test = df_all.drop(df_train.index)
    if len(df_train) > 200_000:
        df_train = df_train.sample(n=200_000, random_state=40)
    max_n_rows = len(df_all)
    print(max_n_rows)
    prev_points = 1000
    # for n_bins in [3,5,6,7,8,9,10,11,12,13,14,15,16]:
    for n_bins in [12]:
        n_points_per_bins= 1
        while True:
            
            print(f"{n_bins=} \t{n_points_per_bins=}")
            df_sampled = lhs_sampling(
                df_train,
                # linear_cols=["kv_cache_per_req", "num_decode_requests", "num_prefill_requests", "prefill_tokens"],
                # log_cols=[],
                linear_cols=[ "kv_cache_per_req"],
                log_cols=["num_decode_requests", "num_prefill_requests", "prefill_tokens"],
                n_bins=n_bins,
                points_per_bin=n_points_per_bins
            )
            # sys.exit(f"{len(df_sampled)=}")
            if len(df_sampled) <= prev_points:
                n_points_per_bins = int(n_points_per_bins*2)
                print(f"Model converged at {best_nb_samples} samples\n{best_rmse=:.6f}, {best_mae=:.6f}, {best_mape=:.6f}")
                return model
            else: 
                prev_points = len(df_sampled)
            model, conv, best_rmse, best_mae, best_mape, best_nb_samples = perf_model_has_converged(df_sampled, df_sampled_test=df_test, max_requests=max_requests)
            if conv:
                print(f"Model converged at {best_nb_samples} samples\n{best_rmse=:.6f}, {best_mae=:.6f}, {best_mape=:.6f}")
                return model
            n_points_per_bins = max(int(n_points_per_bins*1.2), n_points_per_bins+1)
  
    sys.exit("Sampling convergence test did not converge within the specified bin and point limits.")

def plot_concurrency_lookup_table(max_concurrency_lookup_table, tps_array):
    print(f"{max_concurrency_lookup_table=}")
    # Convert to DataFrame
    df_lookup = pd.DataFrame({
        "TPS_target": tps_array,
        "Max_concurrency": max_concurrency_lookup_table
    })

    # Print the full table
    # pd.set_option('display.max_rows', None)  # show all rows
    # print(df_lookup)

    # Optionally save to CSV
    df_lookup.to_csv("max_concurrency_lookup_table.csv", index=False)
  
    import matplotlib.pyplot as plt

    # Plot TPS vs Max Concurrency
    plt.figure(figsize=(8, 5))
    plt.plot(df_lookup["TPS_target"], df_lookup["Max_concurrency"], marker='o', linestyle='-')
    plt.xlabel("Target Decode TPS")
    plt.ylabel("Max Concurrency (N_decode)")
    plt.title("Max Concurrency vs Target Decode TPS")
    plt.grid(True)
    plt.show()
    


def to_r_vector(name, arr):
    s = ", ".join(str(int(x)) if float(x).is_integer() else str(x) for x in arr)
    print(f"{name} <- c({s})")

if __name__ == "__main__":
    
    # round_duration = convergence_test_with_sampling("./monitoring-monotonic_nvidia-a100-80gb-pcie_4_qwen.csv")
    # round_duration.compute_tps() 
    # round_duration.save_model("./pickled_model/monitoring-monotonic_nvidia-a100-80gb-pcie_4_qwen.pkl")
    round_duration = convergence_test_with_sampling("./monitoring-monotonic_nvidia-a100-80gb-pcie_1_mistral.csv")
    round_duration.compute_tps() 
    round_duration.save_model("./pickled_model/monitoring-monotonic_nvidia-a100-80gb-pcie_1_mistral.pkl")
    # # round_duration = convergence_test_with_sampling("./monitoring-monotonic_nvidia-a100-80gb-pcie_2_mistral.csv")
    # round_duration.compute_tps() 
    # round_duration.save_model("./pickled_model/monitoring-monotonic_nvidia-a100-80gb-pcie_2_mistral.pkl")
    # round_duration = convergence_test_with_sampling("./monitoring-monotonic-train.csv")
    # round_duration.compute_tps() 
    # round_duration.save_model("./pickled_model/monitoring-monotonic_nvidia-h100-nvl_1_mistral.pkl")
    # round_duration = convergence_test_with_sampling("./monitoring-monotonic_nvidia-h100-nvl_1_qwen_deduplicated.csv")
    # round_duration.compute_tps()     
    # round_duration.save_model("./pickled_model/monitoring-monotonic_nvidia-h100-nvl_1_qwen.pkl")
    # round_duration = convergence_test_with_sampling("./monitoring-monotonic_nvidia-a100-80gb-pcie_4_mistral.csv")
    # round_duration.compute_tps() 
    # round_duration.save_model("./pickled_model/monitoring-monotonic_nvidia-a100-80gb-pcie_4_mistral.pkl")
    


    
    # round_duration = convergence_test("./monitoring-monotonic_nvidia-a100-80gb-pcie_4_qwen.csv")
    # round_duration = convergence_test("./monitoring-monotonic_nvidia-a100-80gb-pcie_1_mistral.csv")
    # round_duration = convergence_test("./monitoring-monotonic_nvidia-a100-80gb-pcie_2_mistral.csv")
    # round_duration = convergence_test("./monitoring-monotonic-train.csv")


    # round_duration = RoundDurationModel.load_model("./pickled_model/monitoring-monotonic_nvidia-h100-nvl_1_mistral.pkl")
    round_duration = RoundDurationModel.load_model("./pickled_model/monitoring-monotonic_nvidia-a100-80gb-pcie_4_qwen.pkl")
    # round_duration = RoundDurationModel.load_model("./pickled_model/monitoring-monotonic_nvidia-a100-80gb-pcie_1_mistral.pkl")
    # round_duration = RoundDurationModel.load_model("./pickled_model/monitoring-monotonic_nvidia-h100-nvl_1_qwen.pkl")
    # round_duration = RoundDurationModel.load_model("./pickled_model/monitoring-monotonic_nvidia-a100-80gb-pcie_4_mistral.pkl")
    
    print(round_duration.compute_tps())
    max_concurrency_lookup_table, tps_array = round_duration.max_concurrency_lookup_table()
    print(f"{tps_array=}")

    plot_concurrency_lookup_table(max_concurrency_lookup_table, tps_array)
    print(round_duration.compute_tps(force=True))
    print(f"{round_duration.max_requests=}")
    # max_concurrency_lookup_table, tps_array = round_duration.max_concurrency_lookup_table()
    print(f"{tps_array=}")


    to_r_vector("concurrency", max_concurrency_lookup_table)
    to_r_vector("idx", list(range(len(max_concurrency_lookup_table)-1, -1, -1)))
    plot_concurrency_lookup_table(max_concurrency_lookup_table, tps_array)

    sys.exit()


    # test_model()
    
    # test_training_size("XGBOOST", dumb=True)

    # for reg in ["GBR", "XGBOOST" , "LightGBM" , "CATBOOST", "RANDOMFOREST", "EXTRATREES"]: 
    #     test_training_size(reg)
    sys.exit()

    # Note on decode TPS and number of requests:
    # - "decode_tps_per_req" generally **decreases as the number of concurrent decode requests increases**, 
    #   because the GPU work is shared among more requests and each request gets fewer cycles per unit time.
    # - This relationship is captured in our interpolators: you can query TPS for any combination of 
    #   decode requests, prefill requests, and prefill tokens, and see how TPS scales with load.

    # The interpolators are bidirectional:
    # 1) Forward: given number of decode/prefill requests and avg tokens → get TPS
    # 2) Reverse: given target TPS and avg tokens → get maximum number of requests to acheive thsi

    ###########################################################################
    # Vectorized example: multiple configurations at once
    ###########################################################################
    num_decode_req_arr = np.array([1, 2, 5, 10])
    num_prefill_req_arr = np.array([0, 1, 3, 5])
    tot_prefill_tokens_arr = np.array([0, 100, 500, 1200])
    kv_cache_arr = np.array([1e2,1e2,1e4,1e6])

    # Example 1: prefill penalty for multiple configurations
    mean_round_duration_arr = prefill_penalty(num_decode_req_arr, num_prefill_req_arr, tot_prefill_tokens_arr, kv_cache_arr)
    gbr_mean_round_duration_arr = round_duration(num_decode_req_arr, num_prefill_req_arr, tot_prefill_tokens_arr, kv_cache_arr)
    print("Mean round duration for multiple configs:", mean_round_duration_arr)
    print("Mean round duration for multiple configs analytical model + GBR:", gbr_mean_round_duration_arr)

    # Example 2: decode TPS for multiple configurations
    decode_tps_arr = decode_tps_interp(num_decode_req_arr, num_prefill_req_arr, tot_prefill_tokens_arr, kv_cache_arr)
    print("Decode TPS per request for multiple configs:", decode_tps_arr)

    # Example 3: estimate required decode requests to reach a target TPS for multiple prefill setups
    target_decode_tps_arr = np.array([1.0, 2.0, 2.5, 3.0])
    avg_prefill_tokens_arr = np.array([0, 50, 150, 300])
    mean_num_prefill_arr = np.array([0, 1, 3, 5])

    est_decode_requests_arr = decode_req_interp(mean_num_prefill_arr, avg_prefill_tokens_arr, target_decode_tps_arr, kv_cache_arr)
    print("Estimated mean decode requests to reach target TPS:", est_decode_requests_arr)
