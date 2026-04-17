import numpy as np
import time
import cloudpickle as pickle

from scipy.optimize import minimize

from sklearn.metrics import mean_absolute_error, root_mean_squared_error

from xgboost import XGBRegressor


class RoundDurationModel:
    """
    Analytical + Gradient X Boosting Regressor (XGBOOST) model for predicting LLM serving round duration.

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
        self.median_KV = 0
        self.prefill_tps = None
        self.decode_tps = None

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
        self.regressor = XGBRegressor(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.1,
                objective='reg:squarederror'
            )
        print(f"Training XGBOOST on residuals...")
        start_time = time.monotonic()
        self.regressor.fit(X, residuals)
        gbr_duration = time.monotonic() - start_time
        print(f"XGBOOST training time: {gbr_duration:.2f}s")
        print(f"XGBOOST trained on residuals")

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

        print(f"\n===== XGBOOST Contribution Analysis =====")
        print(f"MAE analytical      \t: {mae_analytical:.6f}")
        print(f"MAE analytical+XGBOOST\t: {mae_full:.6f}")
        print(f"MAE reduction       \t: {(1 - mae_full/mae_analytical)*100:.2f}%")

        print()

        print(f"RMSE analytical     \t: {rmse_analytical:.6f}")
        print(f"RMSE analytical+XGBOOST\t: {rmse_full:.6f}")
        print(f"RMSE reduction      \t: {(1 - rmse_full/rmse_analytical)*100:.2f}%")

        print()

        print(f"Residual variance explained by XGBOOST (R²): {r2_residual:.3f}")
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
        if KV is None:
            KV = np.full(len(N_decode), self.median_KV)

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

    def compute_tps(self):
        if self.prefill_tps is not None and self.decode_tps is not None:
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

        max_val = 0
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
                if val != N_max:
                    max_val = val
                else:
                    val = max_val
                results.append(val)
            else:
                # If none satisfy target, return -1
                results.append(-100)
        
        return np.array(results)
    

    # Evaluate the concurrency requirement for each TPS value and convert it to integers.
    def max_concurrency_lookup_table(self, model_nb_requests_max=512, step = 0.1):
        
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