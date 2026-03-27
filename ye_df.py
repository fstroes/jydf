import numpy as np
from statsmodels.nonparametric.kernel_regression import KernelReg
from scipy.optimize import minimize
from jydf import JYDF
from concurrent.futures import ThreadPoolExecutor
import os


class StatsKernelWrapper:
    """
    Wrapper for statsmodels KernelReg (Nadaraya-Watson / Local Constant).
    'lc' = local constant, 'll' = local linear.
    """
    def __init__(self, x, bw=0.1):
        self.x = x
        self.bw = [bw]  # statsmodels expects a list of bandwidths for each exog

    def predict_y_hat(self, y):
        # We initialize the model with the fixed bandwidth
        # data_type='c' indicates a continuous variable
        model = KernelReg(endog=y, exog=self.x, reg_type='lc', bw=self.bw, var_type='c')
        y_hat, _ = model.fit()
        return y_hat


class StatsKernelWrapperAhead:
    """
    Wrapper for statsmodels KernelReg (Nadaraya-Watson / Local Constant).
    'lc' = local constant, 'll' = local linear.
    """
    def __init__(self, x, skip=0, bw=0.1):
        self.x = x
        self.bw = [bw] # statsmodels expects a list of bandwidths for each exog
        self.skip = skip

    def predict_y_hat(self, y):
        # We initialize the model with the fixed bandwidth
        # data_type='c' indicates a continuous variable
        y_hat = [0] * (self.skip + 1)
        for i, _ in enumerate(y):
            if i > self.skip:
                model = KernelReg(endog=y[:i], exog=self.x[:i], reg_type='lc', bw=self.bw, var_type='c')
                y_hat_i, _ = model.fit()
                y_hat.append(y_hat_i[-1])

        return np.array(y_hat)


class AheadOptimizeH:
    """
    A Meta-Wrapper that uses Scipy to find the optimal bandwidth h
    for one-step-ahead prediction error on any given y.
    """

    def __init__(self, base_model, initial_h=0.5):
        self.model = base_model
        self.h = initial_h

    def _objective(self, h_val, y):
        # h_val is passed as a list/array by scipy
        h = h_val[0]

        self.model.bw = [h]
        preds = self.model.predict_y_hat(y)

        # Calculate MSE only on the points after 'skip'
        skip = self.model.skip
        mse = np.mean((y[skip:] - preds[skip:]) ** 2)
        return mse

    def predict_y_hat(self, y):
        """
        1. Find the best h for 'y' using Brent's method (bounded scalar search).
        2. Return the y_hat vector at that optimal h.
        """
        # We use 'bounded' because bandwidth is a 1D scalar and must be positive
        res = minimize(
            self._objective,
            x0=[self.h],
            args=(y,),
            method='L-BFGS-B',
            bounds=[(0.001, 100.0)],
            options={'ftol': 1e-3}  # <-- Add this options dictionary
        )

        self.h = res.x[0]

        # Set the model to the winner and return the final predictions
        self.model.bw = [self.h]
        return self.model.predict_y_hat(y)


class SimplePolyWrapper:
    """
    Minimalist Polynomial Model: y_hat = X * (X.T * X)^-1 * X.T * y
    """
    def __init__(self, x, degree=3):
        # Pre-calculate the Vandermonde matrix (the 'exogenous' part)
        self.X = np.vander(x, degree + 1)
        # Pre-calculate the Moore-Penrose pseudoinverse for speed
        self.X_pinv = np.linalg.pinv(self.X)

    def predict_y_hat(self, y):
        """Standard OLS: Just a matrix multiplication."""
        beta = self.X_pinv @ y
        return self.X @ beta


def run_single_estimation(seed, x_data, y_data):
    """Worker function for a single iteration of the DF estimation."""
    # Ensure each thread has a unique random state for the JYDF tau perturbations
    np.random.seed(seed)

    # Re-initialize models to ensure clean state
    poly_fit_3 = SimplePolyWrapper(x_data, degree=3)
    nw = StatsKernelWrapper(x_data, bw=.1)
    nw_ahead = StatsKernelWrapperAhead(x_data, bw=.1, skip=1)
    nw_ahead_fit_h = AheadOptimizeH(base_model=nw_ahead)

    jy = JYDF(tau=0.2)

    # Run the estimates
    # (Using iterations=1 inside, but we repeat the whole process 10 times)
    d1 = jy.estimate_df(poly_fit_3, y_data, iterations=1)
    d2 = jy.estimate_df(nw, y_data, iterations=1)
    d3 = jy.estimate_df(nw_ahead, y_data, iterations=1)
    d4 = jy.estimate_df(nw_ahead_fit_h, y_data, iterations=1)
    return np.array([d1, d2, d3, d4])


if __name__ == "__main__":
    T = 50
    np.random.seed(22)
    n_runs = 400
    # Use as many cores as available
    max_workers = os.cpu_count()

    print(f"Starting {n_runs} runs across {max_workers} cores...")

    seeds = list(range(n_runs))
    xs =[np.random.normal(0, 1,  T) for i in range(n_runs)]
    ys = [np.random.normal(0, 1,  T) for i in range(n_runs)]


    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Map the function across the seeds
        # We pass x and y as constants to each call
        futures = [executor.submit(run_single_estimation, s, x, y) for s, x, y in zip(seeds, xs, ys)]

        for future in futures:
            results.append(future.result())

    # Average the results across the 10 runs
    results_array = np.array(results)
    avg_df = np.mean(results_array, axis=0)

    print("-" * 30)
    print(f"Final Averaged DF (n_df_estimation={n_runs}, T={T}, scale_x=scale_y=1):")
    print(f"Poly (deg 3):      {avg_df[0]:.4f}")
    print(f"NW bw=0.1 (Fixed h=1):    {avg_df[1]:.4f}")
    print(f"NW bw=0.1 one step ahead prediction (Fixed h=1):    {avg_df[2]:.4f}")
    print(f"NW one step ahead prediction (h+1) but find optimal bandwidth for h+1 using h+1 error in sample:  {avg_df[3]:.4f}")
    print("-" * 30)

