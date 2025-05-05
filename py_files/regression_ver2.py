
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
from statsmodels.tsa.arima.model import ARIMA
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler
from concurrent.futures import ProcessPoolExecutor
import functools
import time
import matplotlib.pyplot as plt
import matplotlib
# real-world dataset
from sklearn.datasets import load_diabetes
from sklearn.datasets import fetch_california_housing
from sklearn.datasets import make_regression
from sklearn.datasets import make_friedman1
from sklearn.model_selection import train_test_split
from statsmodels.tsa.ar_model import AutoReg
import random



def diabetes():
    # get the datasets
    data = load_diabetes()

    # Convert the features (X) into a DataFrame
    X_df = pd.DataFrame(data.data, columns=data.feature_names)

    # Convert the target (y) into a DataFrame
    y_df = pd.DataFrame(data.target, columns=['Target'])

    # Combine X and y into a single DataFrame
    diabetes_df = pd.concat([X_df, y_df], axis=1)

    # Select only the columns 'age', 'sex', 'bmi' as X, and 'Target' as y
    X_selected = diabetes_df[['age', 'sex', 'bmi']]
    y_selected = diabetes_df['Target']

    X_train, X_test, y_train, y_test = train_test_split(X_selected, y_selected, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test



def california_housing():
    data = fetch_california_housing()

    X_df = pd.DataFrame(data.data, columns=data.feature_names)
    y_df = pd.DataFrame(data.target, columns=['Target'])
    X_df = X_df.iloc[:500]
    y_df = y_df.iloc[:500]

    california_df = pd.concat([X_df, y_df], axis=1)

    X_selected = california_df[['MedInc', 'AveRooms', 'AveOccup']]
    y_selected = california_df['Target']

    X_train, X_test, y_train, y_test = train_test_split(X_selected, y_selected, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test



def regression():
    X, y = make_regression(n_samples=500, n_features=10, noise=0.1, random_state=42)

    X_df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
    y_df = pd.DataFrame(y, columns=['Target'])

    regression_df = pd.concat([X_df, y_df], axis=1)

    X_selected = regression_df[['feature_0', 'feature_1', 'feature_2']]
    y_selected = regression_df['Target']

    X_train, X_test, y_train, y_test = train_test_split(X_selected, y_selected, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test



def friedman1():
    X, y = make_friedman1(n_samples=500, n_features=10, noise=0.1, random_state=42)

    X_df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
    y_df = pd.DataFrame(y, columns=['Target'])

    friedman_df = pd.concat([X_df, y_df], axis=1)

    X_selected = friedman_df[['feature_0', 'feature_1', 'feature_2']]
    y_selected = friedman_df['Target']

    X_train, X_test, y_train, y_test = train_test_split(X_selected, y_selected, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test

# ==================== Non-convex Optimization Functions ====================




def nonconvex_f(beta, X, y):
   """
   Non-convex objective function.
   Uses dot product for multi-dimensional beta.
   """
   residual = y - np.dot(X, beta)
   residual_sq = residual ** 2
   cost = 0.5 * np.sum(residual_sq / (1 + residual_sq))
   return cost




def nonconvex_grad_f(beta, X, y):
   """
   Gradient of the non-convex function.
   Computes the gradient with respect to beta.
   """
   residual = y - np.dot(X, beta)
   residual_sq = residual ** 2
   factor = residual / (1 + residual_sq)
   gradient = -np.dot(X.T, factor)
   return gradient




def nonconvex_hessian_f(beta, X, y):
    residual = y - np.dot(X, beta)
    residual_sq = residual ** 2
    factor = (1 - residual_sq) / (1 + residual_sq)**2
    factor = np.asarray(factor)
    H = np.dot(X.T, X * factor[:, np.newaxis])
    return H



def nonconvex_f(beta, X, y, c=1.0):
    """
    Non-convex Cauchy loss.
    Uses dot product for multi-dimensional beta.
    c: scale parameter controlling the “flatness” of the tails.
    """
    residual = y - np.dot(X, beta)
    residual_sq = residual**2
    cost = 0.5 * c**2 * np.sum(np.log(1 + residual_sq / c**2))
    return cost

def nonconvex_grad_f(beta, X, y, c=1.0):
    """
    Gradient of the non-convex Cauchy loss.
    Computes the gradient with respect to beta.
    """
    residual = y - np.dot(X, beta)
    residual_sq = residual**2
    # d/dr [0.5 c^2 log(1 + r^2/c^2)] = r / (1 + r^2/c^2)
    factor = residual / (1 + residual_sq / c**2)
    gradient = -np.dot(X.T, factor)
    return gradient

def nonconvex_hessian_f(beta, X, y, c=1.0):
    """
    Hessian of the non-convex Cauchy loss.
    Computes a 2D Hessian matrix.
    """
    residual = y - np.dot(X, beta)
    residual_sq = residual**2
    # d/dr [r / (1 + r^2/c^2)] = (1 - r^2/c^2) / (1 + r^2/c^2)^2
    factor = (1 - residual_sq / c**2) / (1 + residual_sq / c**2)**2
    factor = np.asarray(factor)  # <-- 加这一行！
    H = np.dot(X.T, X * factor[:, np.newaxis])
    return H



# Welsch loss
# def nonconvex_f(beta, X, y, c=1.0):
#     """
#     Non-convex Welsch (exponential) loss.
#     Uses dot product for multi-dimensional beta.
#     c: scale parameter controlling down-weighting of large residuals.
#     """
#     residual = y - np.dot(X, beta)
#     residual_sq = residual**2
#     cost = 0.5 * c**2 * np.sum(1 - np.exp(-residual_sq / c**2))
#     return cost

# def nonconvex_grad_f(beta, X, y, c=1.0):
#     """
#     Gradient of the non-convex Welsch loss.
#     Computes the gradient with respect to beta.
#     """
#     residual = y - np.dot(X, beta)
#     residual_sq = residual**2
#     # d/dr [0.5 c^2 (1 - exp(-r^2/c^2))] = r * exp(-r^2/c^2)
#     factor = residual * np.exp(-residual_sq / c**2)
#     gradient = -np.dot(X.T, factor)
#     return gradient

# def nonconvex_hessian_f(beta, X, y, c=1.0):
#     """
#     Hessian of the non-convex Welsch loss.
#     Computes a 2D Hessian matrix.
#     """
#     residual = y - np.dot(X, beta)
#     residual_sq = residual**2
#     # d/dr [r e^{-r^2/c^2}] = e^{-r^2/c^2} * (1 - 2r^2/c^2)
#     factor = np.exp(-residual_sq / c**2) * (1 - 2 * residual_sq / c**2)
#     H = np.dot(X.T, X * factor[:, np.newaxis])
#     return H

# def nonconvex_f(beta, X, y, c=4.685):
#     """
#     Tukey’s biweight (bisquare) loss.
#     - Non-convex overall, but convex for |residual| small.
#     - c: tuning constant (≈4.685 gives 95% efficiency under Gaussian noise).
#     """
#     residual = y - np.dot(X, beta)
#     w = residual**2 / c**2
#     # loss_i = (c^2/6) [1 - (1 - w)^3]   if w <= 1,   else c^2/6
#     mask = w <= 1
#     loss = np.empty_like(residual)
#     loss[mask]   = c**2/6 * (1 - (1 - w[mask])**3)
#     loss[~mask]  = c**2/6
#     return np.sum(loss)

# def nonconvex_grad_f(beta, X, y, c=4.685):
#     """
#     Gradient of Tukey’s biweight loss.
#     ∂/∂r  loss_i = r * (1 - w)^2  for w<=1,  else 0
#     """
#     residual = y - np.dot(X, beta)
#     w = residual**2 / c**2
#     mask = w <= 1
#     psi = np.zeros_like(residual)
#     psi[mask] = residual[mask] * (1 - w[mask])**2
#     # gradient = - X^T psi
#     return -np.dot(X.T, psi)

# def nonconvex_hessian_f(beta, X, y, c=4.685):
#     """
#     Hessian of Tukey’s biweight loss.
#     ∂/∂r [r (1 - w)^2] = (1 - w)*(1 - 5w)  for w<=1,  else 0
#     """
#     residual = y - np.dot(X, beta)
#     w = residual**2 / c**2
#     mask = w <= 1
#     factor = np.zeros_like(residual)
#     factor[mask] = (1 - w[mask]) * (1 - 5*w[mask])
#     # H = X^T diag(factor) X
#     return np.dot(X.T, X * factor[:, np.newaxis])





# ==================== Optimization Utilities ====================




# def sgd_with_bound(f, grad_f, start_point, end_point, X, y,
#                   learning_rate=0.01, iterations=1000, tol=1e-6):
#    """
#    Simple SGD optimizer with a bound check.
#    """
#    x = np.array(start_point, dtype=float)
#    x_end = np.array(end_point, dtype=float)
#    for i in range(iterations):
#        gradient = grad_f(x, X, y)
#        x_new = x - learning_rate * gradient
#        if np.all(x_new >= x_end):
#            break
#        if np.linalg.norm(x_new - x) < tol:
#            break
#        x = x_new
#    return x, f(x, X, y)

def sgd_with_bound(f, grad_f, start_point, end_point, X, y,
                   learning_rate=0.01, iterations=1000,
                   tol=1e-6, batch_size=None, random_state=None):
    """
    Simple SGD optimizer with a bound check and optional mini-batches.
    """
    rng = np.random.default_rng(random_state)
    x = np.array(start_point, dtype=float)
    x_end = np.array(end_point, dtype=float)
    n_samples = X.shape[0]

    for i in range(iterations):
        # Select batch
        if batch_size is None or batch_size >= n_samples:
            X_batch, y_batch = X, y
        else:
            idx = rng.choice(n_samples, size=batch_size, replace=False)
            if isinstance(X, (pd.DataFrame, pd.Series)):
                X_batch = X.iloc[idx]
            else:
                X_batch = X[idx]
            
            if isinstance(y, (pd.DataFrame, pd.Series)):
                y_batch = y.iloc[idx]
            else:
                y_batch = y[idx]

        # Compute gradient and take step
        gradient = grad_f(x, X_batch, y_batch)
        x_new = x - learning_rate * gradient

        # Check bound
        if np.all(x_new >= x_end):
            break

        # Check convergence
        if np.linalg.norm(x_new - x) < tol:
            break

        x = x_new

    return x, f(x, X, y)




def rearrange_bounds(bounds):
   """
   Rearrange bounds for differential evolution.
   """
   min_bounds, max_bounds = bounds
   rearranged = list(zip(min_bounds, max_bounds))
   return rearranged




# Define a picklable wrapper function for differential evolution
def de_search_method3(f, bounds, X, y, maxiters=100):
   """
   Global optimization using differential evolution.
   """
   def f_wrapper(beta):
       return f(beta, X, y)
  
   result = differential_evolution(
       f_wrapper,
       rearrange_bounds(bounds),
       maxiter=maxiters
   )
   return result.x, result.fun




def generate_uniform_start_end_pairs(start_point, end_point, n):
   """
   Generates n-1 uniform pairs between start_point and end_point.
   """
   points_lst = []
   start = np.array(start_point)
   end = np.array(end_point)
   points = [start + t * (end - start) for t in np.linspace(0, 1, n)]
   for i in range(n - 1):
       start_pt = points[i]
       end_pt = points[i + 1]
       points_lst.append([start_pt, end_pt])
   return points_lst



def process_point_with_no_zoom_in(f, grad_f, hessian_f, global_search, pt, X, y,
                                 learning_rate=0.01, iterations=1000, tol=1e-6, batch_size=32):
   """
   Processes a single pair of start and end points with SGD and,
   if necessary, applies a global search when the Hessian is non-convex.
   """
   points = []
   results = []
   start, end = pt
   point, result = sgd_with_bound(f, grad_f, start, end, X, y, learning_rate, iterations, tol, batch_size=batch_size)
   H = hessian_f(point, X, y)
   eigenvalues = np.linalg.eigvalsh(H)
   is_convex = np.all(eigenvalues >= 0)
   points.append(point)
   results.append(result)




   if not is_convex:
       # Update bounds for the global search based on dimensionality
       num_dims = len(point)
       lower_bounds = start * num_dims
       upper_bounds = end * num_dims
       ds_point, ds_min = global_search(f, (lower_bounds, upper_bounds), X, y, maxiters=10)
       points.append(ds_point)
       results.append(ds_min)




   return points, results




# def sgd_opt_global_search(start_intervals, end_intervals, n, f, grad_f, hessian_f, global_search,
#                          X, y, learning_rate=0.01, max_iterations=1000, tol=1e-6):
#    """
#    Runs multiple instances of SGD (with possible global search) in parallel.
#    """
#    iters = int(max_iterations / n)
#    points_lst = generate_uniform_start_end_pairs(start_intervals, end_intervals, n)
  
#    all_points = []
#    all_results = []
  
#    # Run sequentially instead of using ProcessPoolExecutor
#    for pt in points_lst:
#        points, results = process_point_with_no_zoom_in(
#            f, grad_f, hessian_f, global_search, pt, X, y,
#            learning_rate, iters, tol)
#        all_points.extend(points)
#        all_results.extend(results)
  
#    best_idx = np.argmin(all_results)
#    return all_points[best_idx], all_results[best_idx]


# parallel computing
executor = ProcessPoolExecutor()

def sgd_opt_global_search(start_intervals, end_intervals, n, f, grad_f, hessian_f, global_search,
                          X, y, learning_rate=0.01, max_iterations=1000, tol=1e-6, batch_size=32):
    iters = int(max_iterations / n)
    points_lst = generate_uniform_start_end_pairs(start_intervals, end_intervals, n)
    futures = [executor.submit(
        process_point_with_no_zoom_in, f, grad_f, hessian_f,
        global_search, pt, X, y, learning_rate, iters, tol, batch_size) for pt in points_lst]
    
    results = [future.result() for future in futures]
    
    optimized_points = []
    optimized_results = []

    for res in results:
        points, vals = res
        optimized_points.extend(points)
        optimized_results.extend(vals)

    best_idx = np.argmin(optimized_results)
    best_beta_custom = optimized_points[best_idx]
    best_loss_custom = optimized_results[best_idx]
    return best_beta_custom,best_loss_custom



# ==================== Data Preprocessing Functions ====================




def impute_na_values(df, method='mean', n_neighbors=5):
   """
   Function to impute NaN values in a DataFrame using different methods.
  
   Parameters:
   - df: pandas DataFrame with NaN values
   - method: Imputation method ('mean', 'median', or 'knn')
   - n_neighbors: Number of neighbors for KNN imputation (only used for 'knn' method)
  
   Returns:
   - A DataFrame with NaN values imputed.
   """
   if method == 'mean':
       return df.apply(lambda col: col.fillna(col.mean()) if col.isna().any() else col, axis=0)
   elif method == 'median':
       return df.apply(lambda col: col.fillna(col.median()) if col.isna().any() else col, axis=0)
   elif method == 'knn':
       imputer = KNNImputer(n_neighbors=n_neighbors)
       return pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
   else:
       raise ValueError("Method must be one of 'mean', 'median', or 'knn'.")




def standard_scale_df(df: pd.DataFrame) -> pd.DataFrame:
   """StandardScaler"""
   scaler = StandardScaler()
   scaled = scaler.fit_transform(df)
   return pd.DataFrame(scaled, columns=df.columns, index=df.index)




def minmax_scale_df(df: pd.DataFrame) -> pd.DataFrame:
   """MinMaxScaler"""
   scaler = MinMaxScaler()
   scaled = scaler.fit_transform(df)
   return pd.DataFrame(scaled, columns=df.columns, index=df.index)




def maxabs_scale_df(df: pd.DataFrame) -> pd.DataFrame:
   """MaxAbsScaler"""
   scaler = MaxAbsScaler()
   scaled = scaler.fit_transform(df)
   return pd.DataFrame(scaled, columns=df.columns, index=df.index)




def scale_dataframe(df, method='standard'):
   """
   Scale each column of a DataFrame.
  
   Parameters:
   - df: pandas DataFrame, the input data to be scaled
   - method: Scaling method, one of 'standard', 'minmax', or 'maxabs'
  
   Returns:
   - A scaled DataFrame
   """
   if method == 'standard':
       return standard_scale_df(df)
   elif method == 'minmax':
       return minmax_scale_df(df)
   elif method == 'maxabs':
       return maxabs_scale_df(df)
   else:
       raise ValueError("Method must be one of 'standard', 'minmax', or 'maxabs'")




# ==================== End-to-End Pipeline with Soft Parameters ====================




def softmax(x):
   """
   Compute softmax values for each set of scores in x.
   """
   e_x = np.exp(x - np.max(x))
   return e_x / e_x.sum()




def pipeline_with_soft_parameters(params, X_raw, y):
   """
   Complete pipeline with soft parameters for each step.
  
   Parameters:
   - params: numpy array containing all parameters
       - params[0:3]: Softmax weights for imputation methods [mean, median, knn]
       - params[3:6]: Softmax weights for scaling methods [standard, minmax, maxabs]
       - params[6:]: Regression coefficients
   - X_raw: Raw input data with missing values
   - y: Target values
  
   Returns:
   - The non-convex objective function value
   """
   # Convert raw input to DataFrame if it's not already
   if not isinstance(X_raw, pd.DataFrame):
       X_raw = pd.DataFrame(X_raw)
  
   # Extract and normalize parameters
   impute_weights = softmax(params[0:3])
   scale_weights = softmax(params[3:6])
   regression_params = params[6:]
  
   # Imputation step - apply weighted combination
   X_imp_mean = impute_na_values(X_raw, method='mean')
   X_imp_median = impute_na_values(X_raw, method='median')
   X_imp_knn = impute_na_values(X_raw, method='knn')
  
   X_imputed = pd.DataFrame(
       impute_weights[0] * X_imp_mean.values +
       impute_weights[1] * X_imp_median.values +
       impute_weights[2] * X_imp_knn.values,
       columns=X_raw.columns
   )
  
   # Scaling step - apply weighted combination
   X_scale_standard = standard_scale_df(X_imputed)
   X_scale_minmax = minmax_scale_df(X_imputed)
   X_scale_maxabs = maxabs_scale_df(X_imputed)
  
   X_scaled = pd.DataFrame(
       scale_weights[0] * X_scale_standard.values +
       scale_weights[1] * X_scale_minmax.values +
       scale_weights[2] * X_scale_maxabs.values,
       columns=X_raw.columns
   )
  
   # Regression step
   X_final = X_scaled.values
   return nonconvex_f(regression_params, X_final, y)




def pipeline_gradient(params, X_raw, y, epsilon=1e-6):
   """
   Compute numerical gradient for the pipeline parameters.
  
   Uses central difference approximation.
   """
   n_params = len(params)
   grad = np.zeros(n_params)
  
   for i in range(n_params):
       params_plus = params.copy()
       params_plus[i] += epsilon
      
       params_minus = params.copy()
       params_minus[i] -= epsilon
      
       f_plus = pipeline_with_soft_parameters(params_plus, X_raw, y)
       f_minus = pipeline_with_soft_parameters(params_minus, X_raw, y)
      
       grad[i] = (f_plus - f_minus) / (2 * epsilon)
  
   return grad




def pipeline_hessian(params, X_raw, y, epsilon=1e-5):
   """
   Compute numerical Hessian for the pipeline parameters.
  
   Uses finite difference approximation.
   """
   n_params = len(params)
   hessian = np.zeros((n_params, n_params))
  
   for i in range(n_params):
       for j in range(n_params):
           # Compute central points
           params_pp = params.copy()
           params_pp[i] += epsilon
           params_pp[j] += epsilon
          
           params_pm = params.copy()
           params_pm[i] += epsilon
           params_pm[j] -= epsilon
          
           params_mp = params.copy()
           params_mp[i] -= epsilon
           params_mp[j] += epsilon
          
           params_mm = params.copy()
           params_mm[i] -= epsilon
           params_mm[j] -= epsilon
          
           f_pp = pipeline_with_soft_parameters(params_pp, X_raw, y)
           f_pm = pipeline_with_soft_parameters(params_pm, X_raw, y)
           f_mp = pipeline_with_soft_parameters(params_mp, X_raw, y)
           f_mm = pipeline_with_soft_parameters(params_mm, X_raw, y)
          
           hessian[i, j] = (f_pp - f_pm - f_mp + f_mm) / (4 * epsilon * epsilon)
  
   return hessian


def _compute_pipeline_output(params, X_raw):
    """
    Helper to compute the final design matrix X_final and regression_params.
    """
    # Ensure DataFrame
    X_df = pd.DataFrame(X_raw) if not isinstance(X_raw, pd.DataFrame) else X_raw.copy()

    # Soft parameters
    impute_weights = softmax(params[0:3])
    scale_weights = softmax(params[3:6])
    regression_params = params[6:]

    # Imputation
    X_imp_mean   = impute_na_values(X_df, method='mean')
    X_imp_median = impute_na_values(X_df, method='median')
    X_imp_knn    = impute_na_values(X_df, method='knn')
    X_imputed = (
        impute_weights[0] * X_imp_mean.values +
        impute_weights[1] * X_imp_median.values +
        impute_weights[2] * X_imp_knn.values
    )
    X_imputed = pd.DataFrame(X_imputed, columns=X_df.columns)

    # Scaling
    X_std   = standard_scale_df(X_imputed)
    X_mm    = minmax_scale_df(X_imputed)
    X_maxab = maxabs_scale_df(X_imputed)
    X_scaled = (
        scale_weights[0] * X_std.values +
        scale_weights[1] * X_mm.values +
        scale_weights[2] * X_maxab.values
    )
    X_scaled = pd.DataFrame(X_scaled, columns=X_df.columns)

    return regression_params, X_scaled.values

def pipeline_with_soft_parameters(params, X_raw, y):
    """
    Full pipeline ending in nonconvex_f.
    """
    regression_params, X_final = _compute_pipeline_output(params, X_raw)
    return nonconvex_f(regression_params, X_final, y)

def pipeline_gradient(params, X_raw, y, epsilon=1e-6):
    """
    Gradient using:
    - nonconvex_grad_f for regression block
    - numerical central diff + nonconvex_f for soft parameters
    """
    n = len(params)
    grad = np.zeros(n)

    # Forward pass for analytic part
    regression_params, X_final = _compute_pipeline_output(params, X_raw)
    # Analytical gradient for regression block
    grad[6:] = nonconvex_grad_f(regression_params, X_final, y)

    # Numerical for soft parameters
    for i in range(6):
        p_plus = params.copy();  p_plus[i] += epsilon
        p_minus = params.copy(); p_minus[i] -= epsilon
        rp_plus, X_plus = _compute_pipeline_output(p_plus, X_raw)
        rp_minus, X_minus = _compute_pipeline_output(p_minus, X_raw)
        f_plus  = nonconvex_f(rp_plus, X_plus, y)
        f_minus = nonconvex_f(rp_minus, X_minus, y)
        grad[i] = (f_plus - f_minus) / (2 * epsilon)

    return grad

def pipeline_hessian(params, X_raw, y, epsilon=1e-5):
    """
    Hessian using:
    - nonconvex_hessian_f for regression block
    - numerical central diff + nonconvex_f for other entries
    """
    n = len(params)
    hessian = np.zeros((n, n))

    # Compute analytic Hessian block
    regression_params, X_final = _compute_pipeline_output(params, X_raw)
    hessian[6:, 6:] = nonconvex_hessian_f(regression_params, X_final, y)

    # Numerical for soft and cross-terms
    for i in range(6):
        for j in range(n):
            pp = params.copy(); pp[i] += epsilon; pp[j] += epsilon
            pm = params.copy(); pm[i] += epsilon; pm[j] -= epsilon
            mp = params.copy(); mp[i] -= epsilon; mp[j] += epsilon
            mm = params.copy(); mm[i] -= epsilon; mm[j] -= epsilon
            rp_pp, X_pp = _compute_pipeline_output(pp, X_raw)
            rp_pm, X_pm = _compute_pipeline_output(pm, X_raw)
            rp_mp, X_mp = _compute_pipeline_output(mp, X_raw)
            rp_mm, X_mm = _compute_pipeline_output(mm, X_raw)
            f_pp = nonconvex_f(rp_pp, X_pp, y)
            f_pm = nonconvex_f(rp_pm, X_pm, y)
            f_mp = nonconvex_f(rp_mp, X_mp, y)
            f_mm = nonconvex_f(rp_mm, X_mm, y)
            hessian[i, j] = (f_pp - f_pm - f_mp + f_mm) / (4 * epsilon**2)
            hessian[j, i] = hessian[i, j]

    return hessian





# ==================== Optimization Methods Comparison ====================



# def standard_sgd(f, grad_f, X, y, max_iterations=1000, learning_rate=0.01, tol=1e-6):
#    """
#    Standard SGD implementation without bounds.
#    """
#    n_features = X.shape[1]
#    # Random initialization
#    x = np.random.randn(n_features + 6)
  
# #    for i in range(max_iterations):
# #        gradient = grad_f(x, X, y)
# #        x_new = x - learning_rate * gradient
# #        if np.linalg.norm(x_new - x) < tol:
# #            break
# #        x = x_new
  
# #    return x, f(x, X, y)


def standard_sgd(f, grad_f, X, y,
                 start_point=None,
                 max_iterations=1000,
                 learning_rate=0.01,
                 tol=1e-6,
                 batch_size=None,
                 random_state=None):
    """
    Standard SGD with optional external start_point and batch sampling
    (supports both numpy arrays and pandas DataFrames/Series).
    
    Parameters
    ----------
    f : callable
        Objective function, f(x, X, y).
    grad_f : callable
        Gradient function, grad_f(x, X_batch, y_batch).
    X : array-like or DataFrame
        Feature matrix.
    y : array-like or Series
        Target vector.
    start_point : array-like or None
        Initial guess for x. If None, randomly initialized.
    max_iterations : int
        Maximum number of SGD iterations.
    learning_rate : float
        Step size.
    tol : float
        Tolerance for convergence.
    batch_size : int or None
        Mini-batch size. If None, use full batch.
    random_state : int or None
        Random seed.
    
    Returns
    -------
    x : array
        Optimized parameters.
    loss : float
        Final loss value.
    """
    rng = np.random.default_rng(random_state)
    n_samples, n_features = X.shape

    # Initialize starting point
    if start_point is not None:
        x = np.array(start_point, dtype=float)
    else:
        x = np.random.randn(n_features + 6)

    for i in range(max_iterations):
        # Sample batch
        if batch_size is None or batch_size >= n_samples:
            X_batch, y_batch = X, y
        else:
            idx = rng.choice(n_samples, size=batch_size, replace=False)
            if isinstance(X, (pd.DataFrame, pd.Series)):
                X_batch = X.iloc[idx]
            else:
                X_batch = X[idx]
            
            if isinstance(y, (pd.DataFrame, pd.Series)):
                y_batch = y.iloc[idx]
            else:
                y_batch = y[idx]

        # Gradient step
        grad = grad_f(x, X_batch, y_batch)
        x_new = x - learning_rate * grad

        # Convergence check
        if np.linalg.norm(x_new - x) < tol:
            x = x_new
            break

        x = x_new

    return x, f(x, X, y)



def de_only_search(f, X, y, n_features, max_iterations=100):
   """
   Differential evolution only, without SGD refinement.
   """
   # Define bounds
   lower_bounds = [-100] * 6 + [-100] * n_features
   upper_bounds = [100] * 6 + [100] * n_features
   bounds = rearrange_bounds((lower_bounds, upper_bounds))
  
   def f_wrapper(beta):
       return f(beta, X, y)
  
   result = differential_evolution(
       f_wrapper,
       bounds,
       maxiter=max_iterations,
       popsize=100,
       #strategy='best1bin',
       #tol=1e-7, 
       #mutation=(0.5, 1),
       #recombination=0.7
   )
  
   return result.x, result.fun


import cma

def cma_es_search(f, X, y, n_features, max_iterations=100):
    """
    CMA-ES based global optimization for the pipeline.
    
    Parameters:
    - f: Objective function (like pipeline_with_soft_parameters)
    - X, y: Data
    - n_features: Number of features in X
    - max_iterations: Maximum generations (iterations)
    
    Returns:
    - best_params: Best found parameters
    - best_loss: Corresponding loss
    """
    # Define bounds
    lower_bounds = [-100] * 6 + [-100] * n_features
    upper_bounds = [100] * 6 + [100] * n_features
    bounds = [lower_bounds, upper_bounds]
    
    def f_wrapper(beta):
        return f(beta, X, y)
    
    # Initial guess: random near zero
    x0 = np.random.uniform(-1, 1, size=len(lower_bounds))
    sigma0 = 5.0  # Initial search radius
    
    # Run CMA-ES
    es = cma.CMAEvolutionStrategy(
        x0, sigma0,
        {
            'bounds': bounds,
            'maxfevals': max_iterations * len(x0),  # max_iterations generations roughly
            'popsize': 30000,
            'verb_disp': 0  # no printing
        }
    )
    
    es.optimize(f_wrapper)
    
    best_params = es.result.xbest
    best_loss = es.result.fbest
    
    return best_params, best_loss


from pyswarm import pso

def pso_search(f, X, y, n_features, max_iterations=100):
    """
    Particle Swarm Optimization (PSO) for the pipeline.
    
    Parameters:
    - f: Objective function
    - X, y: Data
    - n_features: Number of features in X
    - max_iterations: Maximum generations
    
    Returns:
    - best_params: Best found parameters
    - best_loss: Corresponding loss
    """
    # Define bounds
    lower_bounds = [-10] * 6 + [-100] * n_features
    upper_bounds = [10] * 6 + [100] * n_features
    
    def f_wrapper(beta):
        return f(beta, X, y)
    
    # Run PSO
    best_params, best_loss = pso(
        f_wrapper,
        lb=lower_bounds,
        ub=upper_bounds,
        maxiter=max_iterations,
        swarmsize=300,  # Number of particles
        debug=False
    )
    
    return best_params, best_loss



# ==================== Main Optimization Function ====================




def optimize_full_pipeline(X_raw, y, n_features=None, max_iterations=1000, batch_size=32):
   """
   Optimize the full pipeline including imputation, scaling, and regression parameters.
  
   Parameters:
   - X_raw: Raw input data with missing values
   - y: Target values
   - n_features: Number of features in X (automatically inferred if None)
   - max_iterations: Maximum number of iterations for the optimization
  
   Returns:
   - Optimal parameters and minimum objective value
   """
   if n_features is None:
       if isinstance(X_raw, pd.DataFrame):
           n_features = X_raw.shape[1]
       else:
           n_features = X_raw.shape[1]
  
   # Total parameters: 3 for imputation + 3 for scaling + n_features for regression
   n_params = 6 + n_features
  
   # Define bounds for differential evolution
   # Soft parameters can be in a wider range since they'll be passed through softmax
   lower_bounds = [-10] * 6 + [-100] * n_features
   upper_bounds = [10] * 6 + [100] * n_features
  
   # Use regular SGD optimization with multiple starting points instead of multiprocessing
   best_params, min_value = sgd_opt_global_search(
       lower_bounds, upper_bounds,
       5,  # Reduced number of segments for faster execution
       pipeline_with_soft_parameters,
       pipeline_gradient,
       pipeline_hessian,
       de_search_method3,
       X_raw, y,
       learning_rate=0.01,
       max_iterations=max_iterations,
       tol=1e-6,
       batch_size=batch_size
   )
  
   return best_params, min_value




# ==================== Parameter Interpretation Function ====================




def interpret_parameters(params, feature_names=None):
   """
   Interpret the optimized parameters.
  
   Parameters:
   - params: Optimized parameters
   - feature_names: Names of features for regression coefficients
  
   Returns:
   - Dictionary with interpreted parameters
   """
   impute_weights = softmax(params[0:3])
   scale_weights = softmax(params[3:6])
   regression_params = params[6:]
  
   impute_methods = ['mean', 'median', 'knn']
   scale_methods = ['standard', 'minmax', 'maxabs']
  
   interpretation = {
       'imputation': {method: weight for method, weight in zip(impute_methods, impute_weights)},
       'scaling': {method: weight for method, weight in zip(scale_methods, scale_weights)},
       'regression': {}
   }
  
   if feature_names is not None:
       interpretation['regression'] = {
           name: coef for name, coef in zip(feature_names, regression_params)
       }
   else:
       interpretation['regression'] = {
           f'feature_{i}': coef for i, coef in enumerate(regression_params)
       }
  
   return interpretation




# ==================== Visualization Functions ====================




def plot_optimization_results(results, true_beta=None):
    """
    Create visualizations for the optimization results comparison.

    Parameters:
    - results: Dictionary of optimization results
    - true_beta: Optional array of true regression coefficients for comparison
    """
    import matplotlib.pyplot as plt
    import matplotlib
    import numpy as np
    matplotlib.use('Agg')  # Use non-interactive backend

    successful_methods = {k: v for k, v in results.items() if v["success"]}
    if not successful_methods:
        print("No successful optimization methods to plot.")
        return

    # === (1) First big comparison figure ===
    fig = plt.figure(figsize=(16, 12))

    methods = list(successful_methods.keys())
    times = [successful_methods[m]["time"] for m in methods]
    losses = [successful_methods[m]["loss"] for m in methods]

    # 1.1 Scatter Time vs Loss
    ax1 = fig.add_subplot(2, 2, 1)
    sc = ax1.scatter(times, losses, s=100, c=range(len(methods)), cmap='viridis', alpha=0.7)
    for i, method in enumerate(methods):
        ax1.annotate(method, (times[i], losses[i]), fontsize=9,
                     xytext=(5, 5), textcoords='offset points')
    ax1.set_xlabel('Execution Time (seconds)')
    ax1.set_ylabel('Loss Value')
    ax1.set_title('Optimization Performance: Time vs. Loss')
    ax1.grid(True, linestyle='--', alpha=0.7)

    # 1.2 Barplot of Time
    ax2 = fig.add_subplot(2, 2, 2)
    colors = plt.cm.viridis(np.linspace(0, 1, len(methods)))
    bars = ax2.bar(methods, times, color=colors, alpha=0.7)
    ax2.set_xlabel('Method')
    ax2.set_ylabel('Time (seconds)')
    ax2.set_title('Execution Time Comparison')
    ax2.set_xticklabels(methods, rotation=45, ha='right')
    ax2.grid(True, axis='y', linestyle='--', alpha=0.7)
    for bar in bars:
        height = bar.get_height()
        ax2.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3), textcoords="offset points",
                     ha='center', va='bottom')

    # 1.3 Barplot of Loss
    ax3 = fig.add_subplot(2, 2, 3)
    bars = ax3.bar(methods, losses, color=colors, alpha=0.7)
    ax3.set_xlabel('Method')
    ax3.set_ylabel('Loss Value')
    ax3.set_title('Loss Value Comparison')
    ax3.set_xticklabels(methods, rotation=45, ha='right')
    ax3.grid(True, axis='y', linestyle='--', alpha=0.7)
    for bar in bars:
        height = bar.get_height()
        ax3.annotate(f'{height:.4f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3), textcoords="offset points",
                     ha='center', va='bottom')

    fig.suptitle('Optimization Methods Comparison', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig('optimization_comparison_overall.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

    print("Saved: optimization_comparison_overall.png")

    # === (2) Now separately plot for each method that has interpretation ===
    for method_name, method_result in successful_methods.items():
        interp = method_result.get('interp', None)

        # Try interpreting if missing
        if interp is None:
            try:
                interp = interpret_parameters(method_result["params"])
            except Exception as e:
                print(f"Warning: Cannot interpret parameters for {method_name}: {e}")
                continue  # skip this method

        # Now we plot separately for this method
        fig, axs = plt.subplots(1, 3, figsize=(20, 6))

        # 2.1 Regression Coefficients
        reg_coefs = list(interp['regression'].values())
        feature_names = list(interp['regression'].keys())
        y_pos = np.arange(len(feature_names))

        axs[0].barh(y_pos, reg_coefs, color='skyblue', alpha=0.8)
        if true_beta is not None and len(true_beta) == len(reg_coefs):
            axs[0].barh(y_pos, true_beta, color='red', alpha=0.3, label='True Coefficients')
            axs[0].legend()

        axs[0].set_yticks(y_pos)
        axs[0].set_yticklabels(feature_names)
        axs[0].set_xlabel('Coefficient Value')
        axs[0].set_title(f'Regression Coefficients ({method_name})')
        axs[0].grid(True, axis='x', linestyle='--', alpha=0.7)

        # 2.2 Imputation Weights
        imp_methods = list(interp['imputation'].keys())
        imp_weights = list(interp['imputation'].values())

        axs[1].pie(imp_weights, labels=imp_methods, autopct='%1.1f%%', startangle=90,
                   colors=plt.cm.Blues(np.linspace(0.3, 0.7, len(imp_methods))))
        axs[1].set_title('Imputation Method Weights')

        # 2.3 Scaling Weights
        scale_methods = list(interp['scaling'].keys())
        scale_weights = list(interp['scaling'].values())

        axs[2].pie(scale_weights, labels=scale_methods, autopct='%1.1f%%', startangle=90,
                   colors=plt.cm.Greens(np.linspace(0.3, 0.7, len(scale_methods))))
        axs[2].set_title('Scaling Method Weights')

        fig.suptitle(f'Parameter Interpretations for {method_name}', fontsize=18)
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        save_name = f'interpretation_{method_name.replace(" ", "_")}.png'
        fig.savefig(save_name, dpi=300, bbox_inches='tight')
        plt.close(fig)

        print(f"Saved: {save_name}")

    return True




def run_multiple_sgds_and_collect_points_and_signs(f, grad_f, X, y, num_models=10, steps_per_model=100, learning_rate=0.01, param_dim=None, low=-1.0, high=1.0):
    """
    Run multiple SGD trajectories, collecting both parameter vectors and their sign vectors.

    Args:
    - f: Objective function
    - grad_f: Gradient function
    - X, y: Dataset
    - num_models: Number of SGD initializations
    - steps_per_model: Number of steps per SGD
    - learning_rate: Learning rate for SGD
    - param_dim: Dimension of parameter vector
    - low, high: Range for uniform initialization

    Returns:
    - all_points_array: shape=(num_models * steps_per_model, param_dim), collected parameter vectors
    - all_signs_array: shape=(num_models * steps_per_model, param_dim), corresponding sign vectors
    """
    if param_dim is None:
        n_features = X.shape[1]
        param_dim = n_features + 6  # Number of parameters in the pipeline

    all_points = []  # Store raw parameter vectors
    all_signs = []   # Store corresponding sign vectors

    for model_idx in range(num_models):
        # Uniform random initialization
        x = np.random.uniform(low=low, high=high, size=param_dim)

        for step in range(steps_per_model):
            gradient = grad_f(x, X, y)  # Compute the gradient
            x_new = x - learning_rate * gradient  # SGD update

            all_points.append(x_new.copy())  # Save the updated parameter vector
            all_signs.append(np.sign(x_new))  # Save the sign vector

            x = x_new  # Move to the new point

    all_points_array = np.array(all_points)
    all_signs_array = np.array(all_signs)

    return all_points_array, all_signs_array


def extrapolate_with_ar(all_points_array, extrapolate_steps=5000, lags=5):
    """
    Use an AR model to extrapolate from existing points and extend exploration coverage.

    Args:
    - all_points_array: shape=(n_samples, param_dim), input parameter vectors
    - extrapolate_steps: Number of future points to generate
    - lags: Number of lags to use in AR modeling

    Returns:
    - extrapolated_points: shape=(extrapolate_steps, param_dim), generated future points
    """
    n_samples, n_dims = all_points_array.shape

    extrapolated_points = np.zeros((extrapolate_steps, n_dims))

    for dim in range(n_dims):
        # Extract time series for each dimension
        series = all_points_array[:, dim]

        try:
            model = AutoReg(series, lags=lags, old_names=False)
            model_fit = model.fit()

            # Predict future points
            preds = model_fit.predict(start=n_samples, end=n_samples + extrapolate_steps - 1)

            extrapolated_points[:, dim] = preds

        except Exception as e:
            print(f"Failed to fit AR model for dimension {dim}: {e}")
            # If fitting fails, extend using the last known value
            extrapolated_points[:, dim] = series[-1]

    return extrapolated_points


def evaluate_sign_prediction_with_groundtruth(f, grad_f, X, y, future_points, future_signs):
    """
    Recompute true gradient signs at future points and compare with predicted signs to evaluate accuracy.

    Args:
    - f: Objective function
    - grad_f: Gradient function
    - X, y: Training dataset
    - future_points: Extrapolated parameter vectors
    - future_signs: Predicted sign vectors

    Returns:
    - accuracy: Scalar value representing overall prediction accuracy
    """
    n_points = future_points.shape[0]
    n_dims = future_points.shape[1]

    true_signs = np.zeros_like(future_signs)

    for i in range(n_points):
        gradient = grad_f(future_points[i], X, y)  # Compute true gradient
        true_sign = np.sign(gradient)  # Take the sign
        true_signs[i] = true_sign

    correct = (true_signs == future_signs).astype(int)  # Element-wise comparison

    accuracy = correct.mean()

    return accuracy


def test_usage():
    """
    Full test: 
    - Run multiple SGD paths
    - Extrapolate points with AR models
    - Evaluate sign prediction accuracy against true gradients
    """

    # Collect SGD exploration points and their signs
    original_points, signs = run_multiple_sgds_and_collect_points_and_signs(
        pipeline_with_soft_parameters,
        pipeline_gradient,
        X_train, y_train,
        num_models=10,
        steps_per_model=100,
        learning_rate=0.01
    )

    # Extrapolate future points
    future_points = extrapolate_with_ar(original_points, extrapolate_steps=1000, lags=5)
    future_signs = np.sign(future_points)

    # Evaluate accuracy of extrapolated signs
    accuracy = evaluate_sign_prediction_with_groundtruth(
        pipeline_with_soft_parameters,
        pipeline_gradient,
        X_train, y_train,
        future_points,
        future_signs
    )

    print(f"Accuracy: {accuracy*100:.2f}%")


###################### predictive SGD
############################################ 
############################################ 
############################################ 
############################################ 
############################################ 


def encode_gradient_sign(gradient, threshold=1e-6):
    """
    Encode gradient signs into categorical values:
    -1: negative gradient
     0: near-zero gradient (within |threshold|)
     1: positive gradient
    """
    signs = np.zeros_like(gradient, dtype=int)
    signs[gradient < -threshold] = -1
    signs[gradient > threshold] = 1
    return signs



def collect_gradient_history(initial_points, f, grad_f, X, y,
                             max_iterations=1000,
                             learning_rate=0.01,
                             tol=1e-6,
                             batch_size=None,
                             random_state=None):
    """
    Run (mini-batch) SGD from each point in `initial_points` and collect:
      - x_history:     all visited x
      - loss_history:  all loss values
      - gradient_signs_history: gradient-sign vectors at each step

    Returns:
      x_history, loss_history, gradient_signs_history
    """
    rng = np.random.RandomState(random_state)
    n_samples = X.shape[0]
    
    gradient_signs_history = []
    x_history = []
    loss_history = []
    
    for init_x in initial_points:
        x = np.array(init_x, dtype=float)
        
        for _ in range(max_iterations):
            # sample a batch
            if batch_size is None or batch_size >= n_samples:
                X_batch, y_batch = X, y
            else:
                idx = rng.choice(n_samples, size=batch_size, replace=False)
                if isinstance(X, pd.DataFrame):
                    X_batch = X.iloc[idx]
                else:
                    X_batch = X[idx]
                if isinstance(y, pd.Series):
                    y_batch = y.iloc[idx]
                else:
                    y_batch = y[idx]
            
            # compute gradient (on mini-batch)
            gradient = grad_f(x, X_batch, y_batch)
            
            # compute FULL loss (on full dataset!)
            full_loss = f(x, X, y)

            # record history
            gradient_signs_history.append(encode_gradient_sign(gradient))
            x_history.append(x.copy())
            loss_history.append(full_loss)
            
            # step
            x_new = x - learning_rate * gradient
            if np.linalg.norm(x_new - x) < tol:
                x = x_new
                break
            x = x_new

    return x_history, loss_history, gradient_signs_history



def train_ar_model(sign_history, max_lag=3):
    """
    Train an AR(p) model on each parameter's gradient-sign history,
    selecting the order p ∈ [1..max_lag] that gives the lowest AIC.
    """
    # Convert sign_history to numpy array if it's not already
    sign_history = np.asarray(sign_history, dtype=float)
    
    # Handle different shapes of sign_history
    if len(sign_history.shape) == 1:  # Single parameter case
        n_steps = len(sign_history)
        n_params = 1
        sign_history = sign_history.reshape(n_steps, 1)
    else:
        n_steps, n_params = sign_history.shape
    
    models = []

    for i in range(n_params):
        y = sign_history[:, i]
        
        # need at least p+1 points to fit AR(p)
        if n_steps < 2:
            models.append(None)
            continue
        
        best_aic = np.inf
        best_fit = None
        
        # try orders 1..max_lag (but not exceeding n_steps-1)
        for p in range(1, min(max_lag, n_steps-1) + 1):
            try:
                mod = ARIMA(y, order=(p, 0, 0))
                fit = mod.fit()
                if fit.aic < best_aic:
                    best_aic = fit.aic
                    best_fit = fit
            except Exception as e:
                # skip orders that fail
                continue
        
        models.append(best_fit)

    return models

def train_arma_model(sign_history, p=3, q=1):
    """
    Train an ARMA(p,q) model on each parameter's gradient-sign history.
    """
    # Convert sign_history to numpy array if it's not already
    sign_history = np.asarray(sign_history, dtype=float)
    
    # Handle different shapes of sign_history
    if len(sign_history.shape) == 1:  # Single parameter case
        n_steps = len(sign_history)
        n_params = 1
        sign_history = sign_history.reshape(n_steps, 1)
    else:
        n_steps, n_params = sign_history.shape
    
    models = []
    
    for i in range(n_params):
        y = sign_history[:, i]
        
        # Need enough points to fit the model
        if len(y) > p + q + 1:
            try:
                mod = ARIMA(y, order=(p, 0, q))
                fit = mod.fit()
                models.append(fit)
            except Exception as e:
                print(f"Error fitting ARMA model for parameter {i}: {e}")
                models.append(None)
        else:
            models.append(None)
    
    return models

def predict_next_sign_changes(models, last_signs, model_type='AR'):
    """
    Predict the next gradient sign changes using trained time series models.
    """
    # Convert last_signs to numpy array if it's not already
    last_signs = np.asarray(last_signs, dtype=int)
    
    n_params = len(models)
    predicted_signs = np.zeros(n_params, dtype=int)
    
    for i in range(n_params):
        if models[i] is not None:
            try:
                # Use model to predict next sign
                prediction = models[i].forecast(1)
                # Convert continuous prediction to sign (-1, 0, 1)
                if prediction[0] < -0.33:
                    predicted_signs[i] = -1
                elif prediction[0] > 0.33:
                    predicted_signs[i] = 1
                else:
                    predicted_signs[i] = 0
            except Exception as e:
                # Default to last known sign on error
                if i < len(last_signs):
                    predicted_signs[i] = last_signs[i]
        else:
            # No model available, use last known sign
            if i < len(last_signs):
                predicted_signs[i] = last_signs[i]
    
    return predicted_signs

def determine_convex_region(predicted_signs, current_point, step_size=1.0):
    """
    Determine bounds for a convex region based on predicted gradient sign changes.
    """
    n_params = len(current_point)
    lower_bounds = np.zeros(n_params)
    upper_bounds = np.zeros(n_params)
    
    for i in range(n_params):
        if predicted_signs[i] < 0:  # Negative gradient (moving right decreases function)
            lower_bounds[i] = current_point[i]
            upper_bounds[i] = current_point[i] + step_size
        elif predicted_signs[i] > 0:  # Positive gradient (moving left decreases function)
            lower_bounds[i] = current_point[i] - step_size
            upper_bounds[i] = current_point[i]
        else:  # Near-zero gradient (flat region)
            lower_bounds[i] = current_point[i] - step_size/2
            upper_bounds[i] = current_point[i] + step_size/2
    
    return lower_bounds, upper_bounds

def constrained_sgd(f, grad_f, start_point, bounds, X, y, learning_rate=0.01, max_steps=100, tol=1e-6, batch_size=None, random_state=None):
    """
    Run SGD with constraints to keep optimization within a predicted convex region.
    """
    lower_bounds, upper_bounds = bounds
    x = np.array(start_point, dtype=float)
    
    rng = np.random.RandomState(random_state)
    n_samples = X.shape[0]
    
    for i in range(max_steps):
        # Sample a batch if batch_size is specified
        if batch_size is not None and batch_size < n_samples:
            idx = rng.choice(n_samples, size=batch_size, replace=False)
            X_batch, y_batch = X[idx], y[idx]
        else:
            X_batch, y_batch = X, y
            
        gradient = grad_f(x, X_batch, y_batch)
        x_new = x - learning_rate * gradient
        
        # Project onto the constrained region
        x_new = np.maximum(x_new, lower_bounds)
        x_new = np.minimum(x_new, upper_bounds)
        
        if np.linalg.norm(x_new - x) < tol:
            break
        
        x = x_new
    
    loss = f(x, X, y)
    return x, loss

def generate_random_initial_points(n_points, n_params, lower=-10, upper=10, random_state=None):
    """
    Generate random initial points within specified bounds.
    """
    if random_state is not None:
        np.random.seed(random_state)
    return [lower + (upper - lower) * np.random.rand(n_params) for _ in range(n_points)]


def predictive_sgd_optimization(f, grad_f, X, y, initial_points, n_points=10, n_params=None, 
                              learning_rate=0.01, max_steps=100, ar_lag=3, 
                              arma_p=3, arma_q=1, region_step_size=1.0,
                              use_arma=True, batch_size=None, random_state=42):
    """
    Execute the enhanced optimization approach with gradient sign prediction.
    
    Parameters:
    - f: Objective function
    - grad_f: Gradient function
    - X, y: Data matrices
    - n_points: Number of random initial points
    - n_params: Number of parameters (inferred if None)
    - learning_rate: Learning rate for SGD
    - max_steps: Maximum number of steps for each SGD run
    - ar_lag: Maximum lag parameter for AR models
    - arma_p, arma_q: Order parameters for ARMA models
    - region_step_size: Step size for determining convex regions
    - use_arma: Whether to use ARMA models (True) or AR models (False)
    - batch_size: Size of mini-batches for SGD (None for full batch)
    - random_state: Random seed for reproducibility
    
    Returns:
    - best_params: Best parameter values found
    - best_loss: Best loss value found
    """
    if n_params is None:
        # Infer parameter count from X
        if isinstance(X, pd.DataFrame):
            n_features = X.shape[1]
        else:
            n_features = X.shape[1]
        n_params = n_features  # Just the regression parameters, not pipeline parameters
    
    # Convert pandas objects to numpy arrays if needed
    if isinstance(X, pd.DataFrame):
        X = X.values
    if isinstance(y, pd.Series):
        y = y.values
    
    print(f"Running SGD optimization with {n_points} initial points, {n_params} parameters")
    
    # Step 1: Generate random initial points
    #initial_points = generate_random_initial_points(n_points, n_params, random_state=random_state)
    
    # Step 2: Run SGD from each initial point and collect gradient histories
    print("Collecting gradient history...")
    x_history, loss_history, gradient_signs_history = collect_gradient_history(
        initial_points, f, grad_f, X, y,
        max_iterations=max_steps,
        learning_rate=learning_rate,
        batch_size=batch_size,
        random_state=random_state
    )
    
    # Process the results from collect_gradient_history to find initial solutions
    print("Processing initial SGD results...")
    results = []
    
    # Find the end indices for each initial point's optimization trajectory
    point_indices = []
    current_idx = 0
    point_count = 0
    
    for i in range(len(x_history)):
        if point_count < n_points and i % max_steps == 0:
            # Start of a new initial point's trajectory
            point_indices.append(i)
            point_count += 1
    
    # Add the end index
    point_indices.append(len(x_history))
    
    # Extract results for each initial point
    for i in range(len(point_indices) - 1):
        start_idx = point_indices[i]
        end_idx = point_indices[i + 1]
        
        if start_idx < end_idx:
            # Use the last point in the trajectory as the result
            best_idx_in_range = start_idx + np.argmin(loss_history[start_idx:end_idx])
            x_final = x_history[best_idx_in_range]
            loss = loss_history[best_idx_in_range]
            results.append((x_final, loss))
    
    # Step 3: Train time series models for gradient sign prediction
    print("Training time series models for gradient sign prediction...")
    
    # Make sure gradient_signs_history is properly formatted for model training
    if isinstance(gradient_signs_history, list):
        gradient_signs_history = np.array(gradient_signs_history)
    
    if use_arma:
        print(f"Using ARMA({arma_p},{arma_q}) models...")
        ts_models = train_arma_model(gradient_signs_history, p=arma_p, q=arma_q)
    else:
        print(f"Using AR({ar_lag}) models...")
        ts_models = train_ar_model(gradient_signs_history, max_lag=ar_lag)
    
    # Step 4: Predictive SGD with region constraints
    print("Running predictive SGD with region constraints...")
    enhanced_results = []
    
    for i, x0 in enumerate(initial_points):
        print(f"  - Enhanced optimization from point {i+1}/{n_points}")
        current_point = x0.copy()
        current_loss = f(current_point, X, y)
        current_region_step_size = region_step_size
        
        for step in range(max_steps):
            # Calculate current gradient and its sign
            current_gradient = grad_f(current_point, X, y)
            current_signs = encode_gradient_sign(current_gradient)
            
            # Predict next gradient sign changes
            predicted_signs = predict_next_sign_changes(ts_models, current_signs, 
                                                      'ARMA' if use_arma else 'AR')
            
            # Determine convex region based on predictions
            lower_bounds, upper_bounds = determine_convex_region(
                predicted_signs, current_point, current_region_step_size)
            
            # Run constrained SGD within the predicted region
            new_point, new_loss = constrained_sgd(
                f, grad_f, current_point, (lower_bounds, upper_bounds), 
                X, y, learning_rate, max_steps=10, batch_size=batch_size, random_state=random_state)
            
            # Check if we've improved
            if new_loss < current_loss - 1e-6:
                current_point = new_point
                current_loss = new_loss
                # Reset step size if we found a better point
                current_region_step_size = region_step_size
            else:
                # If no improvement, we may be in a local minimum or need a different region
                current_region_step_size *= 2  # Increase step size to explore larger regions
                if current_region_step_size > 10:
                    break  # Stop if region gets too large
        
        enhanced_results.append((current_point, current_loss))
    
    # Step 5: Find the best result across all optimization runs
    all_results = results + enhanced_results
    
    if all_results:
        best_idx = np.argmin([loss for _, loss in all_results])
        best_params, best_loss = all_results[best_idx]
        print(f"Best loss: {best_loss}")
    else:
        # Handle the case where no results were found
        best_params = np.zeros(n_params)
        best_loss = float('inf')
        print("Warning: No valid results found")
    
    return best_params, best_loss

# Example usage with the pipeline

def pipeline_objective(params, X_raw, y):
    """Wrapper for pipeline_with_soft_parameters"""
    from sklearn.impute import KNNImputer
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler
    
    # Make sure X_raw is a DataFrame
    if not isinstance(X_raw, pd.DataFrame):
        if isinstance(X_raw, np.ndarray):
            X_raw = pd.DataFrame(X_raw)
        else:
            raise ValueError("X_raw must be a DataFrame or numpy array")
    
    def softmax(x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()
    
    # Extract and normalize parameters
    impute_weights = softmax(params[0:3])
    scale_weights = softmax(params[3:6])
    regression_params = params[6:]
    
    # Imputation step - apply weighted combination
    def impute_na_values(df, method='mean', n_neighbors=5):
        if method == 'mean':
            return df.apply(lambda col: col.fillna(col.mean()) if col.isna().any() else col, axis=0)
        elif method == 'median':
            return df.apply(lambda col: col.fillna(col.median()) if col.isna().any() else col, axis=0)
        elif method == 'knn':
            imputer = KNNImputer(n_neighbors=n_neighbors)
            return pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
        else:
            raise ValueError("Method must be one of 'mean', 'median', or 'knn'.")
    
    X_imp_mean = impute_na_values(X_raw, method='mean')
    X_imp_median = impute_na_values(X_raw, method='median')
    X_imp_knn = impute_na_values(X_raw, method='knn')
    
    X_imputed = pd.DataFrame(
        impute_weights[0] * X_imp_mean.values +
        impute_weights[1] * X_imp_median.values +
        impute_weights[2] * X_imp_knn.values,
        columns=X_raw.columns
    )
    
    # Scaling step - apply weighted combination
    def standard_scale_df(df):
        scaler = StandardScaler()
        scaled = scaler.fit_transform(df)
        return pd.DataFrame(scaled, columns=df.columns, index=df.index)
    
    def minmax_scale_df(df):
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(df)
        return pd.DataFrame(scaled, columns=df.columns, index=df.index)
    
    def maxabs_scale_df(df):
        scaler = MaxAbsScaler()
        scaled = scaler.fit_transform(df)
        return pd.DataFrame(scaled, columns=df.columns, index=df.index)
    
    X_scale_standard = standard_scale_df(X_imputed)
    X_scale_minmax = minmax_scale_df(X_imputed)
    X_scale_maxabs = maxabs_scale_df(X_imputed)
    
    X_scaled = pd.DataFrame(
        scale_weights[0] * X_scale_standard.values +
        scale_weights[1] * X_scale_minmax.values +
        scale_weights[2] * X_scale_maxabs.values,
        columns=X_raw.columns
    )
    
    # Regression step with non-convex objective
    X_final = X_scaled.values
    
    def nonconvex_f(beta, X, y):
        residual = y - np.dot(X, beta)
        residual_sq = residual ** 2
        cost = 0.5 * np.sum(residual_sq / (1 + residual_sq))
        return cost
    
    return nonconvex_f(regression_params, X_final, y)

def pipeline_gradient_new(params, X_raw, y, epsilon=1e-6):
    """
    Compute numerical gradient for the pipeline parameters.
    Uses central difference approximation.
    """
    n_params = len(params)
    grad = np.zeros(n_params)
    
    for i in range(n_params):
        params_plus = params.copy()
        params_plus[i] += epsilon
        
        params_minus = params.copy()
        params_minus[i] -= epsilon
        
        f_plus = pipeline_objective(params_plus, X_raw, y)
        f_minus = pipeline_objective(params_minus, X_raw, y)
        
        grad[i] = (f_plus - f_minus) / (2 * epsilon)
    
    return grad

def optimize_pipeline_with_predictive_sgd(X_train, y_train, initial_points, n_points=5, max_steps=50, batch_size=32, use_arma=True):
    """
    Apply predictive SGD optimization to the pipeline with soft parameters.
    """
    # For the pipeline, we need 6 + n_features parameters
    n_features = X_train.shape[1]
    n_params = 6 + n_features  # 3 for imputation, 3 for scaling, n_features for regression
    
    # Run the optimization
    best_params, best_loss = predictive_sgd_optimization(
        f=pipeline_with_soft_parameters,
        grad_f=pipeline_gradient,
        X=X_train,
        y=y_train,
        initial_points=initial_points,
        n_points=n_points,
        n_params=n_params,
        max_steps=max_steps,
        use_arma=use_arma,
        learning_rate=0.01,
        region_step_size=0.5,
        batch_size=batch_size
    )
    
    # Interpret the parameters
    interpretation = interpret_parameters(best_params, feature_names=X_train.columns)
    
    return best_params, best_loss, interpretation



def get_all_results_from_std_sgd(f, grad_f, X, y, initial_points, learning_rate=0.01, max_steps=1000, tol=1e-6, batch_size=32, random_state=None):
 loss_min = np.inf
 x_min = None
 for start_point in initial_points:
     x_est, loss_est =standard_sgd(f, grad_f, X, y, start_point, max_iterations=max_steps, learning_rate=learning_rate, tol=tol, batch_size=batch_size, random_state=random_state)
     if loss_est < loss_min:
         loss_min = loss_est
         x_min = x_est
 return x_min,loss_min




#### new method

def predict_next_signs_change(models, last_signs, max_steps = 10):
    """
    Predict the next gradient sign changes using trained time series models.
    """
    # Convert last_signs to numpy array if it's not already
    last_signs = np.asarray(last_signs, dtype=int)

    
    n_params = len(models)

    sign_num = 1

    for sign_num in range(max_steps):
      sign_num += 1
      predicted_signs = np.zeros(n_params, dtype=int)
    
      for i in range(n_params):
          if models[i] is not None:
              try:
                  # Use model to predict next sign
                  prediction = models[i].forecast(sign_num)
                  # Convert continuous prediction to sign (-1, 0, 1)
                  if prediction[0] < -0.33:
                      predicted_signs[i] = -1
                  elif prediction[0] > 0.33:
                      predicted_signs[i] = 1
                  else:
                      predicted_signs[i] = 0
              except Exception as e:
                  # Default to last known sign on error
                  if i < len(last_signs):
                      predicted_signs[i] = last_signs[i]
          else:
              # No model available, use last known sign
              if i < len(last_signs):
                  predicted_signs[i] = last_signs[i]


      if np.any(predicted_signs != last_signs):
        break
    
    return predicted_signs,sign_num


def determine_flip_region(predicted_signs,sign_num, current_point, step_size=1.0):
    """
    Determine bounds for a flip region based on predicted gradient sign changes.
    """
    n_params = len(current_point)
    lower_bounds = np.zeros(n_params)
    upper_bounds = np.zeros(n_params)
    
    for i in range(n_params):
        if predicted_signs[i] < 0:  # Negative gradient (moving right decreases function)
            lower_bounds[i] = current_point[i]
            upper_bounds[i] = current_point[i] + step_size*sign_num
        elif predicted_signs[i] > 0:  # Positive gradient (moving left decreases function)
            lower_bounds[i] = current_point[i] - step_size*sign_num
            upper_bounds[i] = current_point[i]
        else:  # Near-zero gradient (flat region)
            lower_bounds[i] = current_point[i] - step_size*sign_num/2
            upper_bounds[i] = current_point[i] + step_size*sign_num/2
    
    return lower_bounds, upper_bounds


def constrained_sgd_new(f, grad_f,current_point,bounds, X, y, learning_rate=0.01, max_steps=100, tol=1e-6, batch_size=None, random_state=None):
    """
    Run SGD with constraints to keep optimization within a predicted convex region.
    """
    lower_bounds, upper_bounds = bounds
    x = np.array(current_point, dtype=float)
    lower_bounds = np.array(lower_bounds, dtype=float)
    upper_bounds = np.array(upper_bounds, dtype=float)
    
    rng = np.random.RandomState(random_state)
    n_samples = X.shape[0]
    
    for i in range(max_steps):
        # Sample a batch if batch_size is specified
        if batch_size is not None and batch_size < n_samples:
            idx = rng.choice(n_samples, size=batch_size, replace=False)
            X_batch, y_batch = X[idx], y[idx]
        else:
            X_batch, y_batch = X, y
            
        gradient = grad_f(x, X_batch, y_batch)
        x_new = x - learning_rate * gradient
        
          
        if np.linalg.norm(x_new - x) < tol:
            break

        if np.all(x_new < lower_bounds) or np.all(x_new > upper_bounds):
            # stepping outside the convex region → stop here
            break

        x = x_new
        
           
    loss = f(x, X, y)
    return x, loss



def predictive_sgd_optimization_new(f, grad_f, X, y, initial_points, n_points=10, n_params=None, 
                              learning_rate=0.01, max_steps=100, ar_lag=3, 
                              arma_p=3, arma_q=1, region_step_size=1.0,
                              use_arma=True, batch_size=None,training_frac = 0.3, random_state=42):
    if n_params is None:
        # Infer parameter count from X
        if isinstance(X, pd.DataFrame):
            n_features = X.shape[1]
        else:
            n_features = X.shape[1]
        n_params = n_features  # Just the regression parameters, not pipeline parameters
    
    # Convert pandas objects to numpy arrays if needed
    if isinstance(X, pd.DataFrame):
        X = X.values
    if isinstance(y, pd.Series):
        y = y.values
    
    print(f"Running SGD optimization with {n_points} initial points, {n_params} parameters")
    
    # Step 1: Get the samples
    k_frac = int(len(initial_points) * training_frac)
    training_points = random.sample(initial_points, k_frac)

    test_points = []
    for point in initial_points:
        found = False
        for train_point in training_points:
            if np.array_equal(point, train_point):
                found = True
                break
        if not found:
            test_points.append(point)

    # Step 2: Run SGD from each initial point and collect gradient histories
    print("Collecting gradient history...")
    x_history, loss_history, gradient_signs_history = collect_gradient_history(
        training_points, f, grad_f, X, y,
        max_iterations=max_steps,
        learning_rate=learning_rate,
        batch_size=batch_size,
        random_state=random_state
    )

    # get the best result from training
    best_loss_index = np.argmin(loss_history)
    best_loss = loss_history[best_loss_index]
    best_params = x_history[best_loss_index]

    print(f"Best loss: {best_loss} for training")


    # Step 3: Train time series models for gradient sign prediction
    print("Training time series models for gradient sign prediction...")
    
    # Make sure gradient_signs_history is properly formatted for model training
    if isinstance(gradient_signs_history, list):
        gradient_signs_history = np.array(gradient_signs_history)
    
    if use_arma:
        print(f"Using ARMA({arma_p},{arma_q}) models...")
        ts_models = train_arma_model(gradient_signs_history, p=arma_p, q=arma_q)
    else:
        print(f"Using AR({ar_lag}) models...")
        ts_models = train_ar_model(gradient_signs_history, max_lag=ar_lag)
    
    # Step 4: Predictive SGD with region constraints
    print("Running predictive SGD with region constraints...")
    enhanced_results = []
    
    for start_point in initial_points:
        print(f"  - Enhanced optimization from point {start_point}")
        current_point = start_point.copy()
        current_loss = f(current_point, X, y)
        current_region_step_size = region_step_size
        local_max_steps = max_steps  # 每个start_point自己的局部max_steps

        while local_max_steps > 0:
            current_gradient = grad_f(current_point, X, y)
            current_signs = encode_gradient_sign(current_gradient)
            
            predicted_signs, sign_num = predict_next_signs_change(ts_models, current_signs, local_max_steps)
            local_max_steps -= sign_num
            
            lower_bounds, upper_bounds = determine_flip_region(
                predicted_signs, sign_num, current_point, current_region_step_size)

            new_point, new_loss = constrained_sgd_new(
                f, grad_f, current_point, (lower_bounds, upper_bounds),
                X, y, learning_rate, max_steps=sign_num, batch_size=batch_size, random_state=random_state)

            if new_loss < current_loss - 1e-6:
                current_point = new_point
                current_loss = new_loss
                current_region_step_size = region_step_size
        
        enhanced_results.append((current_point, current_loss))

    
    # Step 5: Find the best result across all optimization runs
    
    if enhanced_results:
        best_idx_enhanced_results = np.argmin([loss for _, loss in enhanced_results])
        best_params_enhanced_results, best_loss_enhanced_results = enhanced_results[best_idx_enhanced_results]
        print(f"Best loss for enhanced results: {best_loss_enhanced_results}")

        if best_loss_enhanced_results < best_loss:
            best_params = best_params_enhanced_results
            best_loss = best_loss_enhanced_results
        print(f"Best loss: {best_loss}")

    return best_params, best_loss



def optimize_pipeline_with_predictive_sgd_new(X_train, y_train, initial_points, n_points=5, max_steps=50, batch_size=32, use_arma=True):
    """
    Apply predictive SGD optimization to the pipeline with soft parameters.
    """
    # For the pipeline, we need 6 + n_features parameters
    n_features = X_train.shape[1]
    n_params = 6 + n_features  # 3 for imputation, 3 for scaling, n_features for regression
    
    # Run the optimization
    best_params, best_loss = predictive_sgd_optimization_new(
        f=pipeline_with_soft_parameters,
        grad_f=pipeline_gradient,
        X=X_train,
        y=y_train,
        initial_points=initial_points,
        n_points=n_points,
        n_params=n_params,
        max_steps=max_steps,
        use_arma=use_arma,
        learning_rate=0.01,
        region_step_size=0.5,
        batch_size=batch_size
    )
    
    # Interpret the parameters
    interpretation = interpret_parameters(best_params, feature_names=X_train.columns)
    
    return best_params, best_loss, interpretation




##### with curvature one 


def _slice_rows(X, y, idx):
    """
    Given X (DataFrame or ndarray) and y (Series/ndarray) and a list/array of
    integer indices, return the corresponding minibatch (Xb, yb).
    """
    if hasattr(X, "iloc"):
        Xb = X.iloc[idx]
    else:
        Xb = X[idx]
    if hasattr(y, "iloc"):
        yb = y.iloc[idx]
    else:
        yb = y[idx]
    return Xb, yb

def train_ar_model_with_curvature(sign_history, max_lag=3):
    sign_history = np.asarray(sign_history, dtype=float)
    if sign_history.ndim == 1:
        sign_history = sign_history.reshape(-1, 1)
    n_steps, n_params = sign_history.shape

    models = []
    for i in range(n_params):
        y = sign_history[:, i]
        best_fit, best_aic = None, np.inf
        if n_steps < 2:
            models.append(None)
            continue
        for p in range(1, min(max_lag, n_steps-1)+1):
            try:
                fit = ARIMA(y, order=(p,0,0)).fit(disp=False)
                if fit.aic < best_aic:
                    best_aic, best_fit = fit.aic, fit
            except:
                pass
        models.append(best_fit)
    return models

def train_arma_model_with_curvature(sign_history, p=3, q=1):
    sign_history = np.asarray(sign_history, dtype=float)
    if sign_history.ndim == 1:
        sign_history = sign_history.reshape(-1, 1)
    models = []
    for i in range(sign_history.shape[1]):
        y = sign_history[:, i]
        if len(y) > p + q + 1:
            try:
                fit = ARIMA(y, order=(p,0,q)).fit(disp=False)
                models.append(fit)
            except:
                models.append(None)
        else:
            models.append(None)
    return models

def predict_next_signs_change_with_curvature(models, last_signs, max_steps=10):
    last_signs = np.asarray(last_signs, dtype=int)
    n = len(models)
    for step in range(1, max_steps+1):
        preds = np.zeros(n, dtype=int)
        for i, m in enumerate(models):
            if m is not None:
                try:
                    val = m.forecast(step)[0]
                    preds[i] = -1 if val < -0.33 else (1 if val > 0.33 else 0)
                except:
                    preds[i] = last_signs[i]
            else:
                preds[i] = last_signs[i]
        if not np.all(preds == last_signs):
            return preds, step
    return last_signs, max_steps

def determine_flip_region_with_curvature(predicted_signs, sign_num, current_point,
                          base_step_size, cur_grad):
    n = len(current_point)
    step = base_step_size * np.linalg.norm(cur_grad)
    lb = np.zeros(n); ub = np.zeros(n)
    for i in range(n):
        if predicted_signs[i] < 0:
            lb[i] = current_point[i]
            ub[i] = current_point[i] + step * sign_num
        elif predicted_signs[i] > 0:
            lb[i] = current_point[i] - step * sign_num
            ub[i] = current_point[i]
        else:
            half = step * sign_num / 2
            lb[i] = current_point[i] - half
            ub[i] = current_point[i] + half
    return lb, ub

def constrained_sgd_with_curvature(f, grad_f, current_point, bounds, X, y,
                    base_lr=0.01, curvature_smooth=1.0,
                    max_steps=100, tol=1e-6,
                    batch_size=None, random_state=None):
    lb, ub = map(np.array, bounds)
    x = np.array(current_point, dtype=float)
    prev_x = prev_grad = None

    rng = np.random.RandomState(random_state)
    n_samples = X.shape[0]

    for _ in range(max_steps):
        if batch_size and batch_size < n_samples:
            idx = rng.choice(n_samples, size=batch_size, replace=False)
            Xb, yb = X[idx], y[idx]
        else:
            Xb, yb = X, y

        grad = grad_f(x, Xb, yb)
        if prev_x is not None:
            num = np.linalg.norm(grad - prev_grad)
            den = np.linalg.norm(x - prev_x) + 1e-8
            curvature = num / den
        else:
            curvature = 0.0

        lr_t = base_lr / (1.0 + curvature_smooth * curvature)
        x_new = x - lr_t * grad

        if (np.all(x_new < lb) or np.all(x_new > ub)
            or np.linalg.norm(x_new - x) < tol):
            break

        prev_x, prev_grad, x = x, grad, x_new

    return x, f(x, X, y)

def collect_gradient_history_with_curvature(
    initial_points, f, grad_f, X, y,
    max_iterations=1000,
    base_lr=0.01,
    curvature_smooth=1.0,
    tol=1e-6,
    batch_size=None,
    random_state=None
):
    rng = np.random.RandomState(random_state)
    n_samples = X.shape[0]
    gradient_signs_history, x_history, loss_history = [], [], []

    for init_x in initial_points:
        x = np.array(init_x, dtype=float)
        prev_x = prev_grad = None

        for _ in range(max_iterations):
            # ---- use safe slicing ----
            if batch_size and batch_size < n_samples:
                idx = rng.choice(n_samples, size=batch_size, replace=False)
                Xb, yb = _slice_rows(X, y, idx)
            else:
                Xb, yb = X, y

            grad = grad_f(x, Xb, yb)

            # !!! compute loss on FULL DATA !!!
            full_loss = f(x, X, y)

            gradient_signs_history.append(encode_gradient_sign(grad))
            x_history.append(x.copy())
            loss_history.append(full_loss)

            # compute local curvature
            if prev_x is not None:
                num = np.linalg.norm(grad - prev_grad)
                den = np.linalg.norm(x - prev_x) + 1e-8
                curvature = num / den
            else:
                curvature = 0.0

            # adaptive learning rate
            lr_t = base_lr / (1.0 + curvature_smooth * curvature)
            x_new = x - lr_t * grad

            if np.linalg.norm(x_new - x) < tol:
                x = x_new
                break

            prev_x, prev_grad, x = x, grad, x_new

    return x_history, loss_history, gradient_signs_history


def optimize_pipeline_with_predictive_sgd_v2_curvature(X_train, y_train, initial_points,max_steps=1000, batch_size=32, use_arma=False):
    """
    Apply predictive SGD optimization to the pipeline with soft parameters.
    """
    # For the pipeline, we need 6 + n_features parameters
    n_features = X_train.shape[1]
    n_params = 6 + n_features  # 3 for imputation, 3 for scaling, n_features for regression
    
    # Run the optimization
    best_params, best_loss = predictive_sgd_optimization_with_curvature(
        f=pipeline_with_soft_parameters,
        grad_f=pipeline_gradient,
        X=X_train,
        y=y_train,
        initial_points=initial_points,
        max_steps=max_steps,
        use_arma=use_arma,
        learning_rate=0.01,
        region_step_size=1.0,
        batch_size=batch_size
    )
    

    # Interpret the parameters
    interpretation = interpret_parameters(best_params, feature_names=X_train.columns)
    
    return best_params, best_loss, interpretation



###### v3 hybrid method

def predict_all_next_signs_change_with_curvature(models, last_signs, max_steps=10):
    last_signs = np.asarray(last_signs, dtype=int)
    n = len(models)
    all_signs_lsts = []
    sign_num_lst = []
    last_sign_num = 0

    for step in range(1, max_steps+1):
        preds = np.zeros(n, dtype=int)
        for i, m in enumerate(models):
            if m is not None:
                try:
                    val = m.forecast(step)[0]
                    preds[i] = -1 if val < -0.33 else (1 if val > 0.33 else 0)
                except:
                    preds[i] = last_signs[i]
            else:
                preds[i] = last_signs[i]

        if not np.all(preds == last_signs):
            all_signs_lsts.append(preds)
            sign_num_lst.append(step - last_sign_num)
            last_sign_num = step
            last_signs = preds.copy()

    if not all_signs_lsts:
        all_signs_lsts.append(last_signs)
        sign_num_lst.append(max_steps)

    return all_signs_lsts, sign_num_lst


def de_search_method_for_hybrid(f, bounds, X, y, maxiters=100):
    lb, ub = bounds
    if lb.shape != ub.shape:
        raise ValueError(f"lb and ub must match: got {lb.shape} vs {ub.shape}")
    de_bounds = [(float(l), float(u)) for l, u in zip(lb, ub)]
    for i, (l, u) in enumerate(de_bounds):
        if u < l:
            raise ValueError(f"Bad bounds for dim {i}: {l} > {u}")
    result = differential_evolution(
        lambda beta: f(beta, X, y),
        de_bounds,
        maxiter=maxiters
    )
    return result.x, result.fun



def optimize_pipeline_with_predictive_sgd_v3_hybrid(
    X_train, y_train, initial_points,
    max_steps=1000, batch_size=32, use_arma=True
):
    n_features = X_train.shape[1]
    best_params, best_loss = predictive_sgd_optimization_with_curvature_hybrid(
        f=pipeline_with_soft_parameters,
        grad_f=pipeline_gradient,
        hessian_f=pipeline_hessian,
        X=X_train, y=y_train,
        initial_points=initial_points,
        max_steps=max_steps,
        use_arma=use_arma,
        learning_rate=0.01,
        region_step_size=1.0,
        batch_size=batch_size
    )
    interpretation = interpret_parameters(best_params, feature_names=X_train.columns)
    return best_params, best_loss, interpretation



# -------------------------------------------------------------------
# Differential‐Evolution search for hybrid method
# -------------------------------------------------------------------
def de_search_method_for_hybrid(f, bounds, X, y, maxiters=100):
    lb, ub = bounds
    if lb.shape != ub.shape:
        raise ValueError(f"lb and ub must match: got {lb.shape} vs {ub.shape}")
    de_bounds = [(float(l), float(u)) for l, u in zip(lb, ub)]
    for i, (l, u) in enumerate(de_bounds):
        if u < l:
            raise ValueError(f"Bad bounds for dim {i}: {l} > {u}")
    result = differential_evolution(
        lambda beta: f(beta, X, y),
        de_bounds,
        maxiter=maxiters
    )
    return result.x, result.fun


# -------------------------------------------------------------------
# Main predictive SGD + DE hybrid
# -------------------------------------------------------------------
def predictive_sgd_optimization_with_curvature_hybrid(
    f, grad_f, hessian_f, X, y, initial_points,
    learning_rate=0.01, curvature_smooth=1.0,
    max_steps=100, ar_lag=3, arma_p=3, arma_q=1,
    region_step_size=2.0, use_arma=True,
    training_frac=0.3, batch_size=None,
    random_state=42
):
    # Split into training/test initial points
    n_pts = len(initial_points)
    k = max(1, int(n_pts * training_frac)) 
    all_idx = list(range(n_pts))
    rnd = random.Random(random_state)
    train_idx = rnd.sample(all_idx, k)
    test_idx  = [i for i in all_idx if i not in train_idx]
    train_pts = [initial_points[i] for i in train_idx]
    test_pts  = [initial_points[i] for i in test_idx]

    # Estimate global convexity probability
    convex_count = 0
    point_is_convex = {}
    for pt in initial_points:
        H = hessian_f(pt, X, y)
        eigs = np.linalg.eigvalsh(H)
        is_conv = bool(np.all(eigs >= 0))
        point_is_convex[tuple(pt)] = is_conv
        convex_count += is_conv
    convex_prob     = convex_count / n_pts
    non_convex_prob = 1 - convex_prob

    # 1) Collect history on training points
    x_hist, loss_hist, sign_hist = collect_gradient_history_with_curvature(
        train_pts, f, grad_f, X, y,
        max_iterations=max_steps,
        base_lr=learning_rate*50,
        curvature_smooth=0,
        batch_size=batch_size,
        random_state=random_state
    )
    best_i = np.argmin(loss_hist)
    best_params, best_loss = x_hist[best_i], loss_hist[best_i]

    # 2) Fit time‐series models
    ts_models = (train_arma_model_with_curvature(sign_hist, p=arma_p, q=arma_q)
                 if use_arma else
                 train_ar_model_with_curvature(sign_hist, max_lag=ar_lag))

    # 3) For each test point, choose SGD vs DE
    enhanced = []
    for pt in test_pts:
        cur_pt   = np.array(pt, dtype=float)
        cur_loss = f(cur_pt, X, y)

        is_conv = point_is_convex[tuple(pt)]
        prob    = convex_prob if is_conv else non_convex_prob

        init_grad = grad_f(cur_pt, X, y)
        init_sign = encode_gradient_sign(init_grad)
        sign_lists, spans = predict_all_next_signs_change_with_curvature(
            ts_models, init_sign, max_steps=max_steps
        )

        if is_conv:
            # choose best region for constrained SGD
            weights     = [span * prob for span in spans]
            best_region = int(np.argmax(weights))
            signs       = sign_lists[best_region]
            ahead       = spans[best_region]

            # 3a) constrained‐SGD
            lb, ub = determine_flip_region_with_curvature(
                signs, ahead, cur_pt, region_step_size, init_grad
            )
            scaled_lr = learning_rate * np.log1p(ahead) * np.sqrt(prob)
            new_pt, new_loss = constrained_sgd_with_curvature(
                f, grad_f, cur_pt, (lb, ub), X, y,
                base_lr=scaled_lr,
                curvature_smooth=curvature_smooth,
                max_steps=ahead,
                batch_size=batch_size,
                random_state=random_state
            )
            if new_loss + 1e-6 < cur_loss:
                cur_pt, cur_loss = new_pt, new_loss
            enhanced.append((cur_pt, cur_loss))

            # 3b) DE in the other regions
            de_iters = max(1, int(10 * non_convex_prob))
            for i, (signs_o, span_o) in enumerate(zip(sign_lists, spans)):
                if i == best_region:
                    continue
                # fix spans
                lb_o, ub_o = determine_flip_region_with_curvature(
                    signs_o, 10, cur_pt, region_step_size, init_grad
                )
                x_de, loss_de = de_search_method_for_hybrid(
                    f, (lb_o, ub_o), X, y, maxiters=de_iters
                )
                enhanced.append((x_de, loss_de))
        else:
            # if not convex: just run DE on all regions
            for signs_o, span_o in zip(sign_lists, spans):
                # fix spans
                lb_o, ub_o = determine_flip_region_with_curvature(
                    signs_o, 10, cur_pt, region_step_size, init_grad
                )
                x_de, loss_de = de_search_method_for_hybrid(
                    f, (lb_o, ub_o), X, y, maxiters=np.max([int(10 * non_convex_prob),max_steps])
                )
                cur_pt = x_de
                enhanced.append((x_de, loss_de))

    # 4) pick best overall
    if enhanced:
        i_best = np.argmin([L for _, L in enhanced])
        if enhanced[i_best][1] < best_loss:
            best_params, best_loss = enhanced[i_best]

    return best_params, best_loss


# -------------------------------------------------------------------
# Wrapper to optimize your pipeline using v3 hybrid approach
# -------------------------------------------------------------------
def optimize_pipeline_with_predictive_sgd_v3_hybrid(
    X_train, y_train, initial_points,
    max_steps=1000, batch_size=32, use_arma=True
):
    n_features = X_train.shape[1]
    best_params, best_loss = predictive_sgd_optimization_with_curvature_hybrid(
        f=pipeline_with_soft_parameters,
        grad_f=pipeline_gradient,
        hessian_f=pipeline_hessian,
        X=X_train, y=y_train,
        initial_points=initial_points,
        max_steps=max_steps,
        use_arma=use_arma,
        learning_rate=0.01,
        region_step_size=1.0,
        batch_size=batch_size
    )
    interpretation = interpret_parameters(best_params, feature_names=X_train.columns)
    return best_params, best_loss, interpretation


import numpy as np
import random

# -------------------------------------------------------------------
# Helper to safely slice rows by integer indices (DataFrame or ndarray)
# -------------------------------------------------------------------
def _slice_rows(X, y, idx):
    """
    Given X (DataFrame or ndarray) and y (Series/ndarray) and a list/array of
    integer row‐indices, return (Xb, yb) for that minibatch.
    """
    if hasattr(X, "iloc"):
        Xb = X.iloc[idx]
    else:
        Xb = X[idx]
    if hasattr(y, "iloc"):
        yb = y.iloc[idx]
    else:
        yb = y[idx]
    return Xb, yb


# -------------------------------------------------------------------
# Constrained‐SGD within a param‐region, with safe slicing
# -------------------------------------------------------------------
def constrained_sgd_with_curvature(
    f, grad_f, x0, bounds, X, y,
    base_lr=0.01, curvature_smooth=1.0,
    max_steps=100, batch_size=None, random_state=None, tol=1e-6
):
    """
    Run SGD starting at x0, constrained to box bounds=(lb,ub) per‐param.
    Uses safe row‐slicing for minibatches.
    """
    lb, ub = bounds
    x = np.array(x0, dtype=float)
    rng = np.random.RandomState(random_state)
    n_samples = X.shape[0]
    prev_x = prev_grad = None

    for _ in range(max_steps):
        # minibatch
        if batch_size is not None and batch_size < n_samples:
            idx = rng.choice(n_samples, size=batch_size, replace=False)
            Xb, yb = _slice_rows(X, y, idx)
        else:
            Xb, yb = X, y

        grad = grad_f(x, Xb, yb)

        # curvature estimate
        if prev_x is not None:
            num = np.linalg.norm(grad - prev_grad)
            den = np.linalg.norm(x - prev_x) + 1e-8
            curvature = num / den
        else:
            curvature = 0.0

        lr_t = base_lr / (1.0 + curvature_smooth * curvature)
        x_new = x - lr_t * grad

        # project into [lb, ub]
        # x_new = np.minimum(np.maximum(x_new, lb), ub)

        if np.linalg.norm(x_new - x) < tol:
            x = x_new
            break

        if np.all(x_new < lb) or np.all(x_new > ub):
            # stepping outside the convex region → stop here
            break

        prev_x, prev_grad, x = x, grad, x_new

    return x, f(x, X, y)

# -------------------------------------------------------------------
# The “v2” predictive‐SGD driver, fixed for safe slicing
# -------------------------------------------------------------------
def predictive_sgd_optimization_with_curvature(
    f, grad_f, X, y, initial_points,
    learning_rate=0.01, curvature_smooth=1.0,
    max_steps=100, ar_lag=3,
    arma_p=3, arma_q=1,
    region_step_size=2.0,
    use_arma=False,
    training_frac=0.3,
    batch_size=None,
    random_state=42
):
    # 1) split initial_points into train/test
    n_pts = len(initial_points)
    k = max(1, int(n_pts * training_frac)) 
    all_idx = list(range(n_pts))
    rnd = random.Random(random_state)
    train_idx = rnd.sample(all_idx, k)
    test_idx  = [i for i in all_idx if i not in train_idx]
    train_pts = [initial_points[i] for i in train_idx]
    test_pts  = [initial_points[i] for i in test_idx]

    # 2) collect gradient history on train_pts
    x_hist, loss_hist, sign_hist = collect_gradient_history_with_curvature(
        train_pts, f, grad_f, X, y,
        max_iterations=max_steps,
        base_lr=learning_rate*50,
        curvature_smooth=curvature_smooth,
        batch_size=batch_size,
        random_state=random_state
    )
    best_i = np.argmin(loss_hist)
    best_params, best_loss = x_hist[best_i], loss_hist[best_i]

    # 3) fit AR/ARMA on sign history
    ts_models = (
        train_arma_model_with_curvature(sign_hist, p=arma_p, q=arma_q)
        if use_arma
        else train_ar_model_with_curvature(sign_hist, max_lag=ar_lag)
    )

    # 4) predictive constrained‐SGD on test_pts
    enhanced = []
    for pt in test_pts:
        cur_pt   = np.array(pt, dtype=float)
        cur_loss = f(cur_pt, X, y)
        steps_left = max_steps

        while steps_left > 0:
            grad = grad_f(cur_pt, X, y)
            signs, ahead = predict_next_signs_change_with_curvature(
                ts_models,
                encode_gradient_sign(grad),
                max_steps=steps_left
            )
            steps_left -= ahead

            lb, ub = determine_flip_region_with_curvature(
                signs, ahead, cur_pt, region_step_size, grad
            )

            new_pt, new_loss = constrained_sgd_with_curvature(
                f, grad_f, cur_pt, (lb, ub),
                X, y,
                base_lr=learning_rate * np.log1p(ahead),
                curvature_smooth=curvature_smooth,
                max_steps=ahead,
                batch_size=batch_size,
                random_state=random_state
            )

            if new_loss < cur_loss - 1e-6:
                cur_pt, cur_loss = new_pt, new_loss
            else:
                break

        enhanced.append((cur_pt, cur_loss))

    # 5) choose best overall
    if enhanced:
        i_best = np.argmin([loss for (_, loss) in enhanced])
        if enhanced[i_best][1] < best_loss:
            best_params, best_loss = enhanced[i_best]

    return best_params, best_loss



def generate_random_initial_points(n_points, n_params, lower=-100, upper=100, random_state=None):
    """
    Generate random initial points within specified bounds.
    """
    if random_state is not None:
        np.random.seed(random_state)
    return [lower + (upper - lower) * np.random.rand(n_params) for _ in range(n_points)]

# -------------------------------------------------------------------
# Example usage & comparison loop
# -------------------------------------------------------------------
def example_usage():
    import time
    import numpy as np

    # assume X_train, y_train already loaded
    X_df, y = X_train, y_train
    n_features = X_df.shape[1]
    n_params = 6 + n_features
    initial_points = generate_random_initial_points(n_points=5, n_params=n_params, random_state=42)

    methods = {
        "Our Approach (v3)": lambda: optimize_pipeline_with_predictive_sgd_v3_hybrid(
            X_df, y, initial_points, max_steps=100, batch_size=32, use_arma=False
        ),
        "Our Approach (v2)": lambda: optimize_pipeline_with_predictive_sgd_v2_curvature(
            X_df, y, initial_points, max_steps=100, batch_size=32, use_arma=False
        ),
        "Our Approach (v1)": lambda: optimize_full_pipeline(
            X_df, y, max_iterations=100, batch_size=32
        ),
        "Standard SGD": lambda: get_all_results_from_std_sgd(
            pipeline_with_soft_parameters, pipeline_gradient,
            X_df, y, initial_points, max_steps=100, batch_size=32
        ),
        "SGD with Bound": lambda: sgd_with_bound(
            pipeline_with_soft_parameters, pipeline_gradient,
            np.random.randn(n_features + 6),
            np.ones(n_features + 6) * np.inf,
            X_df, y, iterations=100, batch_size=32
        ),
        "Differential Evolution": lambda: de_only_search(
            pipeline_with_soft_parameters, X_df, y, n_features, max_iterations=100
        ),
        "CMA-ES": lambda: cma_es_search(
            pipeline_with_soft_parameters, X_df, y, n_features, max_iterations=100
        ),
        "PSO": lambda: pso_search(
            pipeline_with_soft_parameters, X_df, y, n_features, max_iterations=100
        )
    }

    results = {}
    for name, func in methods.items():
        print(f"\nRunning {name}...")
        start = time.time()
        try:
            out = func()
            if name == "Our Approach (v2)":
                params, loss, interp = out
            else:
                params, loss = out[:2]
                interp = out[2] if len(out) > 2 else None
            elapsed = time.time() - start
            results[name] = {"success": True, "time": elapsed, "loss": loss, "params": params, "interp": interp}
            print(f"  - Success: Loss={loss:.6f}, Time={elapsed:.2f}s")
        except Exception as e:
            elapsed = time.time() - start
            results[name] = {"success": False, "time": elapsed, "error": str(e)}
            print(f"  - Failed: {e} (Time={elapsed:.2f}s)")

    # comparison table & best interpretation
    plot_optimization_results(results, None)


if __name__ == "__main__":
    # load one of: diabetes, california_housing, regression, friedman1
    dataset = 'diabetes'
    if dataset == 'diabetes':
        X_train, X_test, y_train, y_test = diabetes()
    elif dataset == 'california_housing':
        X_train, X_test, y_train, y_test = california_housing()
    elif dataset == 'regression':
        X_train, X_test, y_train, y_test = regression()
    elif dataset == 'friedman1':
        X_train, X_test, y_train, y_test = friedman1()
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    example_usage()


# python py_files/regression_ver2.py
