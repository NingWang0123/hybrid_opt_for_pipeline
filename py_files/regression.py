import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler
from concurrent.futures import ProcessPoolExecutor
import functools
import time
import matplotlib.pyplot as plt
import matplotlib
# real-world dataset
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from statsmodels.tsa.ar_model import AutoReg



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
   """
   Hessian of the non-convex function.
   Computes a 2D Hessian matrix.
   """
   residual = y - np.dot(X, beta)
   residual_sq = residual ** 2
   factor = (1 - residual_sq) / (1 + residual_sq)**2
   H = np.dot(X.T, X * factor[:, np.newaxis])
   return H




# ==================== Optimization Utilities ====================




def sgd_with_bound(f, grad_f, start_point, end_point, X, y,
                  learning_rate=0.01, iterations=1000, tol=1e-6):
   """
   Simple SGD optimizer with a bound check.
   """
   x = np.array(start_point, dtype=float)
   x_end = np.array(end_point, dtype=float)
   for i in range(iterations):
       gradient = grad_f(x, X, y)
       x_new = x - learning_rate * gradient
       if np.all(x_new >= x_end):
           break
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
                                 learning_rate=0.01, iterations=1000, tol=1e-6):
   """
   Processes a single pair of start and end points with SGD and,
   if necessary, applies a global search when the Hessian is non-convex.
   """
   points = []
   results = []
   start, end = pt
   point, result = sgd_with_bound(f, grad_f, start, end, X, y, learning_rate, iterations, tol)
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


executor = ProcessPoolExecutor()

def sgd_opt_global_search(start_intervals, end_intervals, n, f, grad_f, hessian_f, global_search,
                          X, y, learning_rate=0.01, max_iterations=1000, tol=1e-6):
    iters = int(max_iterations / n)
    points_lst = generate_uniform_start_end_pairs(start_intervals, end_intervals, n)
    futures = [executor.submit(
        process_point_with_no_zoom_in, f, grad_f, hessian_f,
        global_search, pt, X, y, learning_rate, iters, tol) for pt in points_lst]
    
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




# ==================== Optimization Methods Comparison ====================





def standard_sgd(f, grad_f, X, y, max_iterations=1000, learning_rate=0.01, tol=1e-6):
   """
   Standard SGD implementation without bounds.
   """
   n_features = X.shape[1]
   # Random initialization
   x = np.random.randn(n_features + 6)
  
   for i in range(max_iterations):
       gradient = grad_f(x, X, y)
       x_new = x - learning_rate * gradient
       if np.linalg.norm(x_new - x) < tol:
           break
       x = x_new
  
   return x, f(x, X, y)

'''
def standard_sgd(f, grad_f, X, y, max_iterations=1000, learning_rate=0.01, tol=1e-6, batch_size=None):
    """
    标准SGD带随机小批量采样（如果batch_size=None就是全数据Batch GD）
    
    参数：
    - f: 目标函数
    - grad_f: 目标函数的梯度（注意这里应该是可以处理小批量数据的）
    - X, y: 输入数据和标签
    - max_iterations: 最大迭代步数
    - learning_rate: 学习率
    - tol: 收敛判定阈值
    - batch_size: 小批量大小（None表示全量批量）
    """
    n_samples = X.shape[0]
    n_features = X.shape[1]
    
    # 初始化参数（随机）
    x = np.random.randn(n_features + 6)
    
    for i in range(max_iterations):
        if batch_size is None or batch_size >= n_samples:
            # 用全量数据
            X_batch = X
            y_batch = y
        else:
            # 随机采样一小批
            idx = np.random.choice(n_samples, batch_size, replace=False)
            X_batch = X.iloc[idx] if isinstance(X, pd.DataFrame) else X[idx]
            y_batch = y.iloc[idx] if isinstance(y, pd.Series) else y[idx]
        
        # 计算梯度（基于小批量）
        gradient = grad_f(x, X_batch, y_batch)
        
        # 参数更新
        x_new = x - learning_rate * gradient
        
        # 收敛判断
        if np.linalg.norm(x_new - x) < tol:
            break
        
        x = x_new
    
    # 返回最终参数 和 损失值（在全数据上评估）
    return x, f(x, X, y)
'''



def de_only_search(f, X, y, n_features, max_iterations=100):
   """
   Differential evolution only, without SGD refinement.
   """
   # Define bounds
   lower_bounds = [-10] * 6 + [-100] * n_features
   upper_bounds = [10] * 6 + [100] * n_features
   bounds = rearrange_bounds((lower_bounds, upper_bounds))
  
   def f_wrapper(beta):
       return f(beta, X, y)
  
   result = differential_evolution(
       f_wrapper,
       bounds,
       maxiter=max_iterations,
       #popsize=20,
       #strategy='best1bin'
   )
  
   return result.x, result.fun




# ==================== Main Optimization Function ====================




def optimize_full_pipeline(X_raw, y, n_features=None, max_iterations=1000):
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
       tol=1e-6
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
   matplotlib.use('Agg')  # Use non-interactive backend
  
   # Filter successful methods
   successful_methods = {k: v for k, v in results.items() if v["success"]}
  
   if not successful_methods:
       print("No successful optimization methods to plot.")
       return
  
   # Create figure with multiple subplots
   fig = plt.figure(figsize=(16, 12))
  
   # 1. Performance Comparison - Time vs Loss
   ax1 = fig.add_subplot(2, 2, 1)
   methods = list(successful_methods.keys())
   times = [successful_methods[m]["time"] for m in methods]
   losses = [successful_methods[m]["loss"] for m in methods]
  
   # Create scatter plot
   sc = ax1.scatter(times, losses, s=100, c=range(len(methods)), cmap='viridis', alpha=0.7)
  
   # Add method labels
   for i, method in enumerate(methods):
       ax1.annotate(method, (times[i], losses[i]), fontsize=9,
                   xytext=(5, 5), textcoords='offset points')
  
   ax1.set_xlabel('Execution Time (seconds)')
   ax1.set_ylabel('Loss Value')
   ax1.set_title('Optimization Performance: Time vs. Loss')
   ax1.grid(True, linestyle='--', alpha=0.7)
  
   # 2. Bar chart comparing times
   ax2 = fig.add_subplot(2, 2, 2)
   colors = plt.cm.viridis(np.linspace(0, 1, len(methods)))
   bars = ax2.bar(methods, times, color=colors, alpha=0.7)
   ax2.set_xlabel('Method')
   ax2.set_ylabel('Time (seconds)')
   ax2.set_title('Execution Time Comparison')
   ax2.set_xticklabels(methods, rotation=45, ha='right')
   ax2.grid(True, axis='y', linestyle='--', alpha=0.7)
  
   # Add time values on top of bars
   for bar in bars:
       height = bar.get_height()
       ax2.annotate(f'{height:.2f}',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3),  # 3 points vertical offset
                   textcoords="offset points",
                   ha='center', va='bottom')
  
   # 3. Bar chart comparing losses
   ax3 = fig.add_subplot(2, 2, 3)
   bars = ax3.bar(methods, losses, color=colors, alpha=0.7)
   ax3.set_xlabel('Method')
   ax3.set_ylabel('Loss Value')
   ax3.set_title('Loss Value Comparison')
   ax3.set_xticklabels(methods, rotation=45, ha='right')
   ax3.grid(True, axis='y', linestyle='--', alpha=0.7)
  
   # Add loss values on top of bars
   for bar in bars:
       height = bar.get_height()
       ax3.annotate(f'{height:.4f}',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3),  # 3 points vertical offset
                   textcoords="offset points",
                   ha='center', va='bottom')
  
   # 4. Parameter comparison for best method
   ax4 = fig.add_subplot(2, 2, 4)
   best_method = min(successful_methods.keys(), key=lambda m: successful_methods[m]["loss"])
   best_params = successful_methods[best_method]["params"]
  
   # Get interpretation of best params
   best_interpretation = interpret_parameters(best_params)
  
   # Plot regression coefficients
   reg_coefs = list(best_interpretation["regression"].values())
   feature_names = list(best_interpretation["regression"].keys())
  
   y_pos = np.arange(len(feature_names))
   ax4.barh(y_pos, reg_coefs, color='skyblue', alpha=0.7)
  
   # If true coefficients are provided, overlay them
   if true_beta is not None:
       ax4.barh(y_pos, true_beta, color='red', alpha=0.3, label='True Coefficients')
       ax4.legend()
  
   ax4.set_yticks(y_pos)
   ax4.set_yticklabels(feature_names)
   ax4.set_xlabel('Coefficient Value')
   ax4.set_title(f'Regression Coefficients ({best_method})')
   ax4.grid(True, axis='x', linestyle='--', alpha=0.7)
  
   # Add a title for the entire figure
   fig.suptitle('Optimization Methods Comparison', fontsize=16)
   plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout for the suptitle
  
   # Save the figure
   fig.savefig('optimization_comparison.png', dpi=300, bbox_inches='tight')
   plt.close(fig)
  
   # Create another figure for imputation and scaling weights
   fig2 = plt.figure(figsize=(14, 6))
  
   # 1. Imputation weights
   ax5 = fig2.add_subplot(1, 2, 1)
   imp_methods = list(best_interpretation["imputation"].keys())
   imp_weights = list(best_interpretation["imputation"].values())
  
   ax5.pie(imp_weights, labels=imp_methods, autopct='%1.1f%%', startangle=90,
          colors=plt.cm.Blues(np.linspace(0.3, 0.7, len(imp_methods))))
   ax5.set_title(f'Imputation Method Weights ({best_method})')
  
   # 2. Scaling weights
   ax6 = fig2.add_subplot(1, 2, 2)
   scale_methods = list(best_interpretation["scaling"].keys())
   scale_weights = list(best_interpretation["scaling"].values())
  
   ax6.pie(scale_weights, labels=scale_methods, autopct='%1.1f%%', startangle=90,
          colors=plt.cm.Greens(np.linspace(0.3, 0.7, len(scale_methods))))
   ax6.set_title(f'Scaling Method Weights ({best_method})')
  
   fig2.suptitle('Soft Parameter Weights for Best Method', fontsize=16)
   plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout for the suptitle
  
   # Save the second figure
   fig2.savefig('weights_comparison.png', dpi=300, bbox_inches='tight')
   plt.close(fig2)
  
   print("Plots saved as 'optimization_comparison.png' and 'weights_comparison.png'")
  
   return True




# ==================== Complete Example Usage ====================




def example_usage():
   """
   Example demonstrating how to use the full pipeline optimization.
   """
   import time
  
   # Generate synthetic data with missing values
   np.random.seed(42)


   X_df = X_train

   y = y_train

   n_features = 3 


  
   sgd_max_iterations = 1000

   
   methods = {
       "Our Approach": lambda: optimize_full_pipeline(
           X_df, y, max_iterations=sgd_max_iterations
       ),
       "Standard SGD": lambda: standard_sgd(
           pipeline_with_soft_parameters, pipeline_gradient,
           X_df, y, max_iterations=sgd_max_iterations
       ),
       "SGD with Bound": lambda: sgd_with_bound(
           pipeline_with_soft_parameters, pipeline_gradient,
           np.random.randn(n_features + 6), np.ones(n_features + 6) * np.inf,
           X_df, y, iterations=sgd_max_iterations
       ),
       "Differential Evolution": lambda: de_only_search(
           pipeline_with_soft_parameters, X_df, y, n_features, max_iterations=100
       )
   }
  
   results = {}
   for method_name, method_func in methods.items():
       print(f"\nRunning {method_name}...")
       start_time = time.time()
       try:
           optimal_params, min_value = method_func()
           end_time = time.time()
           results[method_name] = {
               "success": True,
               "time": end_time - start_time,
               "loss": min_value,
               "params": optimal_params
           }
           print(f"  - Success: Loss = {min_value:.6f}, Time = {end_time - start_time:.2f} seconds")
       except Exception as e:
           end_time = time.time()
           results[method_name] = {
               "success": False,
               "time": end_time - start_time,
               "error": str(e)
           }
           print(f"  - Failed: {str(e)}")
           print(f"  - Time elapsed: {end_time - start_time:.2f} seconds")
  
   # Print comparison table
   print("\n" + "="*50)
   print("OPTIMIZATION METHODS COMPARISON")
   print("="*50)
   print(f"{'Method':<25} {'Loss':<15} {'Time (s)':<15} {'Status':<10}")
   print("-"*65)
  
   for method, result in results.items():
       if result["success"]:
           print(f"{method:<25} {result['loss']:<15.6f} {result['time']:<15.2f} {'Success':<10}")
       else:
           print(f"{method:<25} {'N/A':<15} {result['time']:<15.2f} {'Failed':<10}")
  
   # Print detailed parameters for the best method
   best_method = min(
       [m for m, r in results.items() if r["success"]],
       key=lambda m: results[m]["loss"],
       default=None
   )
  
   if best_method:
       print("\n" + "="*50)
       print(f"BEST METHOD: {best_method}")
       print("="*50)
       best_params = results[best_method]["params"]
       interpretation = interpret_parameters(best_params, X_df.columns)
      
       print("\nOptimized Parameters:")
       print("Imputation weights:")
       for method, weight in interpretation['imputation'].items():
           print(f"  - {method}: {weight:.4f}")
      
       print("\nScaling weights:")
       for method, weight in interpretation['scaling'].items():
           print(f"  - {method}: {weight:.4f}")
      
       print("\nRegression coefficients:")
       for feature, coef in interpretation['regression'].items():
           print(f"  - {feature}: {coef:.4f}")
      
  
   # Generate visualization
   plot_optimization_results(results, None)


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



if __name__ == "__main__":
    #example_usage()
    test_usage()
# python py_files/regression.py
