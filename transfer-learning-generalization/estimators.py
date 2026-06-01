import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

np.random.seed(42)

"""
CS7140 Problem Set 1 

This file intentionally contains only the parts that require code:
- Problem 1 (all parts): OLS vs HPS, SPS tuning, plots
- Problem 2(c): simulation validation plots for sub-Gaussian scaling/sum facts

The proof-based parts (Problem 2(a)(b) and all of Problem 3) are in `hw1_writeup.tex`.
"""

def generate_transfer_data(p, n1, n2, delta, sigma=0.1):
    #Generate synthetic data for two transfer learning tasks.

    # Generate beta^(1) randomly
    beta1 = np.random.randn(p)
    
    # Generate beta^(2) with specified distance delta from beta^(1)
    direction = np.random.randn(p)
    direction = direction / np.linalg.norm(direction)  # unit vector
    beta2 = beta1 + delta * direction  # ||beta^(1) - beta^(2)|| = delta
    
    # Generate isotropic Gaussian covariates
    X1 = np.random.randn(n1, p)
    X2 = np.random.randn(n2, p)
    
    # Generate labels with Gaussian noise (mean=0, std=0.1)
    y1 = X1 @ beta1 + np.random.normal(0, sigma, n1)
    y2 = X2 @ beta2 + np.random.normal(0, sigma, n2)
    
    return X1, y1, X2, y2, beta1, beta2

def ols_estimator(X, y):
    """
    Ordinary Least Squares (single-task estimator).
    Uses ONLY target task data.
    
    Formula: beta_hat_OLS = (X^T X)^(-1) X^T y
    """
    return np.linalg.lstsq(X, y, rcond=None)[0]


def hps_estimator(X1, y1, X2, y2):
    """
    Hard Parameter Sharing (hard transfer estimator).
    Pools all data, assumes beta^(1) = beta^(2) = beta.
    
    Minimizes: (1/(n1+n2)) * [sum_i (x_i^(1)^T beta - y_i^(1))^2 + sum_j (x_j^(2)^T beta - y_j^(2))^2]
    """
    X_combined = np.vstack([X1, X2])
    y_combined = np.concatenate([y1, y2])
    return np.linalg.lstsq(X_combined, y_combined, rcond=None)[0]


def compute_test_loss(X_test, y_test, beta_hat):
    #Compute Mean Squared Error on test data.
    predictions = X_test @ beta_hat
    return np.mean((predictions - y_test) ** 2)


def sps_estimator(X1, y1, X2, y2, lam):
    """
    Soft Parameter Sharing (soft transfer estimator).
    
    Let beta be the Task 2 parameter and z = beta^(1) - beta^(2) be the difference.
    Task 1 uses (beta + z), Task 2 uses beta.
    
    Minimizes:
    (1/(n1+n2)) * [sum_i (x_i^(1)^T(beta+z) - y_i^(1))^2 + sum_j (x_j^(2)^T beta - y_j^(2))^2] + lam*||z||^2
    
    The lam*||z||^2 term penalizes large differences between beta^(1) and beta^(2).
    """
    n1, n2 = len(y1), len(y2)
    p = X1.shape[1]
    
    def loss(params):
        beta = params[:p]
        z = params[p:]
        
        # Task 1 loss (uses beta + z)
        pred1 = X1 @ (beta + z)
        loss1 = np.sum((pred1 - y1) ** 2)
        
        # Task 2 loss (uses beta)
        pred2 = X2 @ beta
        loss2 = np.sum((pred2 - y2) ** 2)
        
        # Regularization on z
        reg = lam * np.sum(z ** 2)
        
        return (loss1 + loss2) / (n1 + n2) + reg
    
    # Initialize with HPS solution
    beta_init = hps_estimator(X1, y1, X2, y2)
    z_init = np.zeros(p)
    params_init = np.concatenate([beta_init, z_init])
    
    # Optimize
    result = minimize(loss, params_init, method='L-BFGS-B')
    
    beta_opt = result.x[:p]
    z_opt = result.x[p:]
    
    return beta_opt, z_opt


# Find optimal lambda for each delta using validation
def find_best_lambda(X1, y1, X2_train, y2_train, X2_val, y2_val, lambdas):
    # Find best lambda using validation set.
    best_loss = float('inf')
    best_lam = lambdas[0]
    
    for lam in lambdas:
        beta_sps, _ = sps_estimator(X1, y1, X2_train, y2_train, lam)
        val_loss = compute_test_loss(X2_val, y2_val, beta_sps)
        if val_loss < best_loss:
            best_loss = val_loss
            best_lam = lam
    
    return best_lam

def run_problem1(
    p=100,
    n1=200,
    n2=100,
    sigma=0.1,
    deltas=None,
    n_trials=10,
    test_n=500,
):
    """
    Problem 1 code: generate synthetic tasks, compare OLS vs HPS, implement SPS, tune lambda, plot.
    Saves:
    - hw1_p1_part2_ols_vs_hps.png
    - hw1_p1_part4_hps_vs_sps.png
    """
    if deltas is None:
        deltas = np.linspace(0.01, 1.00, 20)

    print("PROBLEM 1: Transfer Learning Estimators (CODE)")
    print(f"Using p={p}, n1={n1}, n2={n2}, sigma={sigma}, trials={n_trials}")

    # Part 2: OLS vs HPS varying delta
    ols_losses = []
    hps_losses = []

    print("\nProblem 1.2: Running OLS vs HPS experiment...")
    for delta in deltas:
        ols_trial = []
        hps_trial = []

        for _ in range(n_trials):
            X1, y1, X2_train, y2_train, _, beta2 = generate_transfer_data(p, n1, n2, delta, sigma)

            X2_test = np.random.randn(test_n, p)
            y2_test = X2_test @ beta2 + np.random.normal(0, sigma, test_n)

            beta_ols = ols_estimator(X2_train, y2_train)
            beta_hps = hps_estimator(X1, y1, X2_train, y2_train)

            ols_trial.append(compute_test_loss(X2_test, y2_test, beta_ols))
            hps_trial.append(compute_test_loss(X2_test, y2_test, beta_hps))

        ols_losses.append(float(np.mean(ols_trial)))
        hps_losses.append(float(np.mean(hps_trial)))

    plt.figure(figsize=(10, 6))
    plt.plot(deltas, ols_losses, "b-o", label="OLS (Task 2 only)", linewidth=2, markersize=6)
    plt.plot(deltas, hps_losses, "r-s", label="HPS (Both tasks)", linewidth=2, markersize=6)
    plt.xlabel("delta = ||beta(1) - beta(2)|| (model shift)", fontsize=12)
    plt.ylabel("Test MSE on Task 2", fontsize=12)
    plt.title("Problem 1.2: OLS vs HPS", fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("hw1_p1_part2_ols_vs_hps.png", dpi=150)
    plt.show()

    crossover_idx = int(np.argmin(np.abs(np.array(ols_losses) - np.array(hps_losses))))
    crossover_delta = float(deltas[crossover_idx])
    print(f"\nProblem 1.2: crossover delta ~= {crossover_delta:.2f}")

    # Part 3: SPS lambda sweep at a fixed delta (report best test loss over lambda)
    delta_fixed = 0.5
    lambdas = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    X1, y1, X2_train, y2_train, _, beta2 = generate_transfer_data(p, n1, n2, delta_fixed, sigma)
    X2_test = np.random.randn(1000, p)
    y2_test = X2_test @ beta2 + np.random.normal(0, sigma, 1000)

    print(f"\nProblem 1.3: SPS lambda sweep at delta={delta_fixed}")
    sps_results = []
    for lam in lambdas:
        beta_sps, z_sps = sps_estimator(X1, y1, X2_train, y2_train, lam)
        loss = compute_test_loss(X2_test, y2_test, beta_sps)
        sps_results.append((lam, float(loss), float(np.linalg.norm(z_sps))))

    best_lam, best_loss, _ = min(sps_results, key=lambda t: t[1])
    print(f"Problem 1.3: best SPS test loss = {best_loss:.6f} at lambda = {best_lam}")

    # Part 4: HPS vs SPS varying delta (SPS tunes lambda on a validation split)
    hps_losses_p4 = []
    sps_losses_p4 = []
    lambdas_search = [0.001, 0.01, 0.1, 1.0, 10.0]

    print("\nProblem 1.4: Running HPS vs SPS experiment (SPS tunes lambda on validation)...")
    for delta in deltas:
        hps_trial = []
        sps_trial = []

        for _ in range(n_trials):
            X1, y1, X2_full, y2_full, _, beta2 = generate_transfer_data(p, n1, n2 + 50, delta, sigma)
            X2_tr, X2_val = X2_full[:n2], X2_full[n2:]
            y2_tr, y2_val = y2_full[:n2], y2_full[n2:]

            X2_test = np.random.randn(test_n, p)
            y2_test = X2_test @ beta2 + np.random.normal(0, sigma, test_n)

            beta_hps = hps_estimator(X1, y1, X2_tr, y2_tr)
            hps_trial.append(compute_test_loss(X2_test, y2_test, beta_hps))

            tuned_lam = find_best_lambda(X1, y1, X2_tr, y2_tr, X2_val, y2_val, lambdas_search)
            beta_sps, _ = sps_estimator(X1, y1, X2_tr, y2_tr, tuned_lam)
            sps_trial.append(compute_test_loss(X2_test, y2_test, beta_sps))

        hps_losses_p4.append(float(np.mean(hps_trial)))
        sps_losses_p4.append(float(np.mean(sps_trial)))

    plt.figure(figsize=(10, 6))
    plt.plot(deltas, hps_losses_p4, "r-s", label="HPS", linewidth=2, markersize=6)
    plt.plot(deltas, sps_losses_p4, "g-^", label="SPS (tuned lambda)", linewidth=2, markersize=6)
    plt.xlabel("delta = ||beta(1) - beta(2)|| (model shift)", fontsize=12)
    plt.ylabel("Test MSE on Task 2", fontsize=12)
    plt.title("Problem 1.4: HPS vs SPS", fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("hw1_p1_part4_hps_vs_sps.png", dpi=150)
    plt.show()


def run_problem2_part3_simulation(n_samples=10000, sigma_x=1.0, sigma_y=1.5, c=2.0):
    """
    Problem 2(c) code: simulation validation for scaling and sum facts.
    Saves:
    - hw1_p2_subgaussian.png
    """
    print("\nPROBLEM 2(c): Sub-Gaussian simulation validation (CODE)")

    X = np.random.normal(0, sigma_x, n_samples)
    Y = np.random.normal(0, sigma_y, n_samples)

    cX = c * X
    X_plus_Y = X + Y

    plt.figure(figsize=(14, 5))

    # Scaling
    plt.subplot(1, 2, 1)
    plt.hist(cX, bins=50, density=True, alpha=0.7, color="blue", label=f"cX empirical (c={c})")
    x_range = np.linspace(-8, 8, 200)
    var_cX = (c**2) * (sigma_x**2)
    pdf_cX = (1 / np.sqrt(2 * np.pi * var_cX)) * np.exp(-(x_range**2) / (2 * var_cX))
    plt.plot(x_range, pdf_cX, "r-", linewidth=2, label=f"N(0, c^2*sigma_x^2) var={var_cX}")
    plt.title("Scaling: cX vs matched Gaussian", fontsize=12)
    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Sum
    plt.subplot(1, 2, 2)
    plt.hist(X_plus_Y, bins=50, density=True, alpha=0.7, color="green", label="X+Y empirical")
    x_range2 = np.linspace(-10, 10, 200)
    var_sum = (sigma_x**2) + (sigma_y**2)
    pdf_sum = (1 / np.sqrt(2 * np.pi * var_sum)) * np.exp(-(x_range2**2) / (2 * var_sum))
    plt.plot(x_range2, pdf_sum, "r-", linewidth=2, label=f"N(0, sigma_x^2+sigma_y^2) var={var_sum:.2f}")
    plt.title("Sum: X+Y vs matched Gaussian", fontsize=12)
    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("hw1_p2_subgaussian.png", dpi=150)
    plt.show()

    print("\nVariance checks (sanity):")
    print(f"Var(cX) empirical   = {np.var(cX):.4f}, theoretical = {var_cX:.4f}")
    print(f"Var(X+Y) empirical  = {np.var(X_plus_Y):.4f}, theoretical = {var_sum:.4f}")


if __name__ == "__main__":
    run_problem1()
    run_problem2_part3_simulation()
