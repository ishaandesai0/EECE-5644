import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from sklearn.metrics import roc_curve
from scipy import linalg

np.random.seed(42)

# Data Generation Function

def generate_data(N = 10000):
    """
    Generate 10,000 samples per given multivariate Gaussian probability density function
    
    Returns:
    Generated sample Data
    
    """
    
    p0 = 0.65 # Class priors
    p1 = 0.35
    
    m0 = np.array([-0.5, -0.5, -0.5]) # Class 0 Parameters
    c0 = np.array([[1, -0.5, 0.3], [-0.5, 1, -0.5], [0.3, -0.5, 1]])
    
    m1 = np.array([1, 1, 1]) # Class 1 Parameters
    c1 = np.array([[1, 0.3, -0.2], [0.3, 1, 0.3], [-0.2, 0.3, 1]])
    
    # Labels
    labels = (np.random.rand(N) >= p0).astype(int)
    N0 = np.sum(labels == 0)
    N1 = np.sum(labels == 1)
    
    print(f'Generated {N} samples')
    print(f'\nClass 0: {N0} samples')
    print(f'\nClass 1: {N1} samples')
    
    # Generate samples from each class
    X0 = np.random.multivariate_normal(m0, c0, N0)
    X1 = np.random.multivariate_normal(m1, c1, N1)
    
    X = np.zeros((N, 3))
    X[labels == 0] = X0
    X[labels == 1] = X1
    
    return X, labels, m0, c0, m1, c1, p0, p1

def visualize_data(X, labels):
    """
    Visualize 3D data
    """
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    X0 = X[labels == 0]
    X1 = X[labels == 1]
    
    ax.scatter(X0[:, 0], X0[:, 1], X0[:, 2], c='blue', marker='o', alpha=0.3, label ='Class 0', s=20)
    ax.scatter(X1[:, 0], X1[:, 1], X1[:, 2], c='red', marker='^', alpha=0.3, label ='Class 1', s=20)
    
    ax.set_xlabel('X1', fontsize=12)
    ax.set_ylabel('X2', fontsize=12)
    ax.set_zlabel('X3', fontsize=12)
    ax.set_title('3D Scatter Plot of Generated Data', fontsize=14)
    ax.legend(fontsize=12)
    plt.tight_layout()
    plt.show()
    
# PART A: ERM CLASSIFICATION WITH TRUE PDF

def compute_likelihood(X, m0, c0, m1, c1):
    """ 
    Compute likelihood ratio p(x|L=1) / p(x|L=0) for each sample
    Args:
      X: (N, d) array of samples
      m0, c0: Mean and Covariance of Class 0
      m1, c1: Mean and Covariance of Class 1
      
    Returns:
      likelihoodRatio: (N,) array of likelihood ratios
    """
    
    pdf0 = multivariate_normal(mean = m0, cov = c0)
    pdf1 = multivariate_normal(mean = m1, cov = c1)
    
    likelihood0 = pdf0.pdf(X)
    likelihood1 = pdf1.pdf(X)
    
    # To avoid division by zero
    likelihoodRatio = likelihood1 / (likelihood0 + 1e-10)
    return likelihoodRatio

def classify_with_threshold(likelihoodRatio, gamma):
    """
    Classify samples based on likelihood ratio and threshold
    
    Args:
       likelihoodRatio: (N,) array
       gamma: threshold
    
    Returns:
        decisions: (N,) array of deicisions (0 1)
    """
    return (likelihoodRatio > gamma).astype(int)
    
def compute_performance_metrics(decisions, labels):
    """
    Compute TFR, FPR, error probability
    
    Args:
       decisions: (N,) array of decisions
       labels: (N,) array of true labels
       
    Returns:
        TPR, FPR, P_error
    """
    
    # True Positives, False Positives, True Negatives, False Negatives
    TP = np.sum((decisions == 1) & (labels == 1))
    FP = np.sum((decisions == 1) & (labels == 0))
    TN = np.sum((decisions == 0) & (labels == 0))
    FN = np.sum((decisions == 0) & (labels == 1))
    
    # TPR = P(D=1 | L=1)
    TPR = TP / (TP + FN) if (TP + FN) > 0 else 0
    
    # FPR = P(D=1 | L=0)
    FPR = FP / (FP + TN) if (FP + TN) > 0 else 0
    
    # P(error)
    P_error = (FP + FN) / len(labels)
    
    return TPR, FPR, P_error

def make_roc_curve(likelihoodRatio, labels, p0, p1):
    """
    Generate ROC curve by varying threshold gamma
    
    Returns:
        gammas: array of threshold values
        TPRs: array of true positive rates
        FPRs: array of false positive rates
        P_errors: array of error probabilities
    """
    
    # Generate range of gamma values from 0 to max likelihood ratio
    gammas = np.logspace(-3, 3, 1000)
    gammas = np.concatenate(([0], gammas, [np.inf]))
    
    TPRs = []
    FPRs = []
    P_errors = []
    
    for gamma in gammas:
        decisions = classify_with_threshold(likelihoodRatio, gamma)
        TPR, FPR, P_error = compute_performance_metrics(decisions, labels)
        
        TPRs.append(TPR)
        FPRs.append(FPR)
        P_errors.append(P_error)
        
    return np.array(gammas), np.array(TPRs), np.array(FPRs), np.array(P_errors)

def plot_roc_curve(FPRs, TPRs, optimal_idx = 0, title = 'ROC Curve'):
    """
    Plot ROC curve with optimal point marked
    """
    
    plt.figure(figsize=(10,8))
    plt.plot(FPRs, TPRs, 'b-', linewidth=2, label='ROC Curve')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
    
    if optimal_idx is not None:
        plt.plot(FPRs[optimal_idx], TPRs[optimal_idx], 'ro',
                 markersize=12, label='Minimum P(error)', markeredgewidth=2)
    
    plt.xlabel('False Positive Rate (FPR)', fontsize=14)
    plt.ylabel('True Positive Rate (TPR)', fontsize=14)
    plt.title(title, fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.tight_layout()
    plt.show()
    
def Part_A(X, labels, m0, c0, m1, c1, p0, p1):
    """
    Perform Part A: ERM Classification with true data PDF
    """
    print("\n" + "="*80)
    print("PART A: ERM CLASSIFICATION WITH TRUE PDF")
    print("="*80)
    
    # Theoretical optimal threshold
    gamma_theoretical = (p0 / p1)
    print(f"\nTheoretical optimal threshold (0-1 loss):")
    print(f"   y* = P(L=0)/P(L=1) = {p0}/{p1} = {gamma_theoretical:.4f}")
    
    # Compute likelihood ratios
    likelihoodRatio = compute_likelihood(X, m0, c0, m1, c1)
    print(f"   LR range: [{likelihoodRatio.min():.2e}, {likelihoodRatio.max():.2e}]")
    
    # Generate ROC Curve
    gammas, TPRs, FPRs, P_errors = make_roc_curve(likelihoodRatio, labels, p0, p1)
    
    # Find optimal gamma
    optimal_idx = np.argmin(P_errors)
    gamma_empirical = gammas[optimal_idx]
    min_P_error = P_errors[optimal_idx]
    optimal_TPR = TPRs[optimal_idx]
    optimal_FPR = FPRs[optimal_idx]
    
    print(f"\nEmpirical optimal y: {gamma_empirical:.4f}")
    print(f"\nTheoretical optimal y: {gamma_theoretical:.4f}")
    print(f"\nDifference: {gamma_empirical - gamma_theoretical:.4f}")
    print(f"\nMinimum P(error): {min_P_error:.4f}")
    print(f"\nTPR = {optimal_TPR:.4f}, FPR = {optimal_FPR:.4f}")
    
    # Plot ROC Curve
    plot_roc_curve(FPRs, TPRs, optimal_idx, title='ROC Curve - Part A (True PDF)')
    
    return gammas, TPRs, FPRs, P_errors, likelihoodRatio

# PART B: Naive Bayes Classifier

def part_B(X, labels, m0, m1, p0, p1):
    """
    Naive Bayes Classifier
    """
    
    print("\n" + "="*80)
    print("PART B: NAIVE BAYES CLASSIFIER")
    print("="*80)
    
    print("\n Assumption: Clss-conditional covariances are identity matrices")
    
    # Using identity covariance matrices (Incorrect Assumption)
    c0_naive = np.eye(3)
    c1_naive = np.eye(3)
    
    # Compute likelihood ratios with naive assumption
    likelihoodRatio_naive = compute_likelihood(X, m0, c0_naive, m1, c1_naive)
    
    # Generate ROC Curve
    gammas, TPRs, FPRs, P_errors = make_roc_curve(likelihoodRatio_naive, labels, p0, p1)
    
    # Optimal Gamma
    optimal_idx = np.argmin(P_errors)
    gamma_empirical = gammas[optimal_idx]
    min_P_error = P_errors[optimal_idx]
    optimal_TPR = TPRs[optimal_idx]
    optimal_FPR = FPRs[optimal_idx]
    
    print(f"\n Results with Naive Bayes:")
    print(f" Empirical optimal y: {gamma_empirical:.4f}")
    print(f" Minimum P(error): {min_P_error:.4f}")
    print(f" TPR = {optimal_TPR:.4f}, FPR = {optimal_FPR:.4f}")
    
    plot_roc_curve(FPRs, TPRs, optimal_idx, title='ROC Curve - Part B (Naive Bayes)')
    
    return gammas, TPRs, FPRs, P_errors

# Part C: Fisher LDA Classifier

def estimate_parameters(X, labels):
    """
    Estimate class means and shared covariance matrix
    """
    
    X0 = X[labels == 0]
    X1 = X[labels == 1]
    
    m0_est = np.mean(X0, axis=0)
    m1_est = np.mean(X1, axis=0)
    
    c0_est = np.cov(X0.T)
    c1_est = np.cov(X1.T)
    
    return m0_est, c0_est, m1_est, c1_est

def compute_fisher_lda(m0, c0, m1, c1):
    """
    Compute Fisher LDA projection vector
    Returns:
        w_lda: projection vector
    """
    
    # Between-class scatter
    mean_diff = (m1 - m0).reshape(-1, 1)
    S_B = mean_diff @ mean_diff.T
    
    # Within class scatter matrix
    S_W = c0 + c1
    
    # Solve eigenvalue problem
    eigenvalues, eigenvectors = linalg.eig(S_B, S_W)
    
    # Get eigenvector corresponding to largest eigenvalue
    idx = np.argmax(np.real(eigenvalues))
    w_lda = np.real(eigenvectors[:, idx])
    
    # Normalize
    w_lda = w_lda / np.linalg.norm(w_lda)
    
    return w_lda

def Part_C(X, labels, p0, p1):
    """
    Perform Part C: Fisher LDA Classifier
    """
    
    print("\n" + "="*80)
    print("PART C: FISHER LDA CLASSIFIER")
    print("="*80)
    
    # Estimate parameters
    m0_est, c0_est, m1_est, c1_est = estimate_parameters(X, labels)
    
    print(f" Estimated m0: {m0_est}")
    print(f" Estimated m1: {m1_est}")
    
    # Compute Fisher LDA projection vector
    w_lda = compute_fisher_lda(m0_est, c0_est, m1_est, c1_est)
    print(f" Fisher LDA projection vector w: {w_lda}")
    
    # Project data onto LDA direction
    y = X @ w_lda
    
    # Generate ROC curve
    taus = np.linspace(y.min() - 1, y.max() + 1, 1000)
    TPRs = []
    FPRs = []
    P_errors = []
    
    for tau in taus:
        decisions = (y > tau).astype(int)
        TPR, FPR, P_error = compute_performance_metrics(decisions, labels)
        TPRs.append(TPR)
        FPRs.append(FPR)
        P_errors.append(P_error)
        
    TPRs = np.array(TPRs)
    FPRs = np.array(FPRs)
    P_errors = np.array(P_errors)
    
    # Optimal Threshold
    optimal_idx = np.argmin(P_errors)
    tau_optimal = taus[optimal_idx]
    min_P_error = P_errors[optimal_idx]
    optimal_TPR = TPRs[optimal_idx]
    optimal_FPR = FPRs[optimal_idx]
    
    print(f"\n Optimal Threshold Results")
    print(f" Optimal tau: {tau_optimal:.4f}")
    print(f" Minimum P(error): {min_P_error:.4f}")
    print(f" TPR = {optimal_TPR:.4f}, FPR = {optimal_FPR:.4f}")
    
    # Plot Curve
    plot_roc_curve(FPRs, TPRs, optimal_idx, title='ROC Curve - Part C (Fisher LDA)')
    
    return w_lda, taus, TPRs, FPRs, P_errors

if __name__ == "__main__":
    print("=" * 80)
    
    print("\nGenerating Data...")
    X, labels, m0, c0, m1, c1, p0, p1 = generate_data(N=10000)
    
    # Visualize Data
    visualize_data(X, labels)
    
    # Part A
    results_A = Part_A(X, labels, m0, c0, m1, c1, p0, p1)
    
    # Part B
    results_B = part_B(X, labels, m0, m1, p0, p1)
    
    # Part C
    results_C = Part_C(X, labels, p0, p1)
    
    print("\n" + "="*80)