import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from scipy.optimize import minimize
from sklearn.metrics import confusion_matrix, roc_curve, auc

np.random.seed(42)

# Parameters
P_L0 = 0.6
P_L1 = 0.4

# Mixture weights
w01 = w02 = 0.5
w11 = w12 = 0.5

# Mean vectors
m01 = np.array([-0.9, -1.1])
m02 = np.array([0.8, 0.75])
m11 = np.array([-1.1, 0.9])
m12 = np.array([0.9, -0.75])

# Covariance matrices
C = np.array([[0.75, 0], [0, 1.25]])

# Data Generation
def generate_samples(n_samples):
    """
    Generate samples from mixture distribution
    Returns: X (n_samples * 2), labels (n_samples)
    """
    labels = np.random.choice([0, 1], size=n_samples, p=[P_L0, P_L1])
    
    X = np.zeros((n_samples, 2))
    
    for i in range(n_samples):
        if labels[i] == 0:
            # Class 0, mixture of two Gaussians
            component = np.random.choice([0, 1], p=[w01, w02])
            if component == 0:
                X[i] = np.random.multivariate_normal(m01, C)
            else:
                X[i] = np.random.multivariate_normal(m02, C)
        else:
            # Class 1, mixture of two Gaussians
            component = np.random.choice([0, 1], p=[w11, w12])
            if component == 0:
                X[i] = np.random.multivariate_normal(m11, C)
            else:
                X[i] = np.random.multivariate_normal(m12, C)
    
    return X, labels

# Generate datasets
X_train_50, y_train_50 = generate_samples(50)
X_train_500, y_train_500 = generate_samples(500)
X_train_5000, y_train_5000 = generate_samples(5000)
X_validate, y_validate = generate_samples(10000)

print(f"D_train_50: {X_train_50.shape}, Class 0: {np.sum(y_train_50==0)}, Class 1: {np.sum(y_train_50==1)}")
print(f"D_train_500: {X_train_500.shape}, Class 0: {np.sum(y_train_500==0)}, Class 1: {np.sum(y_train_500==1)}")
print(f"D_train_5000: {X_train_5000.shape}, Class 0: {np.sum(y_train_5000==0)}, Class 1: {np.sum(y_train_5000==1)}")
print(f"D_validate: {X_validate.shape}, Class 0: {np.sum(y_validate==0)}, Class 1: {np.sum(y_validate==1)}")

# Visualize validation
plt.figure(figsize=(10, 8))
plt.scatter(X_validate[y_validate==0, 0], X_validate[y_validate==0, 1],
            alpha=0.3, c='blue', label='Class 0', s=10)
plt.scatter(X_validate[y_validate==1, 0], X_validate[y_validate==1, 1],
            alpha=0.3, c='red', label='Class 1', s=10)
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Validation Dataset (10,000 samples)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.axis('equal')
plt.show()

# Part 1

# Theoretically Optimal Classifier
def class_conditional_pdf(x, label):
    """
    Compute p(x|L=label) for a single sample or array of samples
    """
    if label == 0:
        pdf1 = multivariate_normal.pdf(x, mean=m01, cov=C)
        pdf2 = multivariate_normal.pdf(x, mean=m02, cov=C)
        return w01 * pdf1 + w02 * pdf2
    else:
        pdf1 = multivariate_normal.pdf(x, mean=m11, cov=C)
        pdf2 = multivariate_normal.pdf(x, mean=m12, cov=C)
        return w11 * pdf1 + w12 * pdf2
    
def likelihood_ratio(x):
    """
    Compute likelihood ratio p(x|L=1) / p(x|L=0)
    """
    p_x_given_L1 = class_conditional_pdf(x, 1)
    p_x_given_L0 = class_conditional_pdf(x, 0)
    return p_x_given_L1 / p_x_given_L0

def optimal_classifier(x, threshold=P_L0/P_L1):
    """
    Theoretically optimal Bayes classifier
    """
    lr = likelihood_ratio(x)
    return (lr > threshold).astype(int)

# Apply optimal classifier to validation set
y_pred_optimal = optimal_classifier(X_validate)

# Calculate confusion matrix
cm = confusion_matrix(y_validate, y_pred_optimal)
print("Confusion Matrix (Optimal Classifier):")
print(cm)
print(f"True Negatives: {cm[0,0]}, False Positives: {cm[0,1]}")
print(f"False Negatives: {cm[1,0]}, True Positives: {cm[1,1]}")

# Calculate P(error)
P_error_optimal = (cm[0,1] + cm[1,0]) / len(y_validate)
print(f"\nEstimated min P(error): {P_error_optimal:.4f}")

# Generate ROC curve 
discriminant_scores = likelihood_ratio(X_validate)
fpr, tpr, thresholds = roc_curve(y_validate, discriminant_scores)
roc_auc = auc(fpr, tpr)

# Find point on ROC corresponding to optimal threshold
optimal_threshold = P_L0 / P_L1
optimal_idx = np.argmin(np.abs(thresholds - optimal_threshold))

plt.figure(figsize=(10, 8))
plt.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
plt.plot(fpr[optimal_idx], tpr[optimal_idx], 'r*', markersize=20,
         label=f'Min P(error) point\n(FPR={fpr[optimal_idx]:.4f}, TPR={tpr[optimal_idx]:.4f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Theoretically Optimal Classifier')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

print(f"\nROC AUC: {roc_auc:.4f}")
print(f"Optimal threshold (P(L=0)/P(L=1)): {optimal_threshold:.4f}")
print(f"At optimal point - FPR: {fpr[optimal_idx]:.4f}, TPR: {tpr[optimal_idx]:.4f}")

# Part 2
def z_linear(X):
    """
    Transform input for logistic-linear model
    """
    N = X.shape[0]
    return np.column_stack((np.ones(N), X))

def logistic_function(z):
    """
    Sigmoid function with numerical stability
    """
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

def negative_log_likelihood(w, X, y):
    """
    Negative log-likelihood for logistic-linear model
    """
    Z = z_linear(X)
    h = logistic_function(Z @ w)
    # Small epsilon to avoid log(0)
    epsilon = 1e-15
    h = np.clip(h, epsilon, 1 - epsilon)
    nll = -np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))
    return nll

def train_logistic_linear(X_train, y_train):
    """
    Train logistic-linear model using MLE
    """
    w_init = np.random.randn(3) * 0.01
    result = minimize(negative_log_likelihood, w_init, 
                      args=(X_train, y_train),
                      method='BFGS',
                      options={'maxiter': 1000})
    return result.x

def predict_logistic_linear(X, w):
    """
    Make predictions using trained logistic linear model
    """
    Z = z_linear(X)
    h = logistic_function(Z @ w)
    return (h >= 0.5).astype(int)

def discriminant_score_linear(X, w):
    """
    Return discriminant scores
    """
    Z = z_linear(X)
    return logistic_function(Z @ w)

# Logistic Quadratic Functions

def z_quadratic(X):
    """
    Transform input for logistic-quadratic model
    """
    N = X.shape[0]
    x1 = X[:, 0]
    x2 = X[:, 1]
    return np.column_stack([np.ones(N), x1, x2, x1**2, x1*x2, x2**2])

def negative_log_likelihood_quadratic(w, X, y):
    """
    Negative log likelihood for logistic-quadratic model
    """
    Z = z_quadratic(X)
    h = logistic_function(Z @ w)
    epsilon = 1e-15
    h = np.clip(h, epsilon, 1 - epsilon)
    nll = -np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))
    return nll

def train_logistic_quadratic(X_train, y_train):
    """
    Train logistic-quadratic model using MLE
    """
    w_init = np.random.randn(6) * 0.01
    result = minimize(negative_log_likelihood_quadratic, w_init, 
                      args=(X_train, y_train),
                      method='BFGS',
                      options={'maxiter': 1000})
    return result.x

def predict_logistic_quadratic(X, w):
    """
    Make predictions using trained logistic quadratic model
    """
    Z = z_quadratic(X)
    h = logistic_function(Z @ w)
    return (h >= 0.5).astype(int)

def discriminant_score_quadratic(X, w):
    """
    Return discriminant scores
    """
    Z = z_quadratic(X)
    return logistic_function(Z @ w)

# Train and evaluate models on different training set sizes
print("\n" + "=" * 60)
print("PART 2A: LOGISTIC-LINEAR MODELS")
print("=" * 60)

# 50 samples
print("\nTraining logistic-linear on 50 samples...")
w_linear_50 = train_logistic_linear(X_train_50, y_train_50)
y_pred_linear_50 = predict_logistic_linear(X_validate, w_linear_50)
cm_linear_50 = confusion_matrix(y_validate, y_pred_linear_50)
error_linear_50 = (cm_linear_50[0,1] + cm_linear_50[1,0]) / len(y_validate)
print(f"P(error) on validation: {error_linear_50:.4f}")
print(f"Confusion Matrix:\n{cm_linear_50}")

# 500 samples
print("\nTraining logistic-linear on 500 samples...")
w_linear_500 = train_logistic_linear(X_train_500, y_train_500)
y_pred_linear_500 = predict_logistic_linear(X_validate, w_linear_500)
cm_linear_500 = confusion_matrix(y_validate, y_pred_linear_500)
error_linear_500 = (cm_linear_500[0,1] + cm_linear_500[1,0]) / len(y_validate)
print(f"P(error) on validation: {error_linear_500:.4f}")
print(f"Confusion Matrix:\n{cm_linear_500}")

# 5000 samples
print("\nTraining logistic-linear on 5000 samples...")
w_linear_5000 = train_logistic_linear(X_train_5000, y_train_5000)
y_pred_linear_5000 = predict_logistic_linear(X_validate, w_linear_5000)
cm_linear_5000 = confusion_matrix(y_validate, y_pred_linear_5000)
error_linear_5000 = (cm_linear_5000[0,1] + cm_linear_5000[1,0]) / len(y_validate)
print(f"P(error) on validation: {error_linear_5000:.4f}")
print(f"Confusion Matrix:\n{cm_linear_5000}")

print("\n" + "=" * 60)
print("PART 2B: LOGISTIC-QUADRATIC MODELS")
print("=" * 60)

# 50 samples
print("\nTraining logistic-quadratic on 50 samples...")
w_quadratic_50 = train_logistic_quadratic(X_train_50, y_train_50)
y_pred_quadratic_50 = predict_logistic_quadratic(X_validate, w_quadratic_50)
cm_quadratic_50 = confusion_matrix(y_validate, y_pred_quadratic_50)
error_quadratic_50 = (cm_quadratic_50[0,1] + cm_quadratic_50[1,0]) / len(y_validate)
print(f"P(error) on validation: {error_quadratic_50:.4f}")
print(f"Confusion Matrix:\n{cm_quadratic_50}")

# 500 samples
print("\nTraining logistic-quadratic on 500 samples...")
w_quadratic_500 = train_logistic_quadratic(X_train_500, y_train_500)
y_pred_quadratic_500 = predict_logistic_quadratic(X_validate, w_quadratic_500)
cm_quadratic_500 = confusion_matrix(y_validate, y_pred_quadratic_500)
error_quadratic_500 = (cm_quadratic_500[0,1] + cm_quadratic_500[1,0]) / len(y_validate)
print(f"P(error) on validation: {error_quadratic_500:.4f}")
print(f"Confusion Matrix:\n{cm_quadratic_500}")

# 5000 samples
print("\nTraining logistic-quadratic on 5000 samples...")
w_quadratic_5000 = train_logistic_quadratic(X_train_5000, y_train_5000)
y_pred_quadratic_5000 = predict_logistic_quadratic(X_validate, w_quadratic_5000)
cm_quadratic_5000 = confusion_matrix(y_validate, y_pred_quadratic_5000)
error_quadratic_5000 = (cm_quadratic_5000[0,1] + cm_quadratic_5000[1,0]) / len(y_validate)
print(f"P(error) on validation: {error_quadratic_5000:.4f}")
print(f"Confusion Matrix:\n{cm_quadratic_5000}")

# Summary
print("\n" + "=" * 60)
print("SUMMARY OF RESULTS")
print("=" * 60)
print(f"\nTheoretically Optimal Classifier:")
print(f"  P(error) = {P_error_optimal:.4f} (baseline)")
print(f"\nLogistic-Linear Classifier:")
print(f"  50 samples:    P(error) = {error_linear_50:.4f}")
print(f"  500 samples:   P(error) = {error_linear_500:.4f}")
print(f"  5000 samples:  P(error) = {error_linear_5000:.4f}")
print(f"\nLogistic-Quadratic Classifier:")
print(f"  50 samples:    P(error) = {error_quadratic_50:.4f}")
print(f"  500 samples:   P(error) = {error_quadratic_500:.4f}")
print(f"  5000 samples:  P(error) = {error_quadratic_5000:.4f}")