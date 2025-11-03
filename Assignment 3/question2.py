import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import KFold
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# Define True GMM (4 components, 2D, 2 overlapping)
trueNumComponents = 4

# Component weights
trueWeights = np.array([0.25, 0.25, 0.25, 0.25])

# Component 1
m1 = np.array([0, 0])
c1 = np.array([[1.0, 0.3],
               [0.3, 1.0]])

# Component 2 (overlap with Component 1)
m2 = np.array([1.5, 1.5])
c2 = np.array([[1.2, -0.2],
       [-0.2, 1.2]])

# Component 3 (well separated)
m3 = np.array([6, 0])
c3 = np.array([[0.8, 0.1],
               [0.1, 0.9]])

# Component 4 (well separated)
m4 = np.array([3, 6])
c4 = np.array([[1.0, 0.2],
               [0.2, 1.1]])

trueMeans = [m1, m2, m3, m4]
trueCov = [c1, c2, c3, c4]

print("True GMM defined:")
print(f"  Components: {trueNumComponents}")
print(f"  Weights: {trueWeights}")
print(f"  Components 1 and 2 overlap significantly\n")

# Generating Data
def generateGMMData(numSamples, means, covs, weights):
    """
    Generate samples from GMM
    """
    numComponents = len(means)
    
    # Sample which component each data point comes from
    componentSamples = np.random.choice(numComponents, size=numSamples, p=weights)
    
    # Generate samples
    X = np.zeros((numSamples, 2))
    for i in range(numSamples):
        comp = componentSamples[i]
        X[i] = np.random.multivariate_normal(means[comp], covs[comp])
        
    return X

X_vis = generateGMMData(1000, trueMeans, trueCov, trueWeights)
plt.figure(figsize=(8, 6))
plt.scatter(X_vis[:, 0], X_vis[:, 1], alpha=0.5, s=10)
plt.title('True GMM Distribution (1000 samples)')
plt.xlabel('x1')
plt.ylabel('x2')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Cross Validation for Model Order Selection
def cross_validate_gmm(X, model_orders, n_folds=10):
    """
    Perform k-fold cross validation to select GMM order
    Returns best model order and all CV log-likelihoods
    """
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    cv_log_likelihoods = {order: [] for order in model_orders}
    
    for order in model_orders:
        for train_idx, val_idx in kf.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            
            # Train GMM with EM algo
            gmm = GaussianMixture(
                n_components=order,
                covariance_type='full',
                max_iter=200,
                random_state=42,
                n_init=5
            )
        
            try:
                gmm.fit(X_train)
                # Eval on validation set
                log_likelihood = gmm.score(X_val) * len(X_val)
                cv_log_likelihoods[order].append(log_likelihood)
            except:
                cv_log_likelihoods[order].append(-1e10)
    
    # Mean log-likelihood for each order
    mean_log_likelihoods = {order: np.mean(cv_log_likelihoods[order])
                            for order in model_orders}
    
    # Select best order
    best_order = max(mean_log_likelihoods, key=mean_log_likelihoods.get)
    
    return best_order, mean_log_likelihoods

# Run

datasetSizes = [10, 100, 1000]
modelOrders = list(range(1, 11))
numExperiments = 100

# Store results
results = {size: {order: 0 for order in modelOrders} for size in datasetSizes}

print("Running experiments...")
print("="*60)

for size in datasetSizes:
    print(f"\nDataset Size: {size} samples")
    print(f"Running {numExperiments} experiments...")
    
    for exp in range(numExperiments):
        if (exp + 1) % 20 == 0:
            print(f"  Experiment {exp + 1}/{numExperiments}")
            
        X = generateGMMData(size, trueMeans, trueCov, trueWeights)
        bestOrder, _ = cross_validate_gmm(X, modelOrders, n_folds=min(10, size))
        results[size][bestOrder] += 1
        
    print(f"  Completed {numExperiments} experiments")

# Results
print("\n" + "="*60)
print("RESULTS SUMMARY")
print("="*60)

for size in datasetSizes:
    print(f"\nDataset Size: {size} samples")
    print(f"{'Model Order':<15} {'Times Selected':<20} {'Selection Rate'}")
    print("-" * 60)
    
    for order in modelOrders:
        count = results[size][order]
        rate = count / numExperiments
        if count > 0:
            print(f"{order:<15} {count:<20} {rate:.3f} ({rate*100:.1f}%)")
            
# Visualize
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for idx, size in enumerate(datasetSizes):
    orders = list(modelOrders)
    counts = [results[size][order] for order in orders]
    rates = [c / numExperiments * 100 for c in counts]
    
    axes[idx].bar(orders, rates, color='steelblue', alpha=0.7)
    axes[idx].axvline(x=trueNumComponents, color='red', linestyle='--',
                      linewidth=2, label='True order(4)')
    axes[idx].set_xlabel('Num Components', fontsize=11)
    axes[idx].set_ylabel('Selection Rate (%)', fontsize=11)
    axes[idx].set_title(f'{size} Samples', fontsize=12)
    axes[idx].set_xticks(orders)
    axes[idx].grid(True, alpha=0.3, axis='y')
    axes[idx].legend()
    
plt.tight_layout()
plt.show()