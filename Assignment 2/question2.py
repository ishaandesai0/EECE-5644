import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

np.random.seed(42)

# Data Generation
def generate_data(N):
    gmmParameters = {}
    gmmParameters['priors'] = [.3, .4, .3]
    gmmParameters['meanVectors'] = np.array([[-10, 0, 10], [0, 0, 0], [10, 0, -10]])
    gmmParameters['covMatrices'] = np.zeros((3, 3, 3))
    gmmParameters['covMatrices'][:,:,0] = np.array([[1, 0, -3], [0, 1, 0], [-3, 0, 15]])
    gmmParameters['covMatrices'][:,:,1] = np.array([[8, 0, 0], [0, .5, 0], [0, 0, .5]])
    gmmParameters['covMatrices'][:,:,2] = np.array([[1, 0, -3], [0, 1, 0], [-3, 0, 15]])
    x, labels = generateDataGMM(N, gmmParameters)
    return x

def generateDataGMM(N, gmmParameters):
    priors = gmmParameters['priors']
    meanVectors = gmmParameters['meanVectors']
    covMatrices = gmmParameters['covMatrices']
    n = meanVectors.shape[0]
    C = len(priors)
    x = np.zeros((n, N))
    labels = np.zeros((1, N))
    
    u = np.random.random((1, N))
    thresholds = np.zeros((1, C+1))
    thresholds[:,0:C] = np.cumsum(priors)
    thresholds[:,C] = 1
    
    for i in range(C):
        indl = np.where(u <= float(thresholds[:,i]))
        Nl = len(indl[1])
        labels[indl] = (i+1)*1
        u[indl] = 1.1
        x[:,indl[1]] = np.transpose(np.random.multivariate_normal(meanVectors[:,i], covMatrices[:,:,i], Nl))
    
    return x, labels

# Generate training and validation data
print("=" * 60)
print("DATA GENERATION")
print("=" * 60)

Ntrain = 100
data_train = generate_data(Ntrain)
xTrain = data_train[0:2, :] # 2 x 100
yTrain = data_train[2, :] # 100

Nvalidate = 1000
data_validate = generate_data(Nvalidate)
xValidate = data_validate[0:2, :] # 2 x 1000
yValidate = data_validate[2, :] # 1000

print(f"Training set: {Ntrain} samples")
print(f"  x shape: {xTrain.shape}")
print(f"  y shape: {yTrain.shape}")
print(f"Validation set: {Nvalidate} samples")
print(f"  x shape: {xValidate.shape}")
print(f"  y shape: {yValidate.shape}")

# Visualize training data
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(xTrain[0,:], xTrain[1,:], yTrain, c='blue', marker='o', alpha=0.6)
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('Y')
ax.set_title('Training Dataset (100 samples)')
plt.show()

# Visalize validation data
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(xValidate[0,:], xValidate[1,:], yValidate, c='red', marker='o', alpha=0.3, s=10)
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('Y')
ax.set_title('Validation Dataset (1000 samples)')
plt.show()

# Feature Transformation
def cubic_feature(X):
    """
    Transform 2D input to cubic polynomial features.
    """
    x1 = X[0, :]
    x2 = X[1, :]
    N = X.shape[1]
    
    Phi = np.column_stack([
        np.ones(N),
        x1,
        x2,
        x1**2,
        x1*x2,
        x2**2,
        x1**3,
        (x1**2)*x2,
        x1*(x2**2),
        x2**3
    ])
    
    return Phi

# ML Estimator
def train_ML(X, y):
    """ 
    Max likelihood estimator for cubic polynomial
    """
    Phi = cubic_feature(X)
    
    # Closed form solution
    w_ML = np.linalg.solve(Phi.T @ Phi, Phi.T @ y)
    
    return w_ML

# MAP Estimator
def train_MAP(X, y, gamma, sigma2=1.0):
    """
    Maximum A Posterioiri estimator with Gaussian Prior
    """
    Phi = cubic_feature(X)
    
    # Regularization parameter
    lam = sigma2 / gamma
    
    # Ridge regression solution
    I = np.eye(Phi.shape[1])
    w_MAP = np.linalg.solve(Phi.T @ Phi + lam * I, Phi.T @ y)
    
    return w_MAP

# Prediction and Evaluation
def predict(X, w):
    """
    Make predictions using trained model
    """
    Phi = cubic_feature(X)
    return Phi @ w

def mse(y_true, y_pred):
    """
    Mean Squared Error
    """
    return np.mean((y_true - y_pred)**2)

# Train ML Estimator
print("\n" + "=" * 60)
print("ML ESTIMATOR")
print("=" * 60)

w_ML = train_ML(xTrain, yTrain)
print(f"\nTrained ML weights")
print(w_ML)

# Eval on training set
y_train_pred_ML = predict(xTrain, w_ML)
train_mse_ML = mse(yTrain, y_train_pred_ML)
print(f"Training MSE (ML): {train_mse_ML:.4f}")

# Eval on validation set
y_val_pred_ML = predict(xValidate, w_ML)
val_mse_ML = mse(yValidate, y_val_pred_ML)
print(f"Validation MSE (ML): {val_mse_ML:.4f}")

# Train Map Estimators with Different Gamma
print("\n" + "=" * 60)
print("MAP ESTIMATOR - GAMMA SWEEP")
print("=" * 60)

gamma_values = np.logspace(-4, 4, 50)
val_mse_MAP = []
train_mse_MAP = []

print(f"\nTesting {len(gamma_values)} gamma values 10^-4 to 10^4...")

for gamma in gamma_values:
    w_MAP = train_MAP(xTrain, yTrain, gamma)
    
    # Training MSE
    y_train_pred = predict(xTrain, w_MAP)
    train_mse_MAP.append(mse(yTrain, y_train_pred))
    
    # Validation MSE
    y_val_pred = predict(xValidate, w_MAP)
    val_mse_MAP.append(mse(yValidate, y_val_pred))
    
val_mse_MAP = np.array(val_mse_MAP)
train_mse_MAP = np.array(train_mse_MAP)

# Find best gamma
best_idx = np.argmin(val_mse_MAP)
best_gamma = gamma_values[best_idx]
best_val_mse = val_mse_MAP[best_idx]

print(f"\nBest gamma: {best_gamma:.4e}")
print(f"Best validation MSE: {best_val_mse:.4f}")
print(f"ML validation MSE: {val_mse_ML:.4f}")

# Train with best gamma
w_MAP_best = train_MAP(xTrain, yTrain, best_gamma)
print(f"\nBest MAP weights:")
print(w_MAP_best)

# Visualization
print("\n" + "=" * 60)
print("GENERATING PLOTS")
print("=" * 60)

# Plot: Validation MSE vs Gamma
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.semilogx(gamma_values, val_mse_MAP, 'b-', linewidth=2, label='Validation MSE')
plt.semilogx(gamma_values, train_mse_MAP, 'r--', linewidth=2, label='Training MSE')
plt.axvline(best_gamma, color='g', linestyle=':', linewidth=2, label=f'Best γ={best_gamma:.2e}')
plt.axhline(val_mse_ML, color='k', linestyle='--', linewidth=1, label=f'ML val MSE={val_mse_ML:.2f}')
plt.xlabel('Gamma (γ)', fontsize=12)
plt.ylabel('Mean Squared Error', fontsize=12)
plt.title('MSE vs Regularization Parameter (γ)', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.loglog(gamma_values, val_mse_MAP, 'b-', linewidth=2, label='Validation MSE')
plt.loglog(gamma_values, train_mse_MAP, 'r--', linewidth=2, label='Training MSE')
plt.axvline(best_gamma, color='g', linestyle=':', linewidth=2, label=f'Best γ={best_gamma:.2e}')
plt.xlabel('Gamma (γ)', fontsize=12)
plt.ylabel('Mean Squared Error', fontsize=12)
plt.title('MSE vs γ (log-log scale)', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3, which='both')

plt.tight_layout()
plt.show()

# Summary
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"\nML Estimator:")
print(f"  Training MSE: {train_mse_ML:.4f}")
print(f"  Validation MSE: {val_mse_ML:.4f}")
print(f"\nMAP Estimator (Best gamma ={best_gamma:.4e}):")
print(f"  Training MSE: {train_mse_MAP[best_idx]:.4f}")
print(f"  Validation MSE: {best_val_mse:.4f}")
print(f"\nImprovement: {((val_mse_ML - best_val_mse)/val_mse_ML * 100):.2f}%")