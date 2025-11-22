import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import seaborn as sns

np.random.seed(42)

# Data Generation Parameters
r_neg = 2 # Radius for class -1
r_pos = 4 # Radius for class +1
sigma  = 1 # Noise std dev

def generate_data(n_samples, r_neg=2, r_pos=4, sigma=1):
    """
    Generate Concentric circle data
    """
    nPerClass = n_samples // 2
    
    # Class -1
    thetaNeg = np.random.uniform(-np.pi, np.pi, nPerClass)
    xNeg = r_neg * np.column_stack([np.cos(thetaNeg), np.sin(thetaNeg)])
    noiseNeg = np.random.randn(nPerClass, 2) * sigma
    X_neg = xNeg + noiseNeg
    yNeg = -np.ones(nPerClass)
    
    # Class +1
    thetaPos = np.random.uniform(-np.pi, np.pi, nPerClass)
    xPos = r_pos * np.column_stack([np.cos(thetaPos), np.sin(thetaPos)])
    noisePos = np.random.randn(nPerClass, 2) * sigma
    X_pos = xPos + noisePos
    yPos = np.ones(nPerClass)
    
    # Combine
    X = np.vstack([X_neg, X_pos])
    y = np.hstack([yNeg, yPos])
    
    # Shuffle
    indices = np.random.permutation(n_samples)
    X = X[indices]
    y = y[indices]
    
    return X, y

# Generate Training and Test Data
X_train, y_train = generate_data(1000)
X_test, y_test = generate_data(10000)

print(f"Training data: {X_train.shape}, labels: {y_train.shape}")
print(f"Test data: {X_test.shape}, labels: {y_test.shape}")
print(f"Class distribution in training: {np.sum(y_train == -1)} class -1, {np.sum(y_train == 1)} class +1")
print(f"Class distribution in test: {np.sum(y_test == -1)} class -1, {np.sum(y_test == 1)} class +1")

# Visualize training data
plt.figure(figsize=(10, 10))
plt.scatter(X_train[y_train == -1, 0], X_train[y_train == -1, 1],
            c='blue', label='Class -1', alpha=0.6, edgecolors='k')
plt.scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1],
            c='red', label='Class +1', alpha=0.6, edgecolors='k')
plt.xlabel('X1', fontsize=14)
plt.ylabel('X2', fontsize=14)
plt.title('Training Data: Concentric Circles with Noise', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.axis('equal')
plt.tight_layout()
plt.show()

# Visualize Test Data
plt.figure(figsize=(10, 10))
plt.scatter(X_test[y_test == -1, 0], X_test[y_test == -1, 1],
            c='blue', label='Class -1', alpha=0.3, s=10)
plt.scatter(X_test[y_test == 1, 0], X_test[y_test == 1, 1],
            c='red', label='Class +1', alpha=0.3, s=10)
plt.xlabel('X1', fontsize=14)
plt.ylabel('X2', fontsize=14)
plt.title('Test Data: Concentric Circles with Noise', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.axis('equal')
plt.tight_layout()
plt.show()

# SVM with K-Fold Cross-Validation
# Hyperparameter ranges
gammaVals = [0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10]
cVals = [0.1, 1, 10, 100, 1000]

k_folds = 5
kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

# Store results
cv_results = np.zeros((len(gammaVals), len(cVals)))

print("Running SVM with K-Fold Cross-Validation...")
print(f"Testing {len(gammaVals)} gamme values x {len(cVals)} C values = {len(gammaVals) * len(cVals)} combinations")
print(f"Using {k_folds}-fold CV\n")

# Grid search over hyperparameters
for i, gamma in enumerate(gammaVals):
    for j, C in enumerate(cVals):
        fold_accuracies = []
        
        for train_index, val_index in kf.split(X_train):
            X_train_fold = X_train[train_index]
            y_train_fold = y_train[train_index]
            X_val_fold = X_train[val_index]
            y_val_fold = y_train[val_index]
            
            # Train SVM
            svm = SVC(kernel='rbf', gamma=gamma, C=C, random_state=42)
            svm.fit(X_train_fold, y_train_fold)
            
            # Validate
            y_val_pred = svm.predict(X_val_fold)
            accuracy = accuracy_score(y_val_fold, y_val_pred)
            fold_accuracies.append(accuracy)
            
        # Average accuracy across folds
        cv_results[i, j] = np.mean(fold_accuracies)
        
    print(f"Completed gamma = {gamma:.3f}")
    
print("\nCross-Validation Complete!")

# Find best hyperparameters
best_index = np.unravel_index(np.argmax(cv_results), cv_results.shape)
best_gamma = gammaVals[best_index[0]]
best_C = cVals[best_index[1]]
best_accuracy = cv_results[best_index]

print(f"\nBest Hyperparameters:")
print(f"  Gamma: {best_gamma}")
print(f"  C: {best_C}")
print(f"  CV Accuracy: {best_accuracy:.4f}")
print(f"  CV Error: {1 - best_accuracy:.4f}")

# Visualize CV Results
plt.figure(figsize=(12, 8))
sns.heatmap(cv_results, annot=True, fmt='.4f', cmap='viridis',
            xticklabels=[f'{c}' for c in cVals],
            yticklabels=[f'{g}' for g in gammaVals],
            cbar_kws={'label': 'Validation Accuracy'})
plt.xlabel('C (Box Constraint)', fontsize=14)
plt.ylabel('Gamma (Kernel Width)', fontsize=14)
plt.title('SVM Cross-Validation Results (5-Fold)', fontsize=16)
plt.tight_layout()
plt.show()

# Train final SVM on full training set with best hyperparameters
print("\nTraining final SVM model on full training set...")
svm_final = SVC(kernel='rbf', gamma=best_gamma, C=best_C, random_state=42)
svm_final.fit(X_train, y_train)

# Evaluate on test set
y_test_pred = svm_final.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)
test_error = 1 - test_accuracy

print(f"\nTest Set Performance:")
print(f"  Test Accuracy: {test_accuracy:.4f}")
print(f"  Test Error: {test_error:.4f}")
print(f"  Correct: {np.sum(y_test == y_test_pred)} / {len(y_test)}")

# Visualize SVM Decision Boundary on Test Data

def plot_decision_boundary(model, X, y, title, resolution=500):
    """
    Plot decision boundary with test data
    """
    # Create mesh grid
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, resolution),
                         np.linspace(y_min, y_max, resolution))
    
    # Predict on mesh grid
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot
    plt.figure(figsize=(12, 12))
    
    # Plot decision boundary
    plt.contourf(xx, yy, Z, alpha=0.3, levels=[-1.5, 0, 1.5], colors=['blue', 'red'])
    plt.contour(xx, yy, Z, levels=[0], colors='black', linewidths=2)
    
    # Plot test data
    plt.scatter(X[y == -1, 0], X[y == -1, 1],
                c='blue', label='Class -1', alpha=0.4, s=10, edgecolors='none')
    plt.scatter(X[y == 1, 0], X[y == 1, 1],
                c='red', label='Class +1', alpha=0.4, s=10, edgecolors='none')
    plt.xlabel('X1', fontsize=14)
    plt.ylabel('X2', fontsize=14)
    plt.title(title, fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.tight_layout()
    plt.show()
    
plot_decision_boundary(svm_final, X_test, y_test,
                       f'SVM Decision Boundary (y={best_gamma}, C={best_C})\nTest Accuracy: {test_accuracy:.4f}')

# MLP with K-Fold Cross-Validation

# Hyperparameter ranges
hidden_layer_sizes = [3, 5, 10, 15, 20, 30, 50]
activation_functions = ['tanh', 'relu', 'logistic']

# Store results
mlp_cv_results = np.zeros((len(hidden_layer_sizes), len(activation_functions)))

print("Running MLP K-Fold Cross-Validation...")
print(f"Testing {len(hidden_layer_sizes)} hidden layer sizes x {len(activation_functions)} activations = {len(hidden_layer_sizes) * len(activation_functions)} combinations")
print(f"Using {k_folds}-fold CV\n")

# Grid search over hyperparameters
for i, n_neurons in enumerate(hidden_layer_sizes):
    for j, activation in enumerate(activation_functions):
        fold_accuracies = []
        
        for train_index, val_index in kf.split(X_train):
            X_train_fold = X_train[train_index]
            y_train_fold = y_train[train_index]
            X_val_fold = X_train[val_index]
            y_val_fold = y_train[val_index]
            
            # Train MLP
            mlp = MLPClassifier(hidden_layer_sizes=(n_neurons,),
                                activation=activation,
                                solver='adam',
                                max_iter=1000,
                                random_state=42,
                                early_stopping=True,
                                validation_fraction=0.1)
            mlp.fit(X_train_fold, y_train_fold)
            
            # Validate
            y_val_pred = mlp.predict(X_val_fold)
            accuracy = accuracy_score(y_val_fold, y_val_pred)
            fold_accuracies.append(accuracy)
            
        # Average accuracy across folds
        mlp_cv_results[i, j] = np.mean(fold_accuracies)
        
    print(f"Completed hidden neurons = {n_neurons}")
    
print("\nCross-Validation Complete!")

# Find best hyperparameters
best_mlp_index = np.unravel_index(np.argmax(mlp_cv_results), mlp_cv_results.shape)
best_n_neurons = hidden_layer_sizes[best_mlp_index[0]]
best_activation = activation_functions[best_mlp_index[1]]
best_mlp_accuracy = mlp_cv_results[best_mlp_index]

print(f"\nBest MLP Hyperparameters:")
print(f"  Hidden Neurons: {best_n_neurons}")
print(f"  Activation: {best_activation}")
print(f"  CV Accuracy: {best_mlp_accuracy:.4f}")
print(f"  CV Error: {1 - best_mlp_accuracy:.4f}")

# Visualize MLP CV Results
plt.figure(figsize=(10, 10))
sns.heatmap(mlp_cv_results, annot=True, fmt='.4f', cmap='viridis',
            xticklabels=activation_functions,
            yticklabels=hidden_layer_sizes,
            cbar_kws={'label': 'Validation Accuracy'})
plt.xlabel('Activation Function', fontsize=14)
plt.ylabel('Hidden Layer Size', fontsize=14)
plt.title('MLP Cross-Validation Results (5-Fold)', fontsize=16)
plt.tight_layout()
plt.show()

# Train final MLP on full training set with best hyperparameters
print("\nTraining final MLP model on full training set...")
mlp_final = MLPClassifier(hidden_layer_sizes=(best_n_neurons,),
                          activation=best_activation,
                          solver='adam',
                            max_iter=1000,
                            random_state=42,
                            early_stopping=True,
                            validation_fraction=0.1)
mlp_final.fit(X_train, y_train)

# Evaluate on test set
y_test_pred_mlp = mlp_final.predict(X_test)
test_accuracy_mlp = accuracy_score(y_test, y_test_pred_mlp)
test_error_mlp = 1 - test_accuracy_mlp

print(f"\nMLP Test Set Performance:")
print(f"  Test Accuracy: {test_accuracy_mlp:.4f}")
print(f"  Test Error: {test_error_mlp:.4f}")
print(f"  Correct: {np.sum(y_test == y_test_pred_mlp)} / {len(y_test)}")

# Plot MLP Decision Boundary on Test Data
plot_decision_boundary(mlp_final, X_test, y_test,
                       f'MLP Decision Boundary (Neurons={best_n_neurons}, Activation={best_activation})\nTest Accuracy: {test_accuracy_mlp:.4f}')

# Comparison Visualization

# Side by side comparison of SVM and MLP decision boundaries
fig, axes = plt.subplots(1, 2, figsize=(24, 10))

# Function to plot on specific axis
def plot_boundary_on_axis(ax, model, X, y, title, resolution=500):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, resolution),
                         np.linspace(y_min, y_max, resolution))
    
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    ax.contourf(xx, yy, Z, alpha=0.3, levels=[-1.5, 0, 1.5], colors=['blue', 'red'])
    ax.contour(xx, yy, Z, levels=[0], colors='black', linewidths=2)
    ax.scatter(X[y == -1, 0], X[y == -1, 1], c='blue', label='Class -1',
               alpha=0.4, s=10, edgecolors='none')
    ax.scatter(X[y == 1, 0], X[y == 1, 1], c='red', label='Class +1',
               alpha=0.4, s=10, edgecolors='none')
    ax.set_xlabel('X1', fontsize=14)
    ax.set_ylabel('X2', fontsize=14)
    ax.set_title(title, fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
# Plot SVM
plot_boundary_on_axis(axes[0], svm_final, X_test, y_test,
                       f'SVM (y={best_gamma}, C={best_C})\nTest Accuracy: {test_accuracy:.4f}')

# Plot MLP
plot_boundary_on_axis(axes[1], mlp_final, X_test, y_test,
                       f'MLP (Neurons={best_n_neurons}, Activation={best_activation})\nTest Accuracy: {test_accuracy_mlp:.4f}')

plt.tight_layout()
plt.show()

# Performance Comparison Table
print("\n" + "="*60)
print("="*60)

comparisonData = {
    'Metric': ['CV Accuracy', 'CV Error', 'Test Accuracy', 'Test Error',
               'Correct Predictions'],
    'SVM': [f"{best_accuracy:.4f}", f"{1 - best_accuracy:.4f}",
            f"{test_accuracy:.4f}", f"{test_error:.4f}",
            f"{np.sum(y_test == y_test_pred)} / {len(y_test)}"],
    'MLP': [f"{best_mlp_accuracy:.4f}", f"{1 - best_mlp_accuracy:.4f}",
            f"{test_accuracy_mlp:.4f}", f"{test_error_mlp:.4f}",
            f"{np.sum(y_test == y_test_pred_mlp)} / {len(y_test)}"]
}

print("\n{:<25} {:<20} {:<20}".format('Metric', 'SVM', 'MLP'))
print("-"*60)
for i in range(len(comparisonData['Metric'])):
    print("{:<25} {:<20} {:<20}".format(comparisonData['Metric'][i],
                                        comparisonData['SVM'][i],
                                        comparisonData['MLP'][i]
    ))

print("\n" + "="*60)
print("HYPERPARAMETERS")
print("="*60)
print(f"SVM:     Gamma = {best_gamma}, C = {best_C}")
print(f"MLP:     Hidden Neurons = {best_n_neurons}, Activation = {best_activation}")
print("="*60)

# Bar Chart Comparison
fig, ax = plt.subplots(figsize=(10, 6))
metrics = ['CV Accuracy', 'Test Accuracy']
svmScores = [best_accuracy, test_accuracy]
mlpScores = [best_mlp_accuracy, test_accuracy_mlp]

x = np.arange(len(metrics))
width = 0.35

bars1 = ax.bar(x - width/2, svmScores, width, label='SVM', color='steelblue')
bars2 = ax.bar(x + width/2, mlpScores, width, label='MLP', color='coral')

ax.set_ylabel('Accuracy', fontsize=14)
ax.set_title('SVM vs MLP Performance Comparison', fontsize=16)
ax.set_xticks(x)
ax.set_xticklabels(metrics, fontsize=12)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim([0.7, 0.9])

#Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}',
                ha='center', va='bottom', fontsize=11)

plt.tight_layout()
plt.show()