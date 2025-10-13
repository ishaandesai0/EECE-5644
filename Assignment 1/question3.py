import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import pandas as pd
from sklearn.decomposition import PCA
import os

np.random.seed(42)

BASE_DIR = r"C:\\Users\\ishaa\\Documents\\NEU 25-26\\EECE 5644\\Homework\\Assignment 1"

# Loading in dataset
def load_wine_quality():
    """
    Load wine quality dataset    
    """
    print("\nLoading Wine Quality Dataset...")
    
    # Load White
    white_path=os.path.join(BASE_DIR, 'winequality-white.csv')
    
    wine_data = pd.read_csv(white_path, sep=';')
    
    print(f"    Total samples: {len(wine_data)}")
    print(f"    Features: {wine_data.shape[1] - 1}")
    print(f"    Quality scores: {sorted(wine_data['quality'].unique())}")
    print(f"\n    Quality distribution:")
    for quality, count in wine_data['quality'].value_counts().sort_index().items():
        print(f"        Quality {quality}: {count} samples")
        
    # Extract features and labels
    X = wine_data.iloc[:, :-1].values # All columns except last
    Y = wine_data.iloc[:, -1].values  # Last column
    
    return X, Y

def load_har():
    """
    Load Human Activity Recognition dataset
    """
    print("\nLoading Human Activity Recognition Dataset...")
    
    har_dir = os.path.join(BASE_DIR, 'UCI HAR Dataset', 'UCI HAR Dataset')
    
    X_train = np.loadtxt(os.path.join(har_dir, 'train', 'X_train.txt'))
    Y_train = np.loadtxt(os.path.join(har_dir, 'train', 'y_train.txt'))
    
    X_test = np.loadtxt(os.path.join(har_dir, 'test', 'X_test.txt'))
    Y_test = np.loadtxt(os.path.join(har_dir, 'test', 'y_test.txt'))
    
    X = np.vstack((X_train, X_test))
    Y = np.hstack((Y_train, Y_test)).astype(int)
    
    print(f"    Total samples: {len(X)}")
    print(f"    Features: {X.shape[1]}")
    print(f"    Activities (1-6): {sorted(np.unique(Y))}")
    print(f"\n    Activity distribution:")
    
    activity_names = {
        1: "Walking",
        2: "Walking Upstairs",
        3: "Walking Downstairs",
        4: "Sitting",
        5: "Standing",
        6: "Laying"
    }
    
    for activity in sorted(np.unique(Y)):
        count = np.sum(Y == activity)
        print(f"    Activity {activity} ({activity_names[activity]}): {count} samples")
    
    return X, Y

# Gaussian Classifier with Regularization

def estimate_class_parameters(X, Y, alpha=0.01):
    """
    Estimate Gaussian parameters for each class with regularization
    """
    classes = np.unique(Y)
    n_classes = len(classes)
    d = X.shape[1]
    
    means = []
    covariances = []
    priors = []
    
    print(f"\nEstimating Gaussian parameters (with alpha={alpha} regularization)...")
    
    for i, c in enumerate(classes):
        # Get samples
        X_c = X[Y == c]
        n_c = len(X_c)
        
        # Estimate mean
        mean = np.mean(X_c, axis=0)
        
        # Estimate covariance
        cov = np.cov(X_c.T)
        
        try:
            cond_number = np.linalg.cond(cov)
        except:
            cond_number = np.inf
        
        # Apply regularization
        eigenvalues = np.linalg.eigvalsh(cov)
        pos_eigenvalues = eigenvalues[eigenvalues > 1e-10]
        
        if len(pos_eigenvalues) > 0:
            mean_eigenvalue = np.mean(pos_eigenvalues)
            lambda_reg = alpha * mean_eigenvalue
        else:
            lambda_reg = alpha
            
        cov_reg = cov + lambda_reg* np.eye(d)
        
        print(f"    Class {int(c)}: {n_c} samples, cond(C)={cond_number:.2e}, lambda={lambda_reg:.6f}")
        
        means.append(mean)
        covariances.append(cov_reg)
        priors.append(n_c / len(X))
        
    priors = np.array(priors)
    
    return means, covariances, priors, classes

def map_classify_gaussian(X, means, covariances, priors, classes):
    """
    MAP classification using Gaussian Models
    """
    N = X.shape[0]
    n_classes = len(classes)
    
    # Compute log posteriors
    log_posteriors = np.zeros((N, n_classes))
    
    for i, c in enumerate(classes):
        try:
            mvn = multivariate_normal(mean=means[i], cov=covariances[i], allow_singular=True)
            log_posteriors[:, i] = mvn.logpdf(X) + np.log(priors[i] + 1e-10)
        except Exception as e:
            print(f"    Warning: Issue with class {c}: {e}")
            log_posteriors[:, i] = -np.inf
            
    # Get class with highest posterior
    decision_indices = np.argmax(log_posteriors, axis=1)
    decisions = classes[decision_indices]
    
    return decisions

def compute_confusion_matrix_general(decisions, labels, classes):
    """
    Compute confusion matrix
    """
    n_classes = len(classes)
    confusion = np.zeros((n_classes, n_classes))
    
    for i, true_class in enumerate(classes):
        mask = labels == true_class
        n_true = np.sum(mask)
        
        if n_true > 0:
            for j, pred_class in enumerate(classes):
                n_pred = np.sum(decisions[mask] == pred_class)
                confusion[j, i] = n_pred / n_true
    
    return confusion

# Visualization

def visualize_data_pca(X, Y, dataset_name, n_components=2):
    """
    2D PCA Visualization
    """
    print(f"\nPerforming 2D PCA...")
    
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    
    explained_var = pca.explained_variance_ratio_
    print(f"  PC1: {explained_var[0]:.2%}, PC2: {explained_var[1]:.2%}")
    print(f"  Total: {sum(explained_var):.2%}")
    
    plt.figure(figsize=(12, 8))
    
    classes = np.unique(Y)
    colors = plt.cm.tab10(np.linspace(0, 1, len(classes)))
    
    for i, c in enumerate(classes):
        mask = Y == c
        plt.scatter(X_pca[mask, 0], X_pca[mask, 1],
                    c=[colors[i]], label=f'Class {int(c)}',
                    alpha=0.5, s=20, edgecolors='black', linewidths=0.3)
        
    plt.xlabel(f'PC1 ({explained_var[0]:.1%} variance)', fontsize=12)
    plt.ylabel(f'PC2 ({explained_var[1]:.1%} variance)', fontsize=12)    
    plt.title(f'{dataset_name}: 2D PCA Visualization', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
def visualize_data_3d_pca(X, y, dataset_name):
    """
    3D PCA Visualization
    """
    print(f"\nPerforming 3D PCA...")
    
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X)
    
    explained_var = pca.explained_variance_ratio_
    print(f"    PC1: {explained_var[0]:.2%}, PC2: {explained_var[1]:.2%}, PC3: {explained_var[2]:.2%}")
    print(f"    Total: {sum(explained_var):.2%}")
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    classes = np.unique(y)
    colors = plt.cm.tab10(np.linspace(0, 1, len(classes)))
    
    for i, c in enumerate(classes):
        mask = y == c
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1], X_pca[mask, 2],
                   c=[colors[i]], label=f'Class {int(c)}',
                   alpha=0.5, s=20)
        
    ax.set_xlabel(f'PC1 ({explained_var[0]:.1%})', fontsize=11)
    ax.set_ylabel(f'PC2 ({explained_var[1]:.1%})', fontsize=11)
    ax.set_zlabel(f'PC3 ({explained_var[2]:.1%})', fontsize=11)
    ax.set_title(f'{dataset_name}: 3D PCA', fontsize=14)
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.show()
    
# Analysis
def analyze_dataset(X, Y, dataset_name, alpha=0.01):
    """
    Analysis
    """
    print("\n" + "="*80)
    print(f"Analyzing: {dataset_name}")
    print("="*80)
    
    print(f"\nDataset Statistics")
    print(f"    Samples: {X.shape[0]}")
    print(f"    Features: {X.shape[1]}")
    print(f"    Classes: {len(np.unique(Y))}")
    
    # Visualize
    visualize_data_pca(X, Y, dataset_name, n_components=2)
    visualize_data_3d_pca(X, Y, dataset_name)
    
    # Estimate parameters
    means, covariances, priors, classes = estimate_class_parameters(X, Y, alpha=alpha)
    
    # Classify
    print(f"\nClassifying with MAP rule...")
    decisions = map_classify_gaussian(X, means, covariances, priors, classes)
    
    # Confision Matrix
    confusion = compute_confusion_matrix_general(decisions, Y, classes)
    
    print(f"\nConfusion Matrix P(D=i|L=j)")
    print("(Rows = Decisions, Columns = True Labels)")
    
    # Print header
    print("\n    ", end="")
    for c in classes:
        print(f"    L={int(c):2d}", end="")
    print()
    
    # Print matrix
    for i, pred_class in enumerate(classes):
        print(f"D={int(pred_class):2d}", end="")
        for j in range(len(classes)):
            print(f"  {confusion[i,j]:.3f}", end="")
        print()
        
    # Diagonal
    print(f"\nDiagonal (Correct):")
    for i, c in enumerate(classes):
        print(f"    P(D={int(c)}|L={int(c)}) = {confusion[i,i]:.4f}")
    
    # Accuracy
    correct = np.sum(decisions == Y)
    accuracy = correct / len(Y)
    error_rate = 1 - accuracy
    
    print(f"\nOverall Performance:")
    print(f"    Accuracy: {accuracy:.4f} ({100*accuracy:.2f})")
    print(f"    Error Rate: {error_rate:.4f} ({100*error_rate:.2f})")
    
    # Per Class
    print(f"\nPer-Class Accuracy:")
    for c in classes:
        mask = Y == c
        class_correct = np.sum(decisions[mask] == c)
        class_total = np.sum(mask)
        class_acc = class_correct / class_total if class_total > 0 else 0
        print(f"    Class {int(c)}: {class_acc:.4f} ({100*class_acc:.2f})")
        
    return means, covariances, priors, classes, confusion, accuracy

def discuss_app(dataset_name, X, Y, accuracy, confusion):
    """
    Discuss Gaussian appropriateness
    """
    print("\n" + "="*80)
    print(f"Discussion: Gaussian Model for {dataset_name}")
    print("="*80)
    
    print(f"\nPerformance:")
    print(f"    Accuracy: {100*accuracy:.2f}%")
    
    diagonal_mean = np.mean(np.diag(confusion))
    print(f"    Mean diagonal: {diagonal_mean:.3f}")
    
    if accuracy > 0.8:
        print(f"\nGood Fit")
        print(f"    Data appears Gaussian")
    elif accuracy > 0.6:
        print(f"\n Moderate Fit")
        print(f"    Gaussian assumption partially violated")
    else:
        print(f"\n Poor fit")
        print(f"    Gaussian assumption inappropriate")
    

# Main
if __name__ == "__main__":
    # Wine Quality
    print("\n" + "="*80)
    print("Dataset 1: Wine Quality")
    print("="*80)
    try:
        X_wine, Y_wine = load_wine_quality()
        results_wine = analyze_dataset(X_wine, Y_wine, "Wine Quality", alpha=0.01)
        discuss_app("Wine Quality", X_wine, Y_wine, results_wine[5], results_wine[4])
    except Exception as e:
        print(f"\n Error: {e}")
        
    print("\n\n")
    
    # HAR
    print("\n" + "="*80)
    print("Dataset 2: Human Activity Recognition")
    print("="*80)
    try:
        X_har, Y_har = load_har()
        results_har = analyze_dataset(X_har, Y_har, "HAR", alpha=0.01)
        discuss_app("HAR", X_har, Y_har, results_har[5], results_har[4])
    except Exception as e:
        print(f"\n Error: {e}")