import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

np.random.seed(42)


# Designing 4 Gaussians

def get_class_parameters():
    """
    Define parameters for 4 Gaussian distributions in 2D
    
    Class 4 in the middle, Classes 1, 2, 3 around it
    Creates overlapping scenario where 4 overlaps the most
    """
    
    # All class priors equal
    priors = np.array([0.25, 0.25, 0.25, 0.25])
    
    # Class 1: Top
    m1 = np.array([0, 3])
    c1 = np.array([[1.5, 0.5],
                   [0.5, 1.5]])
    
    # Class 2: Bottom-Left
    m2 = np.array([-3, -1.5])
    c2 = np.array([[1.2, -0.3],
                   [-0.3, 1.8]])
    
    # Class 3: Bottom-Right
    m3 = np.array([3, -1.5])
    c3 = np.array([[1.8, 0.4],
                   [0.4, 1.2]])
    
    # Class 4: Center
    m4 = np.array([0, 0])
    c4 = np.array([[2.0, 0.5],
                   [0.5, 2.0]])
    
    means = [m1, m2, m3, m4]
    covariances = [c1, c2, c3, c4]
    
    print("\nClass Parameters:")
    for i in range(4):
        print(f"Class {i+1}:")
        print(f"  Mean: {means[i]}")
        print(f"  Covariance:\n{covariances[i]}")
        print(f"  Prior: {priors[i]}")
        
    return priors, means, covariances

def generate_data(N=10000):
    """
    Generate N samples from a mixture of 4 Gaussians
    
    Returns:
        X: (N, 2) array of samples
        y: (N,) array of class labels (1, 2, 3, or 4)
        priors, means, covariances: distribution parameters
    """
    
    priors, means, covariances = get_class_parameters()
    
    # Random selection of labels according to prior distribution
    labels = np.random.choice([1, 2, 3, 4], size=N, p=priors)
    
    # Count samples per class
    class_counts = {i: np.sum(labels == i) for i in [1, 2, 3, 4]}
    print(f"\nGenerated {N} samples (randomly according to priors):")
    for i in [1, 2, 3, 4]:
        print(f"  Class {i}: {class_counts[i]} samples ({100*class_counts[i]/N:.1f}%)")
        
    # Generate samples from corresponding Gaussians
    X = np.zeros((N, 2))
    for i in [1, 2, 3, 4]:
        mask = labels == i
        n_samples = np.sum(mask)
        if n_samples > 0:
            X[mask] = np.random.multivariate_normal(
                means[i-1],
                covariances[i-1],
                n_samples
            )
    
    return X, labels, priors, means, covariances

def visualize_data(X, labels, title="2D Data Visualization"):
    """
    Visualize 2D data with different markers for each class
    """
    plt.figure(figsize=(10, 10))
    markers = ['o', 's', '^', 'D']
    colors = ['blue', 'green', 'red', 'purple']
    
    for i in [1, 2, 3, 4]:
        mask = labels == i
        plt.scatter(X[mask, 0], X[mask, 1],
                    marker=markers[i-1],
                    c=colors[i-1],
                    alpha=0.4,
                    s=30,
                    label=f'Class {i}',
                    edgecolors='black',
                    linewidths=0.5)
    plt.xlabel('X1', fontsize=14)
    plt.ylabel('X2', fontsize=14)
    plt.title(title, fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.tight_layout()
    plt.show()
    
# Part A: Map Classification
    
def compute_class_posteriors(X, priors, means, covariances):
    """
    Compute posterior probabilities for all classes
    
    P(L=j|x) = p(x|L=j) * P(L=j) / p(x)
    
    Args:
        X: (N, 2) array of samples
        priors: (4,) array of class priors
        means: list of 4 mean vectors
        covariances: list of 4 covariance matrices
    
    Returns:
        posteriors: (N, 4) array where posteriors[i,j] = P(L=j+1|x_i)
    """
    N = X.shape[0]
    likelihoods = np.zeros((N, 4))
    
    # Compute p(x|L=j) * P(L=j) for each class
    for i in range(4):
        pdf = multivariate_normal(mean=means[i], cov=covariances[i])
        likelihoods[:, i] = pdf.pdf(X) * priors[i]
    
    # Normalize to get posteriors P(L=j|x)
    posteriors = likelihoods / (likelihoods.sum(axis=1, keepdims=True) + 1e-10)
    
    return posteriors

def map_classify(X, priors, means, covariances):
    """
    Classify samples using MAP rule
    
    Args:
        X: (N, 2) array of samples
        priors: (4,) array of class priors
        means: list of 4 mean vectors
        covariances: list of 4 covariance matrices
    
    Returns:
        decisions: (N,) array of predicted class labels (1, 2, 3, or 4)
    """
    posteriors = compute_class_posteriors(X, priors, means, covariances)
    decisions = np.argmax(posteriors, axis=1) + 1  # +1 to convert index to class label
    
    return decisions

def compute_confusion_matrix(decisions, labels, num_classes=4):
    """
    Compute confusion matrix
    
    Entry [i, j] = P(D=i|L=j) = (# samples with D=i and L=j) / (# samples with L=j)
    Returns:
        confusion_matrix: (4, 4) array
    """
    
    confusion = np.zeros((num_classes, num_classes))
    
    for true_label in range(1, num_classes + 1):
        mask = labels == true_label
        n_true = np.sum(mask)
        
        if n_true > 0:
            for decision in range(1, num_classes + 1):
                n_decided = np.sum(decisions[mask] == decision)
                confusion[decision-1, true_label-1] = n_decided / n_true
    return confusion


def visualize_classification_results(X, labels, decisions, title="Classification Results"):
    """
    Scatter plot showing correct and incorrect classifications
    Each true class has a different marker shape
    """
    plt.figure(figsize=(12, 10))
    
    markers = ['o', 's', '^', 'D']
    
    for i in [1, 2, 3, 4]:
        mask = labels == i
        correct = (decisions == labels) & mask
        incorrect = (decisions != labels) & mask
        
        if np.any(correct):
            plt.scatter(X[correct, 0], X[correct, 1],
                        marker=markers[i-1],
                        c='green',
                        alpha=0.5,
                        s=50,
                        edgecolors='darkgreen',
                        linewidths=0.5)
        if np.any(incorrect):
            plt.scatter(X[incorrect, 0], X[incorrect, 1],
                        marker=markers[i-1],
                        c='red',
                        alpha=0.7,
                        s=50,
                        edgecolors='darkred',
                        linewidths=1.5)
    plt.xlabel('X1', fontsize=14)
    plt.ylabel('X2', fontsize=14)
    plt.title(title, fontsize=16)
    
    # Custom Legend
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    
    legend_elements = [
        Patch(facecolor = 'green', alpha=0.5, edgecolor='darkgreen', label='Correct'),
        Patch(facecolor = 'red', alpha=0.7, edgecolor='darkred', label='Incorrect'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
               markersize=10, label='Class 1 (dot)'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='gray',
               markersize=10, label='Class 2 (square)'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='gray',
                markersize=10, label='Class 3 (triangle)'),
        Line2D([0], [0], marker='D', color='w', markerfacecolor='gray',
                markersize=10, label='Class 4 (diamond)'),
    ]
    plt.legend(handles=legend_elements, fontsize=11, loc='best')
    
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.tight_layout()
    plt.show()
    
    
def PartA(X, labels, priors, means, covariances):
    """
    Part A: MAP classification with 0-1 loss
    """
    print("\n" + "="*80)
    print("Question 2: Part A - MAP Classification with 0-1 Loss")
    print("="*80)
    
    print("\nDecision Rule: D = argmax_j P(L=j|x)")
    print("This minimizes probability of error (0-1 loss)")
    
    # Classify samples
    decisions = map_classify(X, priors, means, covariances)
    
    # Compute confusion matrix
    confusion = compute_confusion_matrix(decisions, labels)
    
    print("\nConfusion Matrix P(D=i|L=j):")
    print("\n(Rows: Decisions, Columns: True Labels)")
    print("\n    |   L=1   |   L=2   |   L=3   |   L=4   |")
    for i in range(4):
        print(f"D={i+1}  ", end="")
        for j in range(4):
            print(f" {confusion[i, j]:.4f}", end="")
        print()
    
    # Diagonal values
    print("\nDiagonal Values (Correct Classifications):")
    for i in range(4):
        print(f"P(D={i+1}|L={i+1}) = {confusion[i, i]:.4f}")
        
    # Calculate probability of error
    correct = np.sum(decisions == labels)
    P_error = 1 - (correct / len(labels))
    
    print(f"\nOverall Performance:")
    print(f"  Probability of Error: {P_error:.4f}")
    print(f"  Accuracy: {1 - P_error:.4f} ({100*(1 - P_error):.2f}%)")
    
    # Class wise accuracy
    print(f"\nPer-Class Accuracy:")
    for i in range(1, 5):
        mask = labels == i
        class_correct = np.sum(decisions[mask] == i)
        class_total = np.sum(mask)
        class_accuracy = class_correct / class_total if class_total > 0 else 0
        print(f"  Class {i}: {class_accuracy:.4f} ({100*class_accuracy:.2f}%)")
        
    # Visualize
    print("\nGenerating visualization...")
    visualize_classification_results(X, labels, decisions,
                                     title="Part A: MAP Classification Results")
    
    return decisions, confusion, P_error

# Part B: ERM Classification with Asymmetrical Loss

def erm_classify(X, priors, means, covariances, loss_matrix):
    """
    ERM Classification with custom loss matrix
    
    Returns:
        decisions: (N,) array of predicted class labels (1, 2, 3, or 4)
    """
    
    posteriors = compute_class_posteriors(X, priors, means, covariances)
    N = X.shape[0]
    decisions = np.zeros(N, dtype=int)
    
    # For each sample
    for n in range(N):
        # Compute expected risk for each possible decision
        risks = np.zeros(4)
        for decision in range(4):
            for true_class in range(4):
                risks[decision] += loss_matrix[decision, true_class] * posteriors[n, true_class]
                
        # Choose decision with minimum expected risk
        decisions[n] = np.argmin(risks) + 1  # +1 to convert index to class label
        
    return decisions

def compute_expected_risk(decisions, labels, loss_matrix):
    """
    Compute empirical expected risk for each possible decision
    """
    
    N = len(labels)
    total_risk = 0
    
    for n in range(N):
        decision = decisions[n] - 1  # Convert to index
        true_label = labels[n] - 1  # Convert to index
        total_risk += loss_matrix[decision, true_label]
        
    return total_risk / N

def PartB(X, labels, priors, means, covariances):
    """
    Part B: ERM Classification with Asymmetrical Loss
    """
    print("\n" + "="*80)
    print("Question 2 Part B: ERM with Asymmetrical Loss")
    print("="*80)
    
    loss_matrix = np.array([
        [0, 10, 10, 100], # Class 1
        [1, 0, 10, 100], # Class 2
        [1, 1, 0, 100], # Class 3
        [1, 1, 1, 0] # Class 4
    ])
    
    print("\nLoss Matrix:")
    print("Rows: Decisions, Columns: True Labels")
    print("\n   |   L=1   |   L=2   |   L=3   |   L=4   |")
    for i in range(4):
        print(f"D={i+1}  ", end="")
        for j in range(4):
            print(f" {loss_matrix[i, j]:3d}", end="")
        print()
        
    print("\nApplying ERM classification rule...")
    decisions = erm_classify(X, priors, means, covariances, loss_matrix)
    
    # Compute confusion matrix
    confusion = compute_confusion_matrix(decisions, labels)
    
    print("\nConfusion Matrix P(D=i|L=j):")
    print("\n(Rows: Decisions, Columns: True Labels)")
    print("\n    |   L=1   |   L=2   |   L=3   |   L=4   |")
    for i in range(4):
        print(f"D={i+1}  ", end="")
        for j in range(4):
            print(f" {confusion[i, j]:.4f}", end="")
        print()
        
    # Compute Expected Risk
    expected_risk = compute_expected_risk(decisions, labels, loss_matrix)
    print(f"\nExpected Risk: {expected_risk:.4f}")
    
    # Decision statistics
    print("\nDecision Statistics:")
    for i in range(1, 5):
        n_decided = np.sum(decisions == i)
        print(f"  Decided Class {i}: {n_decided}/{len(decisions)} ({100*n_decided/len(decisions):.1f}%)")
        
    print(f"  Class 4 is decided {100*np.sum(decisions==4)/len(decisions):.1f}% of the time!")
    
    # Error analysis for Class 4
    class4_mask = labels == 4
    class4_correct = np.sum(decisions[class4_mask] == 4)
    class4_total = np.sum(class4_mask)
    class4_accuracy = class4_correct / class4_total if class4_total > 0 else 0
    
    print(f"\nClass 4 Protection:")
    print(f"  Class 4 samples correctly classified: {class4_correct}/{class4_total} ({100*class4_accuracy:.1f}%)")

    # Visualize
    print("\nGenerating visualization...")
    visualize_classification_results(X, labels, decisions,
                                     title="Part B: ERM Classification with Asymmetrical Loss Results")
    
    return decisions, confusion, expected_risk

# Main Execution

if __name__ == "__main__":
    print("="*80)
    
    # Generate Data
    print("Generating data...")
    X, labels, priors, means, covariances = generate_data(N=10000)
    
    # Visualize Data
    print("\nVisualizing generated data...")
    visualize_data(X, labels, title="Generated 2D Data from 4 Gaussians")
    
    # Part A
    results_A = PartA(X, labels, priors, means, covariances)
    decisions_A, confusion_A, P_error_A = results_A
    
    # Part B
    results_B = PartB(X, labels, priors, means, covariances)
    decisions_B, confusion_B, expected_risk_B = results_B
    
    # Comparison
    print("\n" + "="*80)
    print("Comparison of Part A and Part B Results:")
    
    print(f"\nPart A (MAP, 0-1 Loss):")
    print(f"  Probability of Error: {P_error_A:.4f}")
    
    print(f"\nPart B (ERM, Asymmetrical Loss):")
    print(f"  Expected Risk: {expected_risk_B:.4f}")
    
    