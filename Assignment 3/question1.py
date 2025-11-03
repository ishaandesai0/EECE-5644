import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# Define Data Distribution (4 classes, 3D, uniform priors)
numClasses = 4
classPriors = np.ones(numClasses) / numClasses

# Class Means and Covariances
m0 = np.array([0, 0, 0])
c0 = np.array([[1.0, 0.3, 0.2],
               [0.3, 1.0, 0.1],
               [0.2, 0.1, 1.0]])

m1 = np.array([3, 0, 0])
c1 = np.array([[1.2, -0.2, 0.1],
               [-0.2, 0.8, 0.0],
               [0.1, 0.0, 1.1]])

m2 = np.array([0, 2.5, 0])
c2 = np.array([[0.9, 0.4, -0.1],
       [0.4, 1.3, 0.2],
       [-0.1, 0.2, 0.9]])

m3 = np.array([0, 0, 3])
c3 = np.array([[1.1, 0.1, 0.3],
       [0.1, 1.0, -0.2],
       [0.3, -0.2, 1.2]])

means = [m0, m1, m2, m3]
covs = [c0, c1, c2, c3]

print("Data distribution defined with 4 classes in 3D\n")

# Generate Data
def generateData(numSamples, means, covariances, classPriors):
    numClasses = len(means)
    samplesPerClass = np.random.multinomial(numSamples, classPriors)
    
    X = []
    y = []
    
    for c in range(numClasses):
        n = samplesPerClass[c]
        if n > 0:
               X_c = np.random.multivariate_normal(means[c], covariances[c], n)
               X.append(X_c)
               y.append(np.full(n, c))
               
    X = np.vstack(X)
    y = np.hstack(y)
    
    # Shuffle
    indices = np.random.permutation(len(y))
    return X[indices], y[indices]

# Generate training datasets
trainSizes = [100, 500, 1000, 5000, 10000]
trainDatasets = {}
for size in trainSizes:
       X_train, y_train = generateData(size, means, covs, classPriors)
       trainDatasets[size] = (X_train, y_train)
       print(f"Generated training set: {size} samples")

X_test, y_test = generateData(100000, means, covs, classPriors)
print(f"Generated test set: 100000 samples\n")

# Theoretical Optimal Classifier
def theoreticalOptimalClassifier(X, means, covariances, classPriors):
       numSamples = X.shape[0]
       numClasses = len(means)
       logPosteriors = np.zeros((numSamples, numClasses))
       
       for c in range(numClasses):
              rv = multivariate_normal(mean=means[c], cov=covariances[c])
              logLikelihood = rv.logpdf(X)
              logPrior = np.log(classPriors[c])
              logPosteriors[:, c] = logLikelihood + logPrior
       
       return np.argmax(logPosteriors, axis=1)

yPredOptimal = theoreticalOptimalClassifier(X_test, means, covs, classPriors)
errorOptimal = np.mean(yPredOptimal != y_test)
print(f"Theoretical Optimal Classifier Test P(error): {errorOptimal:.4f} ({errorOptimal*100:.2f}%)\n")

# Cross Validation for Model Order Selection
def crossValidateMLP(X_train, y_train, perceptronCandidates, nFolds=10):
       kf = KFold(n_splits=nFolds, shuffle=True, random_state=42)
       cvErrors = {}
       
       for numPerceptrons in perceptronCandidates:
              foldErrors = []
              
              for trainIdx, valIdx in kf.split(X_train):
                     X_tr, X_val = X_train[trainIdx], X_train[valIdx]
                     y_tr, y_val = y_train[trainIdx], y_train[valIdx]
                     
                     mlp = MLPClassifier(
                            hidden_layer_sizes=(numPerceptrons,),
                            activation='relu',
                            solver='adam',
                            max_iter=500,
                            random_state=42
                     )
                     mlp.fit(X_tr, y_tr)
                     
                     yValPred = mlp.predict(X_val)
                     foldErrors.append(np.mean(yValPred != y_val))
                     
              cvErrors[numPerceptrons] = np.mean(foldErrors)
              
       return min(cvErrors, key=cvErrors.get), cvErrors

perceptronCandidates = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30]
optimalPerceptrons = {}

print("Cross-validating...")
for size in trainSizes:
       X_train, y_train = trainDatasets[size]
       print(f"  {size} training samples...", end=" ")
       bestP, cvErrs = crossValidateMLP(X_train, y_train, perceptronCandidates)
       optimalPerceptrons[size] = bestP
       print(f"Optimal perceptrons: {bestP}, CV error: {cvErrs[bestP]:.4f}")
       
# Train final MLPs
def trainMLPMultipleInits(X_train, y_train, numPerceptrons, nInits=10):
       bestMLP = None
       bestScore = -np.inf
       
       for init in range(nInits):
              mlp = MLPClassifier(
                     hidden_layer_sizes=(numPerceptrons,),
                     activation='relu',
                     solver ='adam',
                     max_iter=1000,
                     random_state=42 + init
              )
              mlp.fit(X_train, y_train)
              score = mlp.score(X_train, y_train)
              
              if score > bestScore:
                     bestScore = score
                     bestMLP = mlp
       
       return bestMLP

trainedMLPs = {}
testErrors = {}

print("\nTraining final MLPs...")
for size in trainSizes:
       X_train, y_train = trainDatasets[size]
       numP = optimalPerceptrons[size]
       
       print(f"  {size} samples with {numP} perceptrons...", end=" ")
       mlp = trainMLPMultipleInits(X_train, y_train, numP, nInits=10)
       trainedMLPs[size] = mlp
       
       yTestPred = mlp.predict(X_test)
       error = np.mean(yTestPred != y_test)
       testErrors[size] = error
       
       print(f"Test P(error) = {error:.4f} ({error*100:.2f}%)")
       
# Plot results
print("\nGenerating plots...")

plt.figure(figsize=(10, 6))
trainSizesArr = np.array(trainSizes)
testErrorsArr = np.array([testErrors[s] for s in trainSizes])

plt.semilogx(trainSizesArr, testErrorsArr * 100, 'bo-',
             linewidth=2, markersize=8, label='MLP Classifier')
plt.axhline(y=errorOptimal * 100, color='r', linestyle='--',
            linewidth=2, label='Theoretical Optimal')

plt.xlabel('Number of Training Samples', fontsize=12)
plt.ylabel('Test P(error) [%]', fontsize=12)
plt.title('MLP Classification Performance vs Training Set Size', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=11)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
perceptronsArr = np.array([optimalPerceptrons[s] for s in trainSizes])

plt.semilogx(trainSizesArr, perceptronsArr, 'go-',
             linewidth=2, markersize=8)

plt.xlabel('Number of Training Samples', fontsize=12)
plt.ylabel('Optimal Number of Perceptrons', fontsize=12)
plt.title('Model Complexity Selection via Cross-Validation', fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()