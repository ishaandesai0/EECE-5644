import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import KFold
from PIL import Image
import requests
from io import BytesIO
import seaborn as sns

np.random.seed(42)

# Direct link to image
imageURL = "https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/segbench/BSDS300/html/images/plain/normal/color/167083.jpg"

print("Downloading image from URL...")
try:
    response = requests.get(imageURL)
    img = Image.open(BytesIO(response.content))
    imgArray = np.array(img)
    
    print(f"Image shape: {imgArray.shape}")
    print(f"Image size: {imgArray.shape[0]} x {imgArray.shape[1]}")
    print(f"Total pixels: {imgArray.shape[0] * imgArray.shape[1]}")
    
    # Display the image
    plt.figure(figsize=(10, 8))
    plt.imshow(imgArray)
    plt.title("Original Image", fontsize=16)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

except Exception as e:
    print(f"Error loading image: {e}")
    print("Trying alternative path...")
    # Try test folder is train doesn't work
    imageURL = "https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/segbench/BSDS300/images/test/167083.jpg"
    response = requests.get(imageURL)  
    img = Image.open(BytesIO(response.content))
    imgArray = np.array(img)
    
    print(f"Image shape: {imgArray.shape}")
    print(f"Image size: {imgArray.shape[0]} x {imgArray.shape[1]}")
    print(f"Total pixels: {imgArray.shape[0] * imgArray.shape[1]}")
    
    plt.figure(figsize=(10, 8))
    plt.imshow(imgArray)
    plt.title("Original Image", fontsize=16)
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    
# Feature extraction
print("\n" + "="*60)
print("Feature Extraction")
print("="*60)

# Dimensions of the image
height, width, channels = imgArray.shape
n_pixels = height * width

print(f"Creating 5D feature vectors for {n_pixels} pixels...")

# Create feature matrix with features rowNormalized, colNormalized, R Normalized, G Normalized, B Normalized
features = np.zeros((n_pixels, 5))

pixel_index = 0
for i in range(height):
    for j in range(width):
        # Normalize row and col to [0, 1]
        rowNorm = i / (height - 1) if height > 1 else 0
        colNorm = j / (width - 1) if width > 1 else 0
        
        # Normalize RGB to [0, 1]
        rNorm = imgArray[i, j, 0] / 255.0
        gNorm = imgArray[i, j, 1] / 255.0
        bNorm = imgArray[i, j, 2] / 255.0
        
        features[pixel_index] = [rowNorm, colNorm, rNorm, gNorm, bNorm]
        pixel_index += 1
        
print(f"Feature matrix shape: {features.shape}")
print(f"Feature range: [{features.min():.3f}, {features.max():.3f}]")
print("First 5 feature vectors:")
print(features[:5])

# GMM With K-Fold Cross-Validation

print("\n" + "="*60)
print("GMM Model Order Selection")
print("="*60)

# Hyperparameters to test
nComponentsRange = [2, 3, 4, 5, 6, 7, 8, 9, 10]
kFolds = 5

# K-fold cross-validation
kf = KFold(n_splits=kFolds, shuffle=True, random_state=42)

# Store results
cvLogLikelihoods = np.zeros(len(nComponentsRange))

print(f"Testing {len(nComponentsRange)} different numbers of components")
print(f"Using {kFolds}-fold cross-validation\n")

for index, nComponents in enumerate(nComponentsRange):
    foldLogLikelihoods = []
    
    for fold, (trainIndex, valIndex) in enumerate(kf.split(features)):
        X_train_fold = features[trainIndex]
        X_val_fold = features[valIndex]
        # Fit GMM
        gmm = GaussianMixture(n_components=nComponents,
                              covariance_type='full',
                              max_iter=100,
                              random_state=42)
        gmm.fit(X_train_fold)
        
        # Compute validation log-likelihood
        valLogLikelihood = gmm.score(X_val_fold)
        foldLogLikelihoods.append(valLogLikelihood)
        
    # Average log-likelihood across folds
    cvLogLikelihoods[index] = np.mean(foldLogLikelihoods)
    
    print(f"Components: {nComponents:2d} | Avg Val Log-Likelihood: {cvLogLikelihoods[index]:.4f}")

# Find best number of components
bestIndex = np.argmax(cvLogLikelihoods)
bestNComponents = nComponentsRange[bestIndex]
bestLogLikelihood = cvLogLikelihoods[bestIndex]

print(f"\nBest Number of Components: {bestNComponents}")
print(f"Best Validation Log-Likelihood: {bestLogLikelihood:.4f}")

# Visualize CV Results
plt.figure(figsize=(10, 6))
plt.plot(nComponentsRange, cvLogLikelihoods, 'o-', linewidth=2, markersize=8)
plt.axvline(x=bestNComponents, color='red', linestyle='--', label=f'Best K={bestNComponents}')
plt.xlabel("Number of Components (K)", fontsize=14)
plt.ylabel("Average Validation Log-Likelihood", fontsize=14)
plt.title('GMM Model Order Selection (5-Fold CV)', fontsize=16)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=12)
plt.tight_layout()
plt.show()

# Train Final GMM and Segment Image
print("\n" + "="*60)
print("Final GMM Training and Image Segmentation")
print("="*60)

print(f"Training final GMM with K={bestNComponents} on all data...")
finalGMM = GaussianMixture(n_components=bestNComponents,
                           covariance_type='full',
                           max_iter=100,
                           random_state=42)
finalGMM.fit(features)

print("Assigning cluster labels to each pixel...")
labels = finalGMM.predict(features)

# Reshape labels back to image dimensions
labelImage = labels.reshape((height, width))

print(f"Segmentation complete, number of unique segments: {len(np.unique(labels))}")

# Normalize labels to [0, 255] for good contrast
labelImageNormalized = (labelImage.astype(float) / labelImage.max() * 255).astype(np.uint8)

figs, axes = plt.subplots(1, 2, figsize=(20, 10))

# Original Image
axes[0].imshow(imgArray)
axes[0].set_title("Original Image", fontsize=18)
axes[0].axis('off')

# Segmented Image
axes[1].imshow(labelImageNormalized, cmap='tab10')
axes[1].set_title(f'GMM Segmentation (K={bestNComponents} components)', fontsize=18)
axes[1].axis('off')

plt.tight_layout()
plt.show()

# Segmentation with Different Colormap
plt.figure(figsize=(12, 10))
plt.imshow(labelImageNormalized, cmap='viridis')
plt.title(f'GMM-Based Image Segmentation\n(K={bestNComponents} components)', fontsize=18)
plt.colorbar(label='Segment Label')
plt.axis('off')
plt.tight_layout()
plt.show()

print("\n" + "="*60)
print(f"Optimal number of components: {bestNComponents}")
print(f"Best Validation log-likelihood: {bestLogLikelihood:.4f}")
print(f"Image segmented into {len(np.unique(labels))} distinct regions.")
print("="*60)