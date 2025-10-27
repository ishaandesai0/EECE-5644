import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# Parameters
# True Vehicle Location
xTrue = 0.3
yTrue = 0.4

# Priors
sigmaX = 0.25
sigmaY = 0.25

# Measurement Noise Std Dev
sigmaMeasurement = 0.3

print(f"True vehicle location: ({xTrue:.2f}, {yTrue:.2f})")
print(f"Prior std dev: {sigmaX:.2f}")
print(f"Measurement noise std dev: {sigmaMeasurement:.2f}")

# Landmark Placement
def place_landmarks(K):
    """
    Place K landmarks evenly spaced on unit circle.
    """
    angles = np.linspace(0, 2*np.pi, K, endpoint=False)
    landmarks = np.array([[np.cos(a), np.sin(a)] for a in angles])
    return landmarks

# Range Measurements
def generate_measurements(xTrue, yTrue, landmarks, sigma):
    """
    Generate range measurements with Gaussian noise
    """
    K = len(landmarks)
    measurements = np.zeros(K)
    
    for i in range(K):
        # True distance
        dTrue = np.sqrt((xTrue - landmarks[i, 0])**2 + (yTrue - landmarks[i, 1])**2)
        
        # Generate measurement with noise
        while True:
            noise = np.random.normal(0, sigma)
            rI = dTrue + noise
            if rI >= 0:
                measurements[i] = rI
                break
    
    return measurements

# Map Objective Function
def map_objective(pos, measurements, landmarks, sigmaR, sigmaX, sigmaY):
    """
    MAP objective function
    """
    x, y = pos
    
    #Likelihood term
    likelihoodTerm = 0
    for i in range(len(landmarks)):
        dI = np.sqrt((x - landmarks[i, 0])**2 + (y - landmarks[i, 1])**2)
        likelihoodTerm += (measurements[i] - dI)**2 / (sigmaR**2)
        
    # Prior term
    priorTerm = (x**2 / sigmaX**2) + (y**2 / sigmaY**2)
    
    return likelihoodTerm + priorTerm

# Generate contour plots
def plot_contours(K, measurements, landmarks, xTrue, yTrue, sigmaR, sigmaX, sigmaY):
    """
    Plot contour of MAP objective function
    """
    # Make grid
    xRange = np.linspace(-2, 2, 200)
    yRange = np.linspace(-2, 2, 200)
    X, Y = np.meshgrid(xRange, yRange)
    
    # Compute objective at each grid point
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = map_objective([X[i, j], Y[i, j]], measurements, landmarks, 
                                    sigmaR, sigmaX, sigmaY)
    
    # Find MAP estimate
    min_idx = np.unravel_index(np.argmin(Z), Z.shape)
    xMap = X[min_idx]
    yMap = Y[min_idx]
    
    # Create figure
    plt.figure(figsize=(10, 10))
    
    # Plot filled contours
    levels = np.linspace(Z.min(), Z.min() + 50, 20)
    contourFilled = plt.contourf(X, Y, Z, levels=levels, cmap='viridis', alpha=0.6)
    
    # Plot contour lines
    contourLines = plt.contour(X, Y, Z, levels=levels, colors='black', alpha=0.3, linewidths=0.5)
    
    # Mark true vehicle location
    plt.plot(xTrue, yTrue, 'r+', markersize=20, markeredgewidth=3,
             label='True location', zorder=5)
    
    # Mark MAP estimate
    plt.plot(xMap, yMap, 'g*', markersize=20, markeredgewidth=2, 
             label=f'MAP estimate ({xMap:.2f}, {yMap:.2f})', zorder=5)
    
    for i, lm in enumerate(landmarks):
        plt.plot(lm[0], lm[1], 'wo', markersize=12, markeredgecolor='black', 
                 markeredgewidth=2, zorder=4)
        if i == 0:
            plt.plot([], [], 'wo', markersize=12, markeredgecolor='black', 
                     markeredgewidth=2, label='Landmarks')
            
    # Draw unit circle
    circle = plt.Circle((0, 0), 1, fill=False, color='gray', linestyle='--', linewidth=2)
    plt.gca().add_patch(circle)
    
    plt.xlabel('x', fontsize=14)
    plt.ylabel('y', fontsize=14)
    plt.title(f'MAP Objective Function Contours (K={K})', fontsize=16)
    plt.legend(fontsize=12, loc='upper right')
    plt.axis('equal')
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    plt.colorbar(contourFilled, label='Objective Function Value')
    
    plt.tight_layout()
    plt.show()
    
    return xMap, yMap

# Main

results = []

for K in [1, 2, 3, 4]:
    landmarks = place_landmarks(K)
    print(f"\nLandmark positions:")
    for i, lm in enumerate(landmarks):
        print(f"  Landmark {i+1}: ({lm[0]:.3f}, {lm[1]:.3f})")
        
    measurements = generate_measurements(xTrue, yTrue, landmarks, sigmaMeasurement)
    print(f"\nRange measurements:")
    for i, r in enumerate(measurements):
        dTrue = np.sqrt((xTrue - landmarks[i, 0])**2 + (yTrue - landmarks[i, 1])**2)
        print(f"  r_{i+1} = {r:.3f} (true distance: {dTrue:.3f})")
        
    print(f"\nGenerating contour plot...")
    xMap, yMap = plot_contours(K, measurements, landmarks, xTrue, yTrue, 
                               sigmaMeasurement, sigmaX, sigmaY)
    
    # Calculate error
    error = np.sqrt((xMap - xTrue)**2 + (yMap - yTrue)**2)
    print(f"\nMAP estimate: ({xMap:.3f}, {yMap:.3f})")
    print(f"True location: ({xTrue:.3f}, {yTrue:.3f})")
    print(f"Error: {error:.3f}")
    
    results.append({
        'K': K,
        'xMap': xMap,
        'yMap': yMap,
        'error': error
    })
    
print("\nSummary of Results:")
print(f"\nTrue vehicle location: ({xTrue:.3f}, {yTrue:.3f})")
print("\nMAP Estimates:")
print(f"{'K':<5} {'xMAP':<10} {'yMAP':<10} {'Error':<10}")
print("-" * 35)
for res in results:
    print(f"{res['K']:<5} {res['xMap']:<10.3f} {res['yMap']:<10.3f} {res['error']:<10.3f}")

# Error vs K
plt.figure(figsize=(8, 6))
K_values = [res['K'] for res in results]
errors = [res['error'] for res in results]
plt.plot(K_values, errors, 'bo-', linewidth=2, markersize=10)
plt.xlabel('Number of Landmarks (K)', fontsize=12)
plt.ylabel('Localization Error', fontsize=12)
plt.title('MAP Estimate Error vs Number Landmarks', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.xticks([1, 2, 3, 4])
plt.tight_layout()
plt.show()