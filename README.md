# 10.Parameter-estimation
Nonparametric probability density estimation can be implemented based on the nearest-neighbour approach. In this case, the density estimate
where k is a fixed number of neighbours, N is the number of samples, and V(x) the (local) length/area/volume that contains the specified number of samples.
Implement a nearest-neighbour estimator in Matlab/Python.
Use the provided train data to fit the estimator of the probability density function (PDF).
For having a distribution to compare with, use the normal distribution and estimate its parameters from the data using MLE.
The best value for k ∈ {5, 10, 20, 40} is 10 ✅ based on the Kullback-Leibler divergence between the nearest-neighbour estimate and the normal distribution on the provided test data.
Additional files: Train (CSV), Train (MAT), Test (CSV), Test (MAT)
Hints:
Kullback-Leibler (KL) divergence is an information-theoretic measure for the difference between two distributions.
For distributions p(x) and q(x), it is defined as
where the base of the logarithm is either 2 (bits) or e (nats). Based on the definition, KL divergence is not symmetric, that is,
![ArmanGolbidi](https://github.com/user-attachments/assets/ee35f6d6-4b52-471d-8d37-d46661112aec)

For the testing, p(x) is the estimated PDF from train data, q(x) is the PDF of the normal distribution fitted to the train data, and use provided test data as X.

Let us compare KDE and KNN for the purpose of density estimation:

Plot a histogram of the train data with the correct normalization parameter to get an estimate of the probability density (histogram in MATLAB and matplotlib.pyplot.hist in Python).

On the same figure, do a line plot of the PDF of the normal distribution fitted to the train data. The PDF should be evaluated at the points from test data. If the histogram is properly normalized, the plot and the histogram should have the same scale.

On the same figure, do a line plot of the PDF estimated from train data using the KNN approach with the best k from question 1. The PDF should be evaluated at the points from test data.

On the same figure, do a line plot of the PDF estimated from train data using the KDE approach. The PDF should be evaluated at the points from test data. Try to find a parameter h that would provide a good fit.

Attach the image of the plot to this question.
![1](https://github.com/user-attachments/assets/dcc6b607-a575-4587-86b6-d63218f29d40)
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
from scipy.stats import norm, entropy
from sklearn.neighbors import NearestNeighbors
# Load datasets
trainSet = pd.read_csv('x.csv')
testSet = pd.read_csv('test_x.csv')

# Convert values to a flat array for further processing
trainArray = trainSet.values.flatten()
testArray = testSet.values.flatten()
trainArray

def neighborDensityEstimator(dataset, neighbors, eval_points):
    knn_model = NearestNeighbors(n_neighbors=neighbors).fit(dataset.reshape(-1, 1))
    distances, _ = knn_model.kneighbors(eval_points.reshape(-1, 1))
    radius = distances[:, -1]
    density_val = neighbors / (len(dataset) * (radius ** len(dataset.shape)))
    return density_val
# Maximum likelihood estimation for fitting a normal distribution
mean_est, std_dev_est = norm.fit(trainArray)
print(f"Normal distribution parameters - Mean: {mean_est}, Std Dev: {std_dev_est}")
def kl_divergence(p_val, q_val):
    return entropy(p_val, q_val)

optimal_k = None
lowest_kl_divergence = float('inf')
k_values = [5, 10, 20, 40]

for neighbors in k_values:
    knn_pdf_est = neighborDensityEstimator(trainArray, neighbors, testArray)
    normal_pdf_est = norm.pdf(testArray, mean_est, std_dev_est)
    kl_div_val = kl_divergence(knn_pdf_est, normal_pdf_est)
    print(f"KL Divergence for k = {neighbors}: {kl_div_val}")
    if kl_div_val < lowest_kl_divergence:
        lowest_kl_divergence = kl_div_val
        optimal_k = neighbors

print(f"Optimal k based on KL divergence: {optimal_k}")
# Plot histogram of training data
plt.hist(trainArray, bins=30, density=True, alpha=0.6, color='skyblue', label='Histogram')
plt.xlabel('Values')  # Adding labels for clarity
plt.ylabel('Density')
plt.legend()
plt.title('Training Data')
plt.show()
# Plot normal distribution PDF
normal_pdf = norm.pdf(testArray, mean_est, std_dev_est)
plt.plot(testArray, normal_pdf, label='Normal Distribution PDF', color='darkorange', linewidth=2)
plt.xlabel('Values')
plt.ylabel('Density')
plt.legend()
plt.title('Fitted Normal Distribution PDF')
plt.show()
# Plot KNN PDF with the best k
knn_pdf_best = neighborDensityEstimator(trainArray, optimal_k, testArray)
plt.plot(testArray, knn_pdf_best, label=f'KNN PDF (k={optimal_k})', color='green', linewidth=2)
plt.xlabel('Values')
plt.ylabel('Density')
plt.legend()
plt.title(f'KNN Density Estimate with k={optimal_k}')
plt.show()
# KDE with bandwidth selection
kde = KernelDensity(bandwidth=1.0)
kde.fit(trainArray.reshape(-1, 1))
log_dens = kde.score_samples(testArray.reshape(-1, 1))
kde_pdf = np.exp(log_dens)
plt.plot(testArray, kde_pdf, label='KDE PDF', color='purple', linewidth=2)
plt.xlabel('Values')
plt.ylabel('Density')
plt.legend()
plt.title('KDE PDF Estimate')
plt.show()
# Combined plot for comparison
plt.hist(trainArray, bins=30, density=True, alpha=0.6, color='lightcoral', label='Data Histogram')
plt.plot(testArray, normal_pdf, label='Normal PDF', color='steelblue', linewidth=2)
plt.plot(testArray, knn_pdf_best, label=f'KNN PDF (k={optimal_k})', color='darkgreen', linewidth=2)
plt.plot(testArray, kde_pdf, label='KDE PDF', color='gold', linewidth=2)

# Adjust x and y limits as needed
plt.xlim(0, 10)
plt.ylim(0, 0.8)

# Customizing labels and legend
plt.xlabel('Sample Values')
plt.ylabel('Estimated Density')
plt.legend(loc='upper right')  # Moving the legend to the upper right
plt.title('Comparison of Density Estimations: Data, Normal, KNN, KDE')
plt.show()
```
![image](https://github.com/user-attachments/assets/d0a5a9b6-1f0e-4a40-8fcd-3ccf8faf2bbb)
