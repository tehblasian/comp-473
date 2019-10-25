import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as colormap
from mpl_toolkits import mplot3d

CLASS_2_SAMPLES = np.array([
    [-0.4, 0.58, 0.089],
    [-0.31, 0.27, -0.04],
    [0.38, 0.055, -0.035],
    [-0.15, 0.53, 0.011],
    [-0.35, 0.47, 0.034],
    [0.17, 0.69, 0.1],
    [-0.011, 0.55, -0.18],
    [-0.27, 0.61, 0.12],
    [-0.065, 0.49, 0.0012],
    [-0.12, 0.054, -0.063],
])

CLASS_3_SAMPLES = np.array([
    [0.83, 1.6, -0.014],
    [1.1, 1.6, 0.48],
    [-0.44, -0.41, 0.32],
    [0.047, -0.45, 1.4],
    [0.28, 0.35, 3.1],
    [-0.39, -0.48, 0.11],
    [0.34, -0.079, 0.14],
    [-0.3, -0.22, 2.2],
    [1.1, 1.2, -0.46],
    [0.18, -0.11, -0.49],
])

CLASS_SAMPLES = [CLASS_2_SAMPLES, CLASS_3_SAMPLES]

def lda():
    # Calculate d-dimensional mean vectors
    mean_vectors = calculate_mean_vectors(CLASS_SAMPLES)

    for i, mean_vector in enumerate(mean_vectors):
        print('Mean vector for class {}: {}'.format(i+1, mean_vector))

    print()

    # Compute the scatter matrices
    within_class_scatter_matrix = calculate_within_class_scatter_matrix(CLASS_SAMPLES, mean_vectors)
    print('Within-class scatter matrix:\n', within_class_scatter_matrix)
    print()

    between_class_scatter_matrix = calculate_between_class_scatter_matrix(CLASS_SAMPLES, mean_vectors)
    print('Between-class scatter matrix:\n', between_class_scatter_matrix)
    print()

    # Find the eigenvalues
    eigenvalues, eigenvectors = np.linalg.eig(np.linalg.inv(within_class_scatter_matrix).dot(between_class_scatter_matrix))
    for i in range(len(eigenvalues)):
        print('Eigenvalue {}: {}'.format(i+1, eigenvalues[i]))
        print('Corresponding eigenvector:\n{}'.format(eigenvectors[i].reshape(len(eigenvectors[i]), 1)))
        print()

    # Sort by largest eigenvalue
    eigen_values_vectors = sorted(list(zip(eigenvalues, eigenvectors)), key=lambda pair: np.abs(pair[0]), reverse=True)

    # Select linear discriminants
    # We have at most c - 1 useful eigenvalues
    linear_discriminants = [eigen_values_vectors[i][1] for i in range(len(CLASS_SAMPLES)-1)]
    print('The k most optimal directions are', linear_discriminants)
    print()

    # Compute W
    num_classes = len(CLASS_SAMPLES)
    W = np.empty((num_classes + 1, 1))

    print('The optimal direction W is: \n{}'.format(W))
    print()
    
    # Transform samples onto new subspace
    class_projections = []
    for class_sample in CLASS_SAMPLES:
        class_projections.append((class_sample.dot(W).tolist()))

    # Get the vectors
    vectors = []
    for cp in class_projections:
        v = []
        for proj in cp:
            v.append(W * proj)
        vectors.append(v)

    # Fit each distribution with a univariate Gaussian
    means, sigmas = fit_to_gaussian(class_projections)

    # Find the decision boundary
    decision_boundary = find_decision_boundary_gaussian(means, sigmas)

    # Transform decision boundary onto subspace
    decision_boundary_w = np.array(decision_boundary).dot(W).T

    # Calculate training error
    training_error = calculate_training_error(decision_boundary, class_projections[0], class_projections[1])
    print('Training error with optimal W: {}%'.format(round(training_error * 100, 2)))

    # Transform onto suboptimal subspace
    suboptimal_W = np.array([1, 2, -1.5]).T
    suboptimal_class_projections = []
    for class_sample in CLASS_SAMPLES:
        suboptimal_class_projections.append((class_sample.dot(suboptimal_W.tolist())))

    # Fit each distribution with a univariate Gaussian
    means, sigmas = fit_to_gaussian(class_projections)

    # Find the decision boundary
    suboptimal_decision_boundary = find_decision_boundary_gaussian(means, sigmas)

    # Transform decision boundary onto subspace
    suboptimal_decision_boundary_w = np.array(suboptimal_decision_boundary).dot(W).T

    # Calculate training error
    training_error = calculate_training_error(suboptimal_decision_boundary, suboptimal_class_projections[0], suboptimal_class_projections[1])
    print('Training error with suboptimal W: {}%'.format(round(training_error * 100, 2)))
    
    plot(CLASS_SAMPLES, W, np.array(vectors), decision_boundary_w)
    

def fit_to_gaussian(class_projections):
    means = calculate_mean_vectors([np.array(c) for c in class_projections])
    sigmas = [np.std(c, axis=0) for c in class_projections]
    return means, sigmas
    
def find_decision_boundary_gaussian(means, sigmas):
    mean1, mean2 = means[0], means[1]
    sigma1, sigma2 = sigmas[0], sigmas[1]

    a = 1 / (2 * sigma1**2) - 1 / ( 2 * sigma2**2)
    b = mean2 / (sigma2**2) - mean1 / (sigma1**2)
    c = mean1**2 / (2 * sigma1**2) - mean2**2 / (2 * sigma2**2) - np.log(sigma2 / sigma1)

    return max(np.roots([a.item(), b.item(), c.item()]), key=lambda v: abs(v))

def calculate_training_error(decision_boundary, w1_projections, w2_projections):
    total = len(w1_projections) + len(w2_projections)
    incorrect_classifications = 0
    for projection in w1_projections:
        if projection > decision_boundary:
            incorrect_classifications += 1

    for projection in w2_projections:
        if projection < decision_boundary:
            incorrect_classifications += 1

    return incorrect_classifications / total

def calculate_mean_vectors(class_samples):
    return [np.mean(sample, axis=0) for sample in class_samples]

def calculate_within_class_scatter_matrix(class_samples, mean_vectors):
    num_classes = len(class_samples)
    scatter_within = np.zeros((num_classes + 1, num_classes + 1))
    for w, mean_vector in zip(range(1, num_classes + 1), mean_vectors):
        class_scatter_matrix = np.zeros((num_classes + 1, num_classes + 1))
        for row in class_samples[w-1]:
            row, mean_vector = row.reshape(num_classes + 1, 1), mean_vector.reshape(num_classes + 1, 1)
            class_scatter_matrix += (row - mean_vector).dot((row - mean_vector).T)

        scatter_within += class_scatter_matrix

    return scatter_within

def calculate_between_class_scatter_matrix(class_samples, mean_vectors):
    num_classes = len(class_samples)
    total_mean = get_total_mean_all_samples(class_samples)
    scatter_between = np.zeros((num_classes + 1, num_classes + 1))
    for i, mean_vector in enumerate(mean_vectors):
        N = class_samples[i].size
        mean_vector = mean_vector.reshape(num_classes + 1, 1)
        total_mean = total_mean.reshape(num_classes + 1, 1)
        scatter_between += N * (mean_vector - total_mean).dot((mean_vector - total_mean).T)
        
    return scatter_between

def get_total_mean_all_samples(class_samples):
    all_samples = class_samples[0]
    for i in range(1, len(class_samples)):
        all_samples = np.append(all_samples, class_samples[i], axis=0)

    return np.mean(all_samples, axis=0)

def plot(class_samples, W, class_projections, decision_boundary):
    figure = plt.figure(figsize=(8, 6))
    axis = figure.add_subplot(111, projection='3d')

    axis.quiver(0, 0, 0, *W, color='black', arrow_length_ratio=0.03)

    colors = colormap.rainbow(np.linspace(0, 1, len(class_projections)))
    for i, cp in enumerate(class_projections):
        axis.scatter(cp[:,0], cp[:,1], cp[:,2], c=colors[i])

    axis.scatter(decision_boundary[:,0], decision_boundary[:,1], decision_boundary[:,2], color='green', s=50)
    
    plt.show()

np.set_printoptions(suppress=True)
lda()