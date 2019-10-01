from math import sqrt
from typing import List
import numpy as np

class Class:
    def __init__(self, prior: float, data: List[List[float]]):
        self.prior = prior
        self.data = np.array(data)
        self.mean_vector = np.mean(self.data, axis=0)
        self.covariance_matrix = np.cov(self.data, rowvar=False)

class Discriminant:
    def __init__(self, clazz):
        self.class_prior = clazz.prior
        self.mean_vector = clazz.mean_vector
        self.covariance_matrix = clazz.covariance_matrix

    def calculate_cost(self, feature_vector: List[float]) -> float:
        x = np.array(feature_vector)
        inv_covariance_matrix = self._calculate_covariance_matrix_inv(self.covariance_matrix)
        det_covariance_matrix = self._calculate_covariance_matrix_det(self.covariance_matrix)
        
        cost = (
            -0.5 * np.dot(
                np.dot(
                    np.transpose((x - self.mean_vector)), 
                    inv_covariance_matrix
                ),
                (x - self.mean_vector) 
            )
            -0.5 * np.emath.log(det_covariance_matrix)
            + np.emath.log(self.class_prior))
        return cost

    def _calculate_covariance_matrix_inv(self, covariance_matrix):
        if covariance_matrix.ndim >= 2:
            return np.linalg.inv(covariance_matrix)
        return 1 / covariance_matrix

    def _calculate_covariance_matrix_det(self, covariance_matrix):
        if covariance_matrix.ndim >= 2:
            return np.linalg.det(covariance_matrix)
        return covariance_matrix

class Bhattacharyya:
    def __init__(self, class1, class2):
        self.class1 = class1
        self.class2 = class2

    def _exponent(self):
        combined_covariances_matrix = 0.5 * (self.class1.covariance_matrix + self.class2.covariance_matrix)

        inv_combined_covariances_matrix = self._calculate_combined_covariances_matrix_inv(combined_covariances_matrix)
        det_combined_covariances_matrix = self._calculate_combined_covariances_matrix_det(combined_covariances_matrix)

        det_class1_covariance_matrix = self._calculate_covariance_matrix_det(self.class1.covariance_matrix)
        det_class2_covariance_matrix = self._calculate_covariance_matrix_det(self.class2.covariance_matrix)

        return (
            -1/8 * np.dot(
                np.dot(
                    np.transpose(self.class2.mean_vector - self.class1.mean_vector),
                    inv_combined_covariances_matrix
                ),
                (self.class2.mean_vector - self.class1.mean_vector)
            )
            + 0.5 * np.math.log(
                det_combined_covariances_matrix / sqrt(det_class1_covariance_matrix * det_class2_covariance_matrix)
            )
        )

    def _calculate_combined_covariances_matrix_inv(self, combined_covariances_matrix):
        if combined_covariances_matrix.ndim >= 2:
            return np.linalg.inv(combined_covariances_matrix)
        return 1 / combined_covariances_matrix

    def _calculate_combined_covariances_matrix_det(self, combined_covariances_matrix):
        if combined_covariances_matrix.ndim >= 2:
            return np.linalg.det(combined_covariances_matrix)
        return combined_covariances_matrix

    def _calculate_covariance_matrix_det(self, covariance_matrix):
        if covariance_matrix.ndim >= 2:
            return np.linalg.det(covariance_matrix)
        return covariance_matrix

    def error_bound(self):
        return sqrt(self.class1.prior * self.class2.prior) * pow(np.math.e, self._exponent())

def classify_only_x1(w1_data, w2_data):
    w1_data, w2_data = [[feature[0]] for feature in w1_data], [[feature[0]] for feature in w2_data]
    classify(w1_data, w2_data)

def classify_only_x1x2(w1_data, w2_data):
    w1_data, w2_data = [[feature[0], feature[1]] for feature in w1_data], [[feature[0], feature[1]] for feature in w2_data]
    classify(w1_data, w2_data)

def classify_all_features(w1_data, w2_data):
    classify(w1_data, w2_data)

def classify(w1_data, w2_data):
    w1, w2 = Class(0.5, w1_data), Class(0.5, w2_data)
    g1, g2 = Discriminant(w1), Discriminant(w2)

    total_w1_samples, correctly_classified_w1 = len(w1_data), 0
    for sample in w1_data:
        cost = g1.calculate_cost(sample) - g2.calculate_cost(sample)
        if cost > 0:
            correctly_classified_w1 += 1

    total_w2_samples, correctly_classified_w2 = len(w2_data), 0
    for sample in w2_data:
        cost = g1.calculate_cost(sample) - g2.calculate_cost(sample)
        if cost < 0:
            correctly_classified_w2 += 1

    training_error = round((1 - (correctly_classified_w1 + correctly_classified_w2) / (total_w1_samples + total_w2_samples)) * 100, 2)

    print('  Correctly classified {} instances of w1'.format(correctly_classified_w1))
    print('  Correctly classified {} instances of w2'.format(correctly_classified_w2))
    print('  Training error: {} %'.format(training_error))

    # Get Bhattacharrya bound
    bound = Bhattacharyya(w1, w2)
    print('  Bhattacharyya bound: P(error) <= {}'.format(bound.error_bound()))

def main():
    # Data set
    w1_data = [[-5.01, -8.12, -3.68],
        [-5.43, -3.48, -3.54],
        [1.08, -5.52, 1.66],
        [0.86, -3.78, -4.11],
        [-2.67, 0.63, 7.39],
        [4.94,3.29, 2.08],
        [-2.51, 2.09, -2.59],
        [-2.25, -2.13, -6.94],
        [5.56, 2.86, -2.26],
        [1.03, -3.33, 4.33]]

    w2_data = [[-0.91, -0.18, -0.05],
        [-1.30, -2.06, -3.53],
        [-7.75, -4.54, -0.95],
        [-5.47, 0.50, 3.92],
        [6.14, 5.72, -4.85],
        [3.60, 1.26, 4.36],
        [5.37, -4.63, -3.65],
        [7.18, 1.46, -6.66],
        [-7.39, 1.17, -6.30],
        [-7.50, -6.32, -0.31]]

    print('Classifying using only x1:')
    classify_only_x1(w1_data, w2_data)
    print()

    print('Classifying using only x1 and x2:')
    classify_only_x1x2(w1_data, w2_data)
    print()

    print('Classifying using x1, x2 and x3:')
    classify_all_features(w1_data, w2_data)
    print()

main()
