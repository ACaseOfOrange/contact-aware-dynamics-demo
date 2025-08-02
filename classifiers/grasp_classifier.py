# classifiers/grasp_classifier.py
import numpy as np

class SimpleGraspClassifier:
    def predict(self, tactile_map):
        # Dummy logic: if average pressure > threshold, assume grasp
        return np.mean(tactile_map) > 0.5

