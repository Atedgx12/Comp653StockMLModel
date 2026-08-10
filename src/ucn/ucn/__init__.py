"""
UCN modular package.
Import the main model with:  from ucn import UnifiedCourseNetwork, UCNConfig
"""
from .config import UCNConfig
from .models.unified_network import UnifiedCourseNetwork
from .models.logistic_regression import LogisticRegressionScratch
from .models.mlp import MLPScratch
from .models.naive_bayes import GaussianNaiveBayesScratch
from .training.metrics import accuracy, roc_auc
