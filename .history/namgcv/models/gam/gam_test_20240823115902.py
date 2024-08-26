from GAM import GAM
import numpy as np

# Setting the seed for reproducibility
np.random.seed(123)

# Generating random normal variables
x = np.random.randn(100)
z = np.random.randn(100)
g = GAM()
g._setup(x, z, q=10)