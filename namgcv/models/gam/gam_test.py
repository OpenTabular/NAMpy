from GAM import GAM
import pandas as pd

def main():
    data = pd.read_csv("/home/michaelschlee/ownCloud/Dissertation/NAMgcv/namgcv/models/gam/syn_data_2_vars.csv")
    gam = GAM(data=data)

# Setting the seed for reproducibility
np.random.seed(123)

# Generating random normal variables
x = np.random.normal(size=100)
z = np.random.normal(size=100)

g = GAM()
g._setup(x, z, q=10)