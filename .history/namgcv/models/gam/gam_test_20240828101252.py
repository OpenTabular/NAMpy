from GAM import GAM
import pandas as pd
from sys import platform



if platform == "linux" or platform == "linux2":
    f_path = "/home/michaelschlee/ownCloud/Dissertation/NAMgcv/namgcv/models/gam/syn_data_2_vars.csv"
elif platform == "win32":
    f_path = "C:\\Dissertation\\NAMgcv\\namgcv\\models\\gam\\syn_data_2_vars.csv"
data = pd.read_csv(f_path)
data_2 = pd.read_csv(f_path)

gam = GAM(data=data_2)

gam.fit()