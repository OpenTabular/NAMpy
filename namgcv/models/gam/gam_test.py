from GAM import GAM
import pandas as pd


data = pd.read_csv("/home/michaelschlee/ownCloud/Dissertation/NAMgcv/namgcv/models/gam/syn_data_2_vars.csv")
gam = GAM(data=data.iloc[:, :3])

gam.fit()