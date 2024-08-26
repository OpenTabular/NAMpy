from GAM import GAM
import pandas as pd

def main():
    data = pd.read_csv("/home/michaelschlee/ownCloud/Dissertation/NAMgcv/namgcv/models/gam/syn_data_2_vars.csv")
    gam = GAM(data=data)

if __name__ == "__main__":
    main()

