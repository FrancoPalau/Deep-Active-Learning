import pandas as pd
import numpy as np
import json
import requests

def clase(domain_name, option):
    key = "error"
    url = "http://10.0.0.37:8014/predict?domain=" + domain_name
    print(url)
    response = requests.get(url)
    if key not in response.json():
        if option == "c":
            return response.json()["class"]
        elif option == "p":
            return response.json()["probability"]
    else:
        return np.nan


# Read dataset
df_combined_test = pd.read_csv("sample_test_multiclass.csv")

# Get resposes
df_combined_test["pred_class"]= df_combined_test["domain"].apply(clase, option="c")
df_combined_test["prob"]= df_combined_test["domain"].apply(clase, option="p")

# Save into csv file
df_combined_test.to_csv("results_sample_test_multiclass.csv", index=False)
