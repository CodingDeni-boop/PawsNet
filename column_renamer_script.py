import os
import pandas as pd

for label in os.listdir("./dataset/labels"):
    if not label.endswith(".csv"):
        continue
    path = os.path.join("./dataset/labels", label)
    print(path)
    df = pd.read_csv(path, index_col = 0)
    if len(df.columns) == 4:
        df.insert(4, "digging", 0)
    df.columns = ["background", "supportedrear", "unsupportedrear", "grooming", "digging"]
    df.to_csv(path)
