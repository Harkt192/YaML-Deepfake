import os
import pandas as pd

# for path, dirs, files in os.walk("./data/dataset/test_images"):
#     print(len(files))

df = pd.read_csv(
    "./data/dataset/train_solution.csv",
    names=["id", "label"]
)
print(df.iloc[2171])
