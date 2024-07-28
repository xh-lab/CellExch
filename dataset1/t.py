import pandas as pd

d1 = pd.read_csv("/home/jby2/XH/CellExch/dataset1/generation/original_LRI.csv", index_col=None, header=None)
d2 = pd.read_csv("/home/jby2/XH/CellExch/dataset1/generation/LRI_predicted.csv", index_col=None, header=None, skiprows=1)

df = pd.concat([d1, d2], axis=0, ignore_index=True)
df.to_csv("/home/jby2/XH/CellExch/dataset1/generation/LRI.csv", header=None, index=False)