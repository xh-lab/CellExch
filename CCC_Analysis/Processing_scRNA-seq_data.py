import pandas as pd
import numpy as np


single_cell = pd.read_csv("/home/jby2/XH/CellExch/CCC_Analysis/GSE72056.csv", header=None, index_col=None)

#cell_type = np.delete(single_cell[0], 0)
cell_type = single_cell.iloc[0, 1:]
cell_type = cell_type.to_numpy()
single_cell = single_cell[1:]
malignant = [i for i, x in enumerate(cell_type) if x == 0]
T = [i for i, x in enumerate(cell_type) if x == 1]
B = [i for i, x in enumerate(cell_type) if x == 2]
Mac = [i for i, x in enumerate(cell_type) if x == 3]
End = [i for i, x in enumerate(cell_type) if x == 4]
CAF = [i for i, x in enumerate(cell_type) if x == 5]
NK = [i for i, x in enumerate(cell_type) if x == 6]
gene = single_cell.iloc[:, 0]
gene_data = np.delete(single_cell.values, 0, axis=1)

malignant_means = np.mean(gene_data[:, malignant], axis=1)

T_means = np.mean(gene_data[:, T], axis=1)

B_means = np.mean(gene_data[:, B], axis=1)

Mac_means = np.mean(gene_data[:, Mac], axis=1)

End_means = np.mean(gene_data[:, End], axis=1)

CAF_means = np.mean(gene_data[:, CAF], axis=1)

NK_means = np.mean(gene_data[:, NK], axis=1)



value_pro = np.append(np.expand_dims(gene, axis=1), np.expand_dims(malignant_means, axis=1), axis=1)
value_pro = np.append(value_pro, np.expand_dims(T_means, axis=1), axis=1)
value_pro = np.append(value_pro, np.expand_dims(B_means, axis=1), axis=1)
value_pro = np.append(value_pro, np.expand_dims(Mac_means, axis=1), axis=1)
value_pro = np.append(value_pro, np.expand_dims(End_means, axis=1), axis=1)
value_pro = np.append(value_pro, np.expand_dims(CAF_means, axis=1), axis=1)
value_pro = np.append(value_pro, np.expand_dims(NK_means, axis=1), axis=1)  
value_pro = pd.DataFrame(value_pro)
value_pro.to_csv('/home/jby2/XH/CellExch/CCC_Analysis/The_expression_product_data.csv', index=False, header=False)
print("-----Product data processing completed----")


malignant_than_zero = (gene_data[:, malignant] > 0)
cell_malignant = np.sum(malignant_than_zero, axis=1)/gene_data[:, malignant].shape[1]
T_than_zero = (gene_data[:, T] > 0)
cell_T = np.sum(T_than_zero, axis=1)/gene_data[:, T].shape[1]
B_than_zero = (gene_data[:, B] > 0)
cell_B = np.sum(B_than_zero, axis=1)/gene_data[:, B].shape[1]
Mac_than_zero = (gene_data[:, Mac] > 0)
cell_Mac = np.sum(Mac_than_zero, axis=1)/gene_data[:, Mac].shape[1]
End_than_zero = (gene_data[:, End] > 0)
cell_End = np.sum(End_than_zero, axis=1)/gene_data[:, End].shape[1]
CAF_than_zero = (gene_data[:, CAF] > 0)
cell_CAF = np.sum(CAF_than_zero, axis=1)/gene_data[:, CAF].shape[1]
NK_than_zero = (gene_data[:, NK] > 0)
cell_NK = np.sum(NK_than_zero, axis=1)/gene_data[:, NK].shape[1]
value_cell = np.append(np.expand_dims(gene, axis=1), np.expand_dims(cell_malignant, axis=1), axis=1)
value_cell = np.append(value_cell, np.expand_dims(cell_T, axis=1), axis=1)
value_cell = np.append(value_cell, np.expand_dims(cell_B, axis=1), axis=1)
value_cell = np.append(value_cell, np.expand_dims(cell_Mac, axis=1), axis=1)
value_cell = np.append(value_cell, np.expand_dims(cell_End, axis=1), axis=1)
value_cell = np.append(value_cell, np.expand_dims(cell_CAF, axis=1), axis=1)
value_cell = np.append(value_cell, np.expand_dims(cell_NK, axis=1), axis=1)
value_cell = pd.DataFrame(value_cell)
value_cell.to_csv('/home/jby2/XH/CellExch/CCC_Analysis/The_cell_expression_data.csv', index=False, header=False)
print("-----Cell data processing completed----")

