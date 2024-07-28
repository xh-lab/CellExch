
# CellExch: A graph attention network-based method for analysing ligand-receptor interaction-mediated cell-cell communication
=========================================================================================


[![license](https://img.shields.io/badge/python_-3.11.7_-blue)](https://www.python.org/)
[![license](https://img.shields.io/badge/torch_-2.0.1_-orange)](https://pytorch.org/)
[![license](https://img.shields.io/badge/iFeature_-yellow)](https://github.com/Superzchen/iFeature/)


CellExch is a tool that first constructs an LRIs database by predicting LRIs and then performs cell-cell communication analysis based on the constructed database. Its flowchart is as follows.


![Image text](https://github.com/xh-lab/CellExch/blob/main/CellExch_workflow_00.png)


Workflow of the CellExch method. Part a includes Data Processing and Ligand-Receptor Interactions Prediction. The Data Processing part extracts the numerical features of ligand and receptor through iFeature to construct the feature matrix and reduces the dimension of the feature matrix through principal component analysis and organizes the ligand-receptor interactions into a matrix form. The Ligand-Receptor Interactions Prediction part uses these two matrices as the input of the graph attention network to learn the features of adjacent nodes, and uses the features extracted by the graph attention network as the input of the fully connected layer and performs a binary classification task. Part b includes Ligand-Receptor Interactions filtering and Cell-cell communication strength measurement. The Ligand-Receptor Interactions filtering part filters the LRIs identified by CellExch based on scRNA-seq data (i.e., if the ligand or receptor in a ligand receptor pair is not expressed in a certain cell, it is considered that the ligand receptor interaction will not mediate the corresponding cell-cell communication). The Cell-cell communication strength measurement part combines the scRNA-seq data with the filtered LRIs to calculate the cell result and product result, and the normalized average of the two results is used as the cell-cell communication strength. Part c includes Cell-cell communication visualization, which visualizes the results obtained in part b.

## Table of Contents

- [Installation](#installation)
- [Quick start](#quick-start)
- [Contributing](#contributing)
- [Cite](#cite)
- [Contacts](#contacts)
- [License](#license)


## Installation
CellExch is tested to work under:

```
* Anaconda 24.1.2
* Python 3.11.7
* Torch 2.0.1
* Numpy 1.24.4
* iFeature
* Other basic python toolkits
```
### Installation of other dependencies
* Install [torch_geometric](https://pytorch-geometric.readthedocs.io/en/latest/) using pip install torch_geometric if you encounter any issue.
* Install [seaborn-0.13.2](https://seaborn.pydata.org/) using pip install seaborn if you encounter any issue.
* Install [networkx-2.8.8](https://pypi.org/project/networkx/2.8.8/) using pip install networkx if you encounter any issue.


# Quick start
To reproduce our results:

**Notes**: Due to the large size of some datas, we uploaded them to the Google Drive, if some files cannot be found, please look for them [here](https://drive.google.com/drive/folders/1krX0ulqYsRF_xtDRIgJ0FZXVIP1L1FI-?usp=drive_link). 

## Data Description
| **File name** | **Description** |
| :---: | :--- |
| LRI.csv | The LR pairs identified by CellMsg. | 
| mart_export.txt | Mapping files of protein identifiers to gene names. |
| ligand_sequence.txt and receptor_sequence.txt | ligand and receptor sequence files, they serve as input files for iFeature to generate corresponding ligand or receptor features. |
| ligand_res_fea.csv and receptor_res_fea.csv (stored in google drive) | ligand feature and receptor feature files obtained after processing with iFeature. | 
| ligand-receptor-interaction.csv | This file contains information about ligand-receptor interactions that we collected. |
| final_model.pth (stored in google drive) | The final model for predicting LRIs. |
| LRI_predicted.csv | LRIs that predicted by CellMsg. |
| original_LRI.csv | LRIs that we collected. |

## 1, acquiring feature file from sequence file using iFeature
**Notes**: Since the processing steps for all sequence files are identical, we will proceed to process one of the sequence files.
```
python iFeature.py --file CellMsg/dataset1/ligand_sequence.txt --type AAC --out ligand_aac.csv
python iFeature.py --file CellMsg/dataset1/ligand_sequence.txt --type CKSAAP --out ligand_cksaap.csv
python iFeature.py --file CellMsg/dataset1/ligand_sequence.txt --type CTriad --out ligand_ctriad.csv
python iFeature.py --file CellMsg/dataset1/ligand_sequence.txt --type PAAC --out ligand_paac.csv
Then, the four features were merged to generate the final feature file for all ligands, where each row represents the features of one ligand, with the number of rows equating to the number of ligands.
```

## 2, training an LRI prediction model
**Notes**: The steps for training the LRI prediction model are the same for all datasets. Just change the dataset in the CellExch.py ​​file to the one you need.
```
python CellExch.py
```
## 3, preditcing LRIs using trained model
```
python generate_lrp.py
python ensp_to_gname.py
```
Through the above steps, we obtained predicted LRIs with high confidence, which are then merged with the LRIs previously collected to serve as LRIs identified by CellExch.

## 4, measuring cell-Cell communication strength
```
python CCC_Analysis/Processing_scRNA-seq_data.py
python CCC_Analysis/ccc_strength_mea.py
```
Through the above steps, we obtained the cell-cell communication strength matrix, and we generated the cell communication heatmap and cell communication network.

## 5, visualization analysis of cell-cell communication
```
python CCC_Analysis/The_num_of_LRIs.py
```
Through the steps outlined above, we have obtained LRi_num.pdf and LRi_num.csv, which show the number of LRIs mediating communication between cell types in human melanoma tissues.

```
python CCC_Analysis/Top.py
```
Through the above command, we obtained Top.pdf and Top_data.csv, which display the five most likely LR pairs mediating communication between melanoma cancer cells and six other cell types.

=========================================================================================



# Contributing 
All authors were involved in the conceptualization of the CellExch method. BYJ and SLP conceived and supervised the project. BYJ and HX collected the data. HX completed the coding for the project. BYJ, HX and SLP contributed to the review of the manuscript before submission for publication. All authors read and approved the final manuscript.

# cite
<p align="center">
  <a href="https://clustrmaps.com/site/1c0ne" title="Visit tracker">
     <img src="//www.clustrmaps.com/map_v2.png?d=Uf341eT4GWYarTxGkL949w2hBYAIbAVB9oHe0UjCFC8&cl=ffffff">
   </a>
</p>

<p align="center">
  <a href="#">
     <img src="https://api.visitorbadge.io/api/visitors?path=https%3A%2F%2Fgithub.com%2Fxh-lab%2FCellExch&countColor=%23263759" />
   </a>
</p>


# Contacts
If you have any questions or comments, please feel free to email: Shaoliang Peng (slpeng@hnu.edu.cn); (Boya Ji) byj@hnu.edu.cn; (Hong Xia) hongxia@hnu.edu.cn.

# License

[MIT &copy; Richard McRichface.](../LICENSE)
