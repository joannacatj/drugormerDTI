# Readme

DrugormerDTI: Drug Graphormer for drug–target interaction prediction 

use
1.Run pre_ data.py  to preprocess the dataset and generate atom_feat.npy , proteins.npy, bond_adj.npy, dist_adj.npy,compound.npy document. Please change 'dir_input'  to change the address for saving files.

2.Run main_Davis2.py to train the model and get the test metrics.

3.Modules.py contains the main part of our model,including protein encoder module,drug encoder module and decoder model.

4.model.py contains the train method and test method of our model
