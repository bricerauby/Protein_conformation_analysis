# Protein_conformation_analysis
Mode-seeking for detecting metastable states in protein conformations

### Installation
    First, install the following two dependencies
    - IRMSD (https://github.com/pandegroup/IRMSD)
    - GUHDI (https://gudhi.inria.fr/)
    Then you can install the dependencies with `pip -r requirements.txt`

Because the RMSD matrix library is only compatible with python 2 we are a virtual environemment.
### Compute RMSD matrix  
    First install :
    - IRMSD (https://github.com/pandegroup/IRMSD)
    Then install the other dependencies :
    ` pip -r requirements_2.txt`

    To run the computation of the rmsd matrix use :
    `python compute_rmsd.py --structure_file data/aladip_implicit.xyz --n_neigh 1000 --id_split 0 --num_workers 1`

    Then to merge the densities and the neighbors (that are too heavy to be built directly), you can run : `python merge_density_neighbors.py`. This will create the neighbors and densities matrix required by the tomato algorithm later on.

### Analysis

#### Get familiar with tomato parameters on a toy dataset
    In order to get a sense of the parameters and how the algorithm works, you can run the following commands from the main directory : `python toy_tomato.py`

#### Visualization of the conformations
    You can run `python aladip_visualization.py` to get some plots showing a few conformations and give a sense of each atom position spread.

#### Visualization of the 2D projected dataset
    A first approach consists of studying the 2-dimensional projected dataset that is much lighter. You can run `python dihedral_visualization.py` to get an original plot of the projected conformations in 2D and a clustering obtained by applying a mode-seeking algorithm (tomato).

#### Compute the persistence diagram
    In order to choose the proper number of clusters that the tomato algorithm should look for, you can run the code : `python plot_persistence_diagram.py --density_path data/densities.npy --neighbors_path data/neighbors.npy`

    Then, you can run `python compute_tomato.py --rmsd_path data/test_1420738_rmsd_1000_neigh_id_0_10.json --density_path data/densities.npy --neighbors_path data/neighbors.npy` to apply tomato to the densities and neighbors obtained earlier in the rmsd. The code then saves the plot in the proper output directory visualizing the points using the 2D projection with colors corresponding to the clusters obtained on the whole dataset using the RMSD metric.
