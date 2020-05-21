import numpy as np
import matplotlib.pyplot as plt
from tomaster import tomato

if __name__=='main':
    filename = "data/dihedral.xyz"
    dihedral_coordinates = []
    xyz = open(filename)
    for line in xyz:
        x,y = line.split()
        dihedral_coordinates.append([float(x), float(y)])
    xyz.close()

    dihedral_coor_array = np.array(dihedral_coordinates)


    ### Visualize the original data up to a certain number of conformations
    conf_nb = 50000
    plt.scatter(dihedral_coor_array[:conf_nb,0],
                dihedral_coor_array[:conf_nb,1], s=0.5, alpha=0.5)
    plt.title('Conformations # 0 to {}'.format(conf_nb))
    plt.savefig('data/proj_data.png')


    ### Apply tomato to a part of the dataset
    limit = 500000
    clusters,_ = tomato(dihedral_coor_array[:limit], k=1000, n_clusters=7)
    plt.scatter(dihedral_coor_array[:limit,0],
                dihedral_coor_array[:limit,1], c=clusters, s=0.8, alpha=0.5)

    plt.savefig('data/tomato_proj_part_7_clusters.png')
