import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tomaster import tomato

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', default='data/dihedral.xyz',
                        type=str, help='path to the toy dataset')
    parser.add_argument('--output_dir', default='data',
                        type=str, help='output directory in which the plots will be saved')
    parser.add_argument('--k', default=1000,
                        type=int, help='number of points in the neighborhood')
    parser.add_argument('--n_clusters', default=7,
                        type=int, help='number of clusters to look for')
    parser.add_argument('--conf_nb', default=50000,
                        type=int, help='number of conformations to visualize')
    parser.add_argument('--tomato_limit', default=500000,
                        type=int, help='number of conformations to consider in tomato')
    args = parser.parse_args()
    filename = args.input_file
    dihedral_coordinates = []
    xyz = open(filename)
    for line in xyz:
        x,y = line.split()
        dihedral_coordinates.append([float(x), float(y)])
    xyz.close()

    dihedral_coor_array = np.array(dihedral_coordinates)


    ### Visualize the original data up to a certain number of conformations
    conf_nb = args.conf_nb
    plt.scatter(dihedral_coor_array[:conf_nb,0],
                dihedral_coor_array[:conf_nb,1], s=0.5, alpha=0.5)
    plt.title('Conformations # 0 to {}'.format(conf_nb))
    plt.savefig(os.path.join(args.output_dir,'proj_data.png'))


    ### Apply tomato to a part of the dataset
    limit = args.tomato_limit
    clusters,_ = tomato(dihedral_coor_array[:limit], k=args.k, n_clusters=args.n_clusters)
    plt.scatter(dihedral_coor_array[:limit,0],
                dihedral_coor_array[:limit,1], c=clusters, s=0.8, alpha=0.5)

    plt.savefig(os.path.join(args.output_dir,'tomato_proj_part_{}_clusters.png'.format(args.n_clusters)))
