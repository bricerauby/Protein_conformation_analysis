import numpy as np
import time
import argparse
import matplotlib.pyplot as plt
from tomaster import tomato

def main(raw_args=None):

    parser = argparse.ArgumentParser()
    parser.add_argument('--rmsd_path', default='',
                        type=str, help='path to the rmsd file')
    parser.add_argument('--density_path', default='',
                        type=str, help='path to the density file directly')
    parser.add_argument('--neighbors_path', default='',
                        type=str, help='path to the neighbors file directly')
    parser.add_argument('--k', default=None,
                        type=int, help='number of points in the neighborhood')
    parser.add_argument('--n_clusters', default=7,
                        type=int, help='number of clusters to look for')
    parser.add_argument('--input_file', default='data/dihedral.xyz',
                        type=str, help='path to the 2D dataset')
    parser.add_argument('--output_dir', default='data',
                        type=str, help='output directory in which the plots will be saved')

    args = parser.parse_args(raw_args)

    # Load the projected dataset to plot the cluster results and visualize them easily
    filename = args.input_file
    coordinates = []
    xyz = open(filename)
    for line in xyz:
        x,y = line.split()
        coordinates.append([float(x), float(y)])
    xyz.close()
    dihedral_coor_array = np.array(coordinates)

    start = time.time()
    clusters, _ = tomato(points=None, k=args.k, n_clusters=args.n_clusters,
                         rmsd_path=args.rmsd_path,
                         density_path=args.density_path,
                         neighbors_path=args.neighbors_path)
    end = time.time()
    print('Tomato computation took {} seconds'.format(end-start))
    plt.scatter(dihedral_coor_array[:clusters.shape[0],0], dihedral_coor_array[:clusters.shape[0],1], c=clusters,
                s=0.5, alpha=0.5)
    name = args.rmsd_path.split('/')[-1].split('.')[0]
    plt.savefig(os.path.join(args.output_dir,'tomato_on_{}.png'.format(name)))

    np.save(os.path.join(args.output_dir,'clusters_tomato_on_{}'.format(name), clusters))

if __name__ == '__main__':
    print('python script has started')
    main()
