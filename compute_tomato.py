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
    parser.add_argument('--k', default=None,
                        type=int, help='number of points in the neighborhood')

    args = parser.parse_args(raw_args)

    # Load the projected dataset to plot the cluster results and visualize them easily
    filename = "data/dihedral.xyz"
    coordinates = []
    xyz = open(filename)
    for line in xyz:
        x,y = line.split()
        coordinates.append([float(x), float(y)])
    xyz.close()
    dihedral_coor_array = np.array(coordinates)

    start = time.time()
    clusters,_ = tomato(points=None, k=args.k, n_clusters=7,
                        rmsd_path=args.rmsd_path, density_path=args.density_path)
    end = time.time()
    print('Tomato computation took {} seconds'.format(end-start))
    plt.scatter(dihedral_coor_array[:clusters.shape[0],0], dihedral_coor_array[:clusters.shape[0],1], c=clusters,
                s=0.5, alpha=0.5)
    name = args.rmsd_path.split('/')[-1].split('.')[0]
    plt.savefig('data/tomato_on_{}.png'.format(name))
    plt.show()
    np.save('data/clusters_tomato_on_{}'.format(name), clusters)

if __name__ == '__main__':
    print('python script has started')
    main()
