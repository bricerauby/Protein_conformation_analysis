import numpy as np
import time
import argparse
import matplotlib.pyplot as plt
from tomaster import tomato

def main(raw_args=None):

    parser = argparse.ArgumentParser()
    parser.add_argument('--rmsd_path', default='',
                        type=str, help='path to the rmsd file')
    parser.add_argument('--k', default=None,
                        type=int, help='number of points in the neighborhood')

    args = parser.parse_args(raw_args)

    filename = "data/dihedral.xyz"
    coordinates = []
    xyz = open(filename)
    for line in xyz:
        x,y = line.split()
        coordinates.append([float(x), float(y)])
    xyz.close()
    dihedral_coor_array = np.array(coordinates)

    start = time.time()
    clusters,_ = tomato(points=None, k=args.k, n_clusters=7, rmsd_path=args.rmsd_path)
    end = time.time()
    print('Tomato computation took {} seconds'.format(end-start))
    plt.scatter(dihedral_coor_array[:,0], dihedral_coor_array[:,1], c=clusters,
                s=0.5, alpha=0.5)
    plt.show()

if __name__ == '__main__':
    print('python script has started')
    main()
