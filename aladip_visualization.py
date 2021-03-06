import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', default='data/aladip_implicit.xyz',
                        type=str, help='path to the toy dataset')
    parser.add_argument('--output_dir', default='data',
                        type=str, help='output directory in which the '
                        'plots will be saved')
    args = parser.parse_args()
    filename = args.input_file
    coordinates = []
    xyz = open(filename)
    for line in xyz:
        x, y, z = line.split()
        coordinates.append([float(x), float(y), float(z)])
    xyz.close()

    coor_array = np.array(coordinates).reshape(-1, 10, 3)
    # Visualize the first few conformations
    plt.figure(figsize=(18, 6))
    for conf_nb in range(9):
        plt.subplot(331+conf_nb)
        plt.scatter(coor_array[conf_nb, :, 0], coor_array[conf_nb, :, 1])
        plt.title('Conformation # : {}'.format(conf_nb))
        plt.axis('off')
    plt.savefig(os.path.join(args.output_dir, 'example_conformations.png'))
    # Visualize atom spread
    conf_nb = 1000
    cdict = {0: 'red', 1: 'green', 2: 'blue', 3: 'yellow', 4: 'c',
             5: 'm', 6: 'black', 7: 'gray', 8: 'orange', 9: 'pink'}
    fig, ax = plt.subplots(figsize=(8, 6))
    for col in range(10):
        ax.scatter(coor_array[:conf_nb, :, 0].reshape(-1)[col::10],
                   coor_array[:conf_nb, :, 1].reshape(-1)[col::10],
                   c=cdict[col], label='Atom # {}'.format(col), s=1.5, alpha=1.0)
    ax.legend()
    plt.axis('off')
    plt.savefig(os.path.join(args.output_dir, 'atom_spread.png'))

