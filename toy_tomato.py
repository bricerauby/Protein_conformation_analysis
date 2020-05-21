import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tomaster import tomato

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file',
                        default='data/ToMATo/inputs/spiral_w_density.txt',
                        type=str, help='path to the toy dataset')
    parser.add_argument('--output_dir', default='data',
                        type=str, help='output directory'
                        ' in which the plots will be saved')
    args = parser.parse_args()
    filename = args.input_file
    spi_dens_cors = []
    xyz = open(filename)
    for line in xyz:
        x, y, z = line.split()
        spi_dens_cors.append([float(x), float(y), float(z)])
    xyz.close()
    spi_dens_cors_arr = np.array(spi_dens_cors)

    # Visualize the original spiral data
    plt.figure(figsize=(12, 6))
    plt.scatter(spi_dens_cors_arr[:, 0],
                spi_dens_cors_arr[:, 1], s=0.1, alpha=0.2)
    plt.savefig(os.path.join(args.output_dir, 'toy_original.png'))

    # Play with the parameter k
    plt.figure(figsize=(12, 6))
    for rank in range(2):
        clusters, _ = tomato(spi_dens_cors_arr, k=5+rank, n_clusters=2)

        plt.subplot(121+rank)
        plt.scatter(spi_dens_cors_arr.T[0], spi_dens_cors_arr.T[1],
                    c=clusters, s=0.1, alpha=0.2)
        plt.title("Tomato with k={}".format(5+rank))

    plt.savefig(os.path.join(args.output_dir, 'toy_param_k.png'))
    # Play with the parameter n_clusters
    plt.figure(figsize=(15, 5))
    for rank in range(3):
        clusters, _ = tomato(spi_dens_cors_arr, k=6, n_clusters=2+rank)

        plt.subplot(131+rank)
        plt.scatter(spi_dens_cors_arr.T[0], spi_dens_cors_arr.T[1],
                    c=clusters, s=0.1, alpha=0.2)
        plt.title("Tomato with n_clusters={}".format(2+rank))

    plt.savefig(os.path.join(args.output_dir, 'toy_param_n_clusters.png'))
    # Play with the parameter tau
    plt.figure(figsize=(15, 5))
    for rank in range(3):
        clusters = tomato(spi_dens_cors_arr, k=6, tau=rank+5)

        plt.subplot(131+rank)
        plt.scatter(spi_dens_cors_arr.T[0], spi_dens_cors_arr.T[1], c=clusters,
                    s=0.1, alpha=0.2)
        plt.title("Tomato with tau={}".format(rank+5))

    plt.savefig(os.path.join(args.output_dir, 'toy_param_tau.png'))
