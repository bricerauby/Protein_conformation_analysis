import numpy as np
import json
import time
import gc
import argparse
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_points', default=1420738,
                        type=int, help='number of points to considered in case'
                        ' of a reduce dataset the distance on')
    parser.add_argument('--n_neigh', default=1000,
                        type=int, help='number of closest neighbor to save')
    parser.add_argument('--num_splits', default=20,
                        type=int, help='number of json files to merge')
    parser.add_argument('--output_dir', default='data',
                        type=str, help='output directory in which the '
                        'plots will be saved')
    args = parser.parse_args()
    densities = []
    neighbors = []
    nb_neighbors = args.n_neigh
    len_dataset = args.n_points
    neighbors = np.zeros((len_dataset, nb_neighbors), dtype=np.uint32)

    start_time = time.time()
    for id in range(args.num_splits):
        start_iteration = time.time()
        path = 'test_{}_rmsd_{}_neigh_id_{}_{}.json'.format(args.n_points,
                                                            args.n_neigh,
                                                            id,
                                                            args.num_splits)
        rmsd_path = os.path.join(args.output_dir, path)
        with open(rmsd_path) as json_file:
            rmsd_file = json.load(json_file)
        current_row = -1
        y = 0
        for current_index in range(len(rmsd_file['cols'])):

            x = rmsd_file['rows'][current_index]
            neigh = int(rmsd_file['cols'][current_index])
            if current_row == x - 1:
                y = 0
            elif current_row == x:
                y += 1
            neighbors[x, y] = neigh
            current_row = x

        distances = np.array(rmsd_file['values'])
        del rmsd_file
        gc.collect()

        distances = distances.reshape(-1, nb_neighbors)
        distances = np.concatenate([np.array(
            [0 for _ in range(distances.shape[0])]).reshape(-1, 1), distances],
            axis=1)
        density = ((distances ** 2).mean(axis=-1) + 1e-10) ** -0.5
        densities += list(density)

        print('time taken for iteration {}'.format(id),
              time.time() - start_iteration)
        np.save(os.path.join(args.output_dir, 'densities.npy'), densities)
        np.save(os.path.join(args.output_dir, 'neighbors.npy'), neighbors)
    print('time taken', time.time() - start_time)

    print("neighbors shape", neighbors.shape)
    print()
    print(neighbors[5, 60:70])
    print(neighbors[10000, 60:70])
    print(neighbors[500000, 60:70])
    print(neighbors[1000000, 60:70])
