import numpy as np
import json
import time
import gc

if __name__ == '__main__':
    densities = []
    neighbors = []
    nb_neighbors = 1000
    len_dataset = 1420738
    neighbors = np.zeros((len_dataset, nb_neighbors), dtype=np.uint32)

    start_time = time.time()
    for id in range(20):
        start_iteration = time.time()
        rmsd_path = 'data/test_1420738_rmsd_1000_neigh_id_{}_20.json'.format(
            id)
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
            [0 for _ in range(distances.shape[0])]).reshape(-1, 1), distances], axis=1)
        density = ((distances ** 2).mean(axis=-1) + 1e-10) ** -0.5
        densities += list(density)

        print('time taken for iteration {}'.format(id),
              time.time() - start_iteration)
        np.save('data/densities.npy', densities)
        np.save('data/neighbors.npy', neighbors)
    print('time taken', time.time() - start_time)

    print("neighbors shape", neighbors.shape)
    print()
    print(neighbors[5, 60:70])
    print(neighbors[10000, 60:70])
    print(neighbors[500000, 60:70])
    print(neighbors[1000000, 60:70])
