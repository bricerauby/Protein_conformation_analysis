import numpy as np
import json
import time
if __name__ == '__main__':
    #densities = []
    neighbors = []
    nb_neighbors = 1000
    len_dataset = 1420738
    neighbors = np.zeros((len_dataset, nb_neighbors), dtype=np.float32)

    start_time = time.time()
    for id in range(10):
        rmsd_path = 'data/test_1420738_rmsd_1000_neigh_id_{}_10.json'.format(
            id)
        with open(rmsd_path) as json_file:
            rmsd_file = json.load(json_file)
        start_time = time.time()
        for current_index in range(len(rmsd_file['cols']):
            x = rmsd_file['rows'][current_index]
            y = rmsd_file['cols'][current_index]
            neighbors[x, y] = 1

        break 
        #keys, distances, neigh = rmsd_file['rows'], rmsd_file['values'], rmsd_file['cols']
        # #keys, distances, neigh = np.array(keys), np.array(distances), np.array(neigh)
        # neigh = np.array(rmsd_file['cols'])
        # del rmsd_file

        # if id == 0:
        #     nb_neighbors = len(keys[keys == 0])
        # #neigh = neigh.reshape(-1,nb_neighbors)
        # #distances = distances.reshape(-1,nb_neighbors)

        # #neighbors = np.concatenate([np.array([elt for elt in range(distances.shape[0])]).reshape(-1,1), neighbors], axis=1)
        # #distances = np.concatenate([np.array([0 for _ in range(distances.shape[0])]).reshape(-1,1), distances], axis=1)

        # #density = ((distances ** 2).mean(axis=-1) + 1e-10) ** -0.5

        # #densities += list(density)
        # neighbors += list(neigh)

        # print('{}/10 finished'.format(id+1))

    print('time taken', time.time() - start_time)

    #densities = np.array(densities)
    neighbors = neighbors.reshape(-1, nb_neighbors)
    # neighbors = np.concatenate([np.array([elt for elt in range(neighbors.shape[0])]).reshape(-1, 1),
    #                             neighbors], axis=1)

    # np.save('data/densities',densities)
    np.save('data/neighbors', neighbors)
