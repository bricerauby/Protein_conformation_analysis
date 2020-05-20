import numpy as np
import json

if __name__ == '__main__':
    densities = []
    for id in range(10):
        rmsd_path = 'data/test_1420738_rmsd_1000_neigh_id_{}_10.json'.format(id)
        with open(rmsd_path) as json_file:
            rmsd_file = json.load(json_file)

        keys, distances, neighbors = rmsd_file['rows'], rmsd_file['values'], rmsd_file['cols']
        keys, distances, neighbors = np.array(keys), np.array(distances), np.array(neighbors)

        if id==0:
            nb_neighbors = len(keys[keys==0])
        neighbors = neighbors.reshape(-1,nb_neighbors)
        distances = distances.reshape(-1,nb_neighbors)

        neighbors = np.concatenate([np.array([elt for elt in range(distances.shape[0])]).reshape(-1,1), neighbors], axis=1)
        distances = np.concatenate([np.array([0 for _ in range(distances.shape[0])]).reshape(-1,1), distances], axis=1)

        density = ((distances ** 2).mean(axis=-1) + 1e-10) ** -0.5

        densities += list(density)

        print('{}/10 finished'.format(id+1))

    densities = np.array(densities)
    np.save('data/densities',densities)
