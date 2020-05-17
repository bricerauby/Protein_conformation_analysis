import numpy as np
import scipy.sparse as sps
import argparse
from IRMSD import Conformations
from IRMSD import align_array
import time


def main(raw_args=None):
    """
    Main function to compute the rmsd matrix. Can either take the raw_args
    in argument or get the arguments from the config_file.
    """

    # -----------------------------------------------------------------------------------------
    # First, set the parameters of the function, including the
    #  config file, log directory and the seed.
    parser = argparse.ArgumentParser()
    parser.add_argument('--structure_file', default=None,
                        type=str, help='path to the structure file')
    parser.add_argument('--n_points', default=None,
                        type=int, help='number_of_points')
    parser.add_argument('--n_neigh', default=1000,
                        type=int, help='number_of_neighbor')
    args = parser.parse_args(raw_args)
    err = 'You should provide a structure_file'
    assert args.structure_file is not None, err

    coordinates = []
    xyz = open(args.structure_file)
    for line in xyz:
        x, y, z = line.split()
        coordinates.append([float(x), float(y), float(z)])
    xyz.close()

    coor_atom = np.array(coordinates).reshape(-1, 10, 3)
    if args.n_points is None:
        test_dim = len(coor_atom)
    else:
        test_dim = args.n_points

    nb_kept_values = args.n_neigh
    confs = align_array(coor_atom[:test_dim], 'atom')
    conf_obj = Conformations(confs, 'atom', 10)
    start = time.time()
    rows, cols = [], []
    values = []
    for ref_idx in range(test_dim):
        if ref_idx % 100 == 0:
            print(ref_idx)
        rmsd_to_ref = conf_obj.rmsds_to_reference(
            conf_obj, ref_idx).astype('float32')

        idx_to_keep = np.array([ref_idx]*nb_kept_values)
        cols_to_keep = np.argsort(rmsd_to_ref)[:nb_kept_values]
        values_to_keep = rmsd_to_ref[cols_to_keep]

        rows += idx_to_keep
        values += values_to_keep.tolist()
        cols += cols_to_keep.tolist()

    print('time taken', time.time() - start)
    rmsds = sps.coo_matrix((values, (rows, cols)))
    path = 'data/test_{}_rmsd_{}_neighbors.npz'.format(test_dim,
                                                       nb_kept_values)
    sps.save_npz(path, rmsds)

    return


if __name__ == '__main__':
    print('python script has started')
    main()
