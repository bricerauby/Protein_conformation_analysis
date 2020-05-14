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
    parser.add_argument('--num_workers', type=int, default=4,
                        help='dataloader threads')
    parser.add_argument('--seed', type=int,
                        default=np.random.randint(2**32 - 1),
                        help='the seed for reproducing experiments')
    parser.add_argument('--debug', action='store_true', default=False,
                        help='whether to debug or not')
    args = parser.parse_args(raw_args)
    err = 'You should provide a structure_file'
    assert args.structure_file is not None, err

    print("SEED used is ", args.seed)

    np.random.seed(args.seed)

    coordinates = []
    xyz = open(args.structure_file)
    for line in xyz:
        x, y, z = line.split()
        coordinates.append([float(x), float(y), float(z)])
    xyz.close()

    coor_atom = np.array(coordinates).reshape(-1, 10, 3)

    test_dim = 10000
    nb_kept_values = 100
    confs = align_array(coor_atom[:test_dim], 'atom')
    conf_obj = Conformations(confs, 'atom', 10)
    start = time.time()
    rows, cols = np.array([], dtype=int), np.array([], dtype=int)
    values = np.array([], dtype='float32')
    for ref_idx in range(test_dim):
        rmsd_to_ref = conf_obj.rmsds_to_reference(
            conf_obj, ref_idx).astype('float32')

        idx_to_keep = np.array([ref_idx]*nb_kept_values)
        cols_to_keep = np.argsort(rmsd_to_ref)[-nb_kept_values:]
        values_to_keep = rmsd_to_ref[cols_to_keep]

        rows = np.concatenate([rows,  idx_to_keep])
        cols = np.concatenate([cols, cols_to_keep])
        values = np.concatenate([values, values_to_keep])
    print('time taken', time.time() - start)
    rmsds = sps.coo_matrix((values, (rows, cols)))
    sps.save_npz('data/test_100_rmsd.npz', rmsds)
    # np.save('data/test_1000_rmsd.npy', rmsds)

    return


if __name__ == '__main__':
    main()
