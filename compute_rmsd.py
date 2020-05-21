import numpy as np
import argparse
from IRMSD import Conformations
from IRMSD import align_array
import time
import ujson as json


class RmsdCalculator(object):
    def __init__(self, conf_obj, nb_kept_values):
        self.conf_obj = conf_obj
        self.nb_kept_values = nb_kept_values

    def compute_rmsd_mat(self, ref_idx):
        rmsd_to_ref = self.conf_obj.rmsds_to_reference(self.conf_obj,
                                                       ref_idx)
        assert str(rmsd_to_ref.dtype) == 'float32'
        idx_to_keep = [ref_idx] * self.nb_kept_values
        cols_to_keep = np.argsort(rmsd_to_ref)[1:self.nb_kept_values + 1]
        values_to_keep = rmsd_to_ref[cols_to_keep]
        return idx_to_keep, cols_to_keep, values_to_keep

    def compute_rmsd_mat_list(self, ref_idxs):
        values = []
        rows, cols = [], []
        print(ref_idxs[0], ref_idxs[-1])
        start_time = time.time()
        for ref_idx in ref_idxs:
            res = self.compute_rmsd_mat(ref_idx)
            idx_to_keep, cols_to_keep, values_to_keep = res
            rows += idx_to_keep
            values += values_to_keep.tolist()
            cols += cols_to_keep.tolist()
        print(time.time() - start_time)
        return values, rows, cols


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
    parser.add_argument('--id_worker', default=0,
                        type=int, help='number of workers')
    parser.add_argument('--num_workers', default=0,
                        type=int, help='number of workers')
    args = parser.parse_args(raw_args)
    err = 'You should provide a structure_file'
    assert args.structure_file is not None, err

    try:
        coor_atom = np.load('data/coor_atom.npy')
    except IOError:
        coordinates = []
        xyz = open(args.structure_file)
        for line in xyz:
            x, y, z = line.split()
            coordinates.append([float(x), float(y), float(z)])
        xyz.close()

        coor_atom = np.array(coordinates).reshape(-1, 10, 3)
        np.save('data/coor_atom.npy', coor_atom)

    if args.n_points is None:
        test_dim = len(coor_atom)
    else:
        test_dim = args.n_points
    nb_kept_values = args.n_neigh
    confs = align_array(coor_atom[:test_dim], 'atom')
    conf_obj = Conformations(confs, 'atom', 10)
    start_time = time.time()

    rmsd_calculator = RmsdCalculator(conf_obj, nb_kept_values)
    all_index = list(range(test_dim))
    n_per_worker = test_dim // args.num_workers
    start = args.id_worker * n_per_worker
    end = min(test_dim, (args.id_worker + 1) * n_per_worker)
    res = rmsd_calculator.compute_rmsd_mat_list(all_index[start:end])
    values, rows, cols = res

    print('time taken for rmsd computation', time.time() - start_time)

    start_time = time.time()
    dic = {}
    dic['values'] = values
    dic['rows'] = rows
    dic['cols'] = cols
    path = 'data/test_{}_rmsd_{}_neigh_id_{}_{}.json'.format(test_dim,
                                                             nb_kept_values,
                                                             args.id_worker,
                                                             args.num_workers)
    with open(path, 'wb') as f:
        json.dump(dic, f)
    print('time taken to save json ',
          time.time() - start_time)
    return


if __name__ == '__main__':
    print('python script has started')
    main()
