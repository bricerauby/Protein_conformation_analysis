import numpy as np
import argparse
from IRMSD import Conformations
from IRMSD import align_array


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

    confs = align_array(coor_atom[:1000], 'atom')
    conf_obj = Conformations(confs, 'atom', 10)
    rmsds = conf_obj.rmsds_to_reference(conf_obj, 0)

    np.save('data/computed_rmsd.npy', rmsds)

    return


if __name__ == '__main__':
    main()
