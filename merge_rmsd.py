import argparse
import time
import ujson as json


def main(raw_args=None):

    parser = argparse.ArgumentParser()
    parser.add_argument('--n_points', default=None,
                        type=int, help='number_of_points')
    parser.add_argument('--n_neigh', default=1000,
                        type=int, help='number_of_neighbor')
    parser.add_argument('--num_workers', default=0,
                        type=int, help='number of workers')
    args = parser.parse_args(raw_args)

    n_workers = args.num_workers
    if args.n_points is None:
        test_dim = 'all'
    else:
        test_dim = args.n_points
    nb_kept_vals = args.n_neigh

    values, rows, cols = [], [], []
    start_time = time.time()
    for id_worker in range(n_workers):
        path = 'data/test_{}_rmsd_{}_neigh_id_{}_{}.json'.format(test_dim,
                                                                 nb_kept_vals,
                                                                 id_worker,
                                                                 n_workers)
        with open(path, 'r') as json_file:
            dic = json.load(json_file)

        values += dic["values"]
        rows += dic["rows"]
        cols += dic["cols"]

    print(len(values))
    print('time for merging', time.time()-start_time)

    start_time = time.time()
    dic["values"] = values
    dic["rows"] = rows
    dic["cols"] = cols

    path = 'data/test_{}_rmsd_{}_neigh_merged.json'.format(test_dim,
                                                           nb_kept_vals)
    with open(path, 'wb') as f:
        json.dump(dic, f)

    print('time for saving', time.time()-start_time)


if __name__ == '__main__':
    print('python script has started')
    main()
