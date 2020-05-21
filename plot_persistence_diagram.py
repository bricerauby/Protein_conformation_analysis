from scipy.stats import gaussian_kde as kde
import matplotlib.pyplot as plt
import matplotlib.gridspec as gds
import gudhi
from sklearn.decomposition import PCA
from sklearn.neighbors import KDTree

def plot_density(X, reduced, nbins, den, vec):
    x,y = reduced.T

    u,v = np.mgrid[x.min():x.max():nbins*1j, y.min():y.max():nbins*1j]
    # In if original data dimention > 2, the vizualization will show the
    # density over the reduced (2D) representation of the data calculated by PCA
    # Otherwise, reduced is equal to original data
    #TODO: Change to use original density function
    val = kde(reduced.T)(np.vstack([u.flatten(), v.flatten()]))

    plt.figure(figsize=(18, 10))
    fig = gds.GridSpec(3, 6)

    plt.subplot(fig[0,0:2])
    plt.title('Data Scatter Plot')
    plt.plot(x, y, 'ko')
    plt.xticks([])
    plt.yticks([])
    plt.subplot(fig[0,2:4])
    plt.title('Gaussian KDE')
    plt.pcolormesh(u, v, val.reshape(u.shape), cmap=plt.cm.BuGn_r)
    plt.xticks([])
    plt.yticks([])
    plt.subplot(fig[0,4:6])
    plt.title('Density Contours')
    plt.pcolormesh(u, v, val.reshape(u.shape), cmap=plt.cm.BuGn_r, shading='gouraud')
    plt.contour(u, v, val.reshape(u.shape))
    plt.xticks([])
    plt.yticks([])

    # ax0 = plt.subplot(fig[1:3,0:3], projection='3d')
    # ax0.set_title('Mapped Density over 2D Space')
    # ax0.set_xticks([])
    # ax0.set_yticks([])
    # ax0.set_zticks([])
    # ax0.scatter(u, v, val, s=2, c='lightblue')
    # ax0.set_xlabel('x Coordinate')
    # ax0.set_ylabel('y Coordinate')
    # ax0.set_zlabel('Density Value')
    #
    # ax1 = plt.subplot(fig[1:3,3:6], projection='3d')
    # ax1.set_title('Density Estimate over 2D Space')
    # ax1.set_xticks([])
    # ax1.set_yticks([])
    # ax1.set_zticks([])
    # ax1.scatter(x, y, vec, s=2, c='lightgrey')
    # ax1.set_xlabel('x Coordinate')
    # ax1.set_ylabel('y Coordinate')
    # ax1.set_zlabel('Density Value')

    plt.tight_layout()
    plt.show()

    del x, y, u, v, val

def estimate_density(x, nbins=100, graph=False):

    den = kde(x.T)
    vec = den(np.vstack(([*x.T])))
    if (x.shape[1]>2):
        print('Start PCA...')
        start = time.time()
        reduced = PCA(n_components=2).fit_transform(x)
        end = time.time()
        print('PCA finished : took {} seconds.'.format(end-start))

    if graph:
        print('Start plot_density...')
        start = time.time()
        plot_density(x, reduced, nbins, den, vec)
        end = time.time()
        print('plot_density finished : took {} seconds.'.format(end-start))

    del den

    return vec

# Build the simplex tree and the corresponding filtration
# neighbors refers to the neighboring graph of each element
# graph is a boolean for data visualization
def estimate_clusters(neighbors=6, graph=False, raw_args=None):

    parser = argparse.ArgumentParser()
    parser.add_argument('--structure_file', default='',
                        type=str, help='path to the structure file')
    parser.add_argument('--density_path', default='',
                        type=str, help='path to the density file')
    parser.add_argument('--neighbors_path', default='',
                        type=str, help='path to the neighbors file')
    args = parser.parse_args(raw_args)
    #err = 'You should provide a structure_file'
    #assert args.structure_file is not None, err

    if args.structure_file!='':
        coordinates = []
        xyz = open(args.structure_file)
        for line in xyz:
            x, y, z = line.split()
            coordinates.append([float(x), float(y), float(z)])
        xyz.close()

        x = np.array(coordinates).reshape(-1, 10, 3)

        vec = estimate_density(x, graph=True)

        kdt = KDTree(x, metric='euclidean')

    elif args.density_path!='' and args.neighbors_path!='':
        neighbors = np.load(args.neighbors_path)
        density = np.load(args.density_path)

    sxt = gudhi.SimplexTree()

    print('Building sxt...')
    for ind in tqdm.tqdm(range(x.shape[0])):
        sxt.insert([ind], filtration=-vec[ind])
        if args.structure_file!='':
            nei = kdt.query([x[ind]], neighbors, return_distance=False)[0][1:]
        else:
            nei =
        for idx in nei:
            sxt.insert([ind, idx], filtration=np.mean([-vec[ind], -vec[idx]]))

    sxt.initialize_filtration()
    sxt.persistence()

    if graph:

        dig, res = sxt.persistence(), []
        print('Building res...')
        for ele in tqdm.tqdm(dig):
            if ele[0] == 0:
                res.append(ele)

        plt.figure(figsize=(18, 4))
        fig = gds.GridSpec(1, 2)
        plt.subplot(fig[0,0])
        gudhi.plot_persistence_diagram(res)
        plt.subplot(fig[0,1])
        gudhi.plot_persistence_barcode(res)
        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    estimate_clusters(graph=True)
