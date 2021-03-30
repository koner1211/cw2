import csv
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN


# ładowanie pliku xyz
def cloud_points():
    with open(file="LidarData.xyz", newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for x, y, z in reader:
            yield float(x), float(y), float(z)


if __name__ == '__main__':
    read_points = list(cloud_points())

    #'pakowanie' danych
    X, Y, Z = zip(*read_points)
    #
    p = np.array(read_points)

    # DBSCAN
    clustering = DBSCAN(eps=15, min_samples=10).fit(p)
    labels = clustering.labels_

    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    print('Estimated number of clusters: %d' % n_clusters_)
    print('Estimated number of noise points: %d' % n_noise_)

    #algorytm KMeans
    n_clusters = 3
    k_means = KMeans(n_clusters=n_clusters)
    k_means = k_means.fit(p)
    labels = k_means.predict(p)

    red = labels == 0
    green = labels == 1
    cyan = labels == 2

    fig_2 = plt.figure()
    ax_2 = fig_2.add_subplot(projection='3d')
    plt.title('Points scattering in 3D')
    ax_2.scatter(p[red, 0], p[red, 1], p[red, 2], marker='o')
    ax_2.scatter(p[green, 0], p[green, 1], p[green, 2], marker='^')
    ax_2.scatter(p[cyan, 0], p[cyan, 1], p[cyan, 2], marker='x')
    plt.show()