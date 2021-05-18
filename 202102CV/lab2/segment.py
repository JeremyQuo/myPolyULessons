from utils import read_img_as_array, save_array_as_img, show_array_as_img
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.mixture import GaussianMixture

def naive_k_means(points, k, itr=100):
    '''
    :param points: (N, M)
    :param k: number of clusters
    :param itr: num of iteration

    :return:
    cluster_num: array of shape (N,)
        'points[i]' belongs to cluster 'cluster_num[i]'
    means: array of shape (k, M)
        means[i] is center of cluster i.
    '''
    N, M = points.shape
    # TODO: naive k-means clustering
    # your code here
    # random choose k nodes
    idx = np.random.choice(N, size=k, replace=False)

    for i in range(itr):
        # calculate the distance between each node and k
        chosen_point=points[idx]
        dist = cdist(points, chosen_point)

        # find the minimum's index of each row
        cluster_num=np.argmin(dist, axis=1)
        cluster_num=cluster_num.astype(np.int32)

        # create null numpy to save variables
        idx=np.zeros(shape=k,dtype=int)
        means=np.zeros(shape=(k,M))

        for i in range(k):
            # calculate the centroid of each cluster
            temp=points[cluster_num==i]
            centroid = np.mean(temp, axis=0)
            means[i] = centroid

            # calculate the new idx and send it to next iteration
            dist2centroid = cdist(points, centroid.reshape(1,-1))
            idx[i] = np.argmin(dist2centroid, axis=0).squeeze()
    print(idx)
    assert cluster_num.shape == (N,)
    assert means.shape == (k, M)
    return cluster_num, means

def em_clustering(points, k):
    '''
    :param points: (N, M)
    :param k: number of clusters
    :param itr: num of iteration

    :return:
    cluster_num: array of shape (N,)
        'points[i]' belongs to cluster 'cluster_num[i]'
    means: array of shape (k, M)
        means[i] is center of cluster i.
    '''
    N, M = points.shape
    # TODO: EM clustering
    # your code here
    gmm = GaussianMixture(n_components=8)

    gmm.fit(pixel_feature)

    cluster_num = gmm.predict(pixel_feature)
    cluster_num = cluster_num.astype(np.int32)
    means = gmm.means_

    assert cluster_num.shape == (N,)
    assert means.shape == (k, M)
    return cluster_num, means


def draw_segment_result(cluster_num, colors=None):
    '''
    :param cluster_num: array of shape (H, W), storing cluster number of each pixel
    :return: a image contains visualization of the segmentation.
    '''
    assert cluster_num.dtype == np.int
    num = np.max(cluster_num) + 1
    if colors is None:
        colors = np.random.randint(255, size=(num, 3), dtype=np.uint8)
    segment_visualization = colors[cluster_num]
    return segment_visualization

if __name__ == '__main__':
    # TODO: perform naive_k_means and em_clustering on 'sun.jpg'
    img = read_img_as_array('./sun.jpg')
    H, W, _ = img.shape

    row, col = np.indices((H, W))
    indices = np.stack([row, col], axis=2)
    pixel_feature = np.concatenate([img, indices], axis=-1)
    pixel_feature = pixel_feature.reshape((H * W, 5))
    cluster_num, means = naive_k_means(pixel_feature, 8)
    cluster_num = cluster_num.reshape([H, W])
    segment_visualization = draw_segment_result(cluster_num, colors=means[:, :3])
    save_array_as_img(segment_visualization, 'naive_k_means.png')


    cluster_num, means = em_clustering(pixel_feature, 8)
    cluster_num = cluster_num.reshape([H, W])
    segment_visualization = draw_segment_result(cluster_num, colors=means[:, :3])
    save_array_as_img(segment_visualization, 'em_k_means.png')
    pass
