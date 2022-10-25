import datetime

import numpy as np
from fishervector import FisherVectorGMM
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA

from Feature_Extract import color_extract
from Feature_Extraction_and_Clustering.DeepFashion_Feature_Cluster_Visualize import feature_visualize, \
    feature_visualize_DBSCAN
from Feature_Extract import feature_extract, color_extract
from Feature_Extraction_and_Clustering.dataset import find_image


def feature_Reduce(feature, importance=0.9):
    f_PCA = PCA(n_components=importance, copy=False)
    start = datetime.datetime.now()
    print("start fitting...")
    reduce_feature = f_PCA.fit_transform(feature)
    print(f'fitting time : {datetime.datetime.now() - start}')
    return reduce_feature


def DBSCAN_cluster(feature, eps=1, min_samples=3):
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(feature)
    score, fig, clusters = feature_visualize_DBSCAN(feature, clustering)
    return clustering, clusters, fig


def cluster(feature, save_path, best_kernel=None):
    best_kmeans = KMeans(n_clusters=best_kernel, random_state=2, n_init=30)
    best_kmeans.fit(feature)
    score, fig = feature_visualize(feature, best_kmeans, best_kernel, save_path, save=False)
    # save_result(best_kmeans.labels_, names=img_path, clusters=best_clust, out_path=save_path)
    return best_kmeans, best_kernel, fig


def get_fishervector(feature, n_kernels: list):
    features_size = feature.shape[1]
    features = np.reshape(feature, (-1, 1, features_size))
    fv_gmm = FisherVectorGMM().fit_by_bic(features, choices_n_kernels=n_kernels, model_dump_path=None, verbose=True)

    all_fv = []
    for f in features:
        fv = fv_gmm.predict(np.expand_dims(f, axis=0))
        fv = fv.flatten()
        all_fv.append(fv)
    all_fv = np.asarray(all_fv)

    return all_fv, fv_gmm.n_kernels


if __name__ == '__main__':
    # img_root = "/mnt/Nami/TSSW/時裝週原始資料-2021_segment/男裝/春夏"
    # img_path = find_image(img_root)
    # color_feature = feature_extract(img_path, feature_type='color')
    # material_feature = feature_extract(img_path, feature_type='material')
    # texture_feature = feature_extract(img_path, feature_type='texture')
    #
    # # concat_feature = np.concatenate((color_feature, material_feature, texture_feature), 1)
    # n_clust = [i for i in range(3, 41)]
    #
    # # fv, selected_kernels = get_fishervector(concat_feature, n_clust)
    # # reduce_feature = feature_Reduce(fv)
    #
    # # '''
    # color_fv, selected_kernels = get_fishervector(color_feature, n_clust)
    # color_reduce_feature = feature_Reduce(color_fv)
    #
    # material_fv, selected_kernels = get_fishervector(material_feature, n_clust)
    # material_reduce_feature = feature_Reduce(material_fv)
    # #
    # texture_fv, selected_kernels = get_fishervector(texture_feature, n_clust)
    # texture_reduce_feature = feature_Reduce(texture_fv)
    # #
    # feature = np.concatenate((material_reduce_feature, texture_reduce_feature, color_reduce_feature), 1)
    # # '''
    # cluster(feature, img_path, "/mnt/Nami/TSSW/時裝週原始資料-2021_segment/男裝/春夏_xxx", best_kernel=selected_kernels,
    #         clusters=n_clust)
    # feature = np.concatenate((material_feature, texture_feature), 1)
    # TSNE_visual(feature, 'RESULTS_texture_material')

    n_clust = [i for i in range(3, 51)]
    path = 'D:/時裝週原始資料-2021/男裝/春夏'
    # self.progress_label.setText("load weights ...")
    img_path = find_image(path, [])
    material, texture = feature_extract(img_path)
    # color = color_extract(path)
    print(material.shape)
    print(texture.shape)
    feature = np.concatenate((texture, material), axis=1)
    print(feature.shape)
    # texture_fv, selected_kernels = get_fishervector(texture, n_clust)
    # texture_PCA = feature_Reduce(texture_fv)
    # color_fv, selected_kernels = get_fishervector(color, [c for c in range(3, 51)])
    # color_PCA = feature_Reduce(color_fv)

    feature_fv, selected_kernels = get_fishervector(feature, n_clust)
    feature_PCA = feature_Reduce(feature_fv)
    best_clustering, best_clusters, fig = DBSCAN_cluster(feature_PCA, eps=1, min_samples=3)
    fig.savefig('C:/Users/TSSW/Desktop/clothing/clustering/Final Code/results/clusters_result.png', format='png')
    # n_clusters = best_clusters
    # label = best_clustering.labels_
