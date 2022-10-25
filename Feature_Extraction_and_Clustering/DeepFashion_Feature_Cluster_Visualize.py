import numpy as np
import matplotlib.pyplot as plt
import glob
from sklearn.cluster import KMeans
import joblib
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.manifold import TSNE
import seaborn as sns
import datetime
import os
import pandas as pd
from pathlib import Path
from pandas import DataFrame
import shutil

Color = [
    'black', 'red', 'bright orange', 'orange yellow', 'golden', 'sunshine yellow', 'olive green', 'light lime', 'frog green', 'cool green', 'bright aqua',
    'baby blue', 'baby purple', 'purply pink', 'baby pink',
    'coral', 'tomato', 'clay', 'rust', 'orangeish', 'mocha', 'coffee', 'tan',
    'desert', 'sunflower', 'wheat', 'stone', 'beige', 'custard', 'banana', 'light khaki',
    'tan green', 'electric lime', 'leaf', 'moss', 'grass', 'apple', 'sage', 'dull green',
    'seafoam green', 'light teal', 'greenblue', 'greenish cyan', 'tiffany blue', 'ice blue', 'sky', 'cool blue',
    'bluish', 'windows blue', 'faded blue', 'clear blue', 'lavender blue', 'light lavender', 'heather', 'faded purple',
    'violet pink', 'lavender pink', 'purplish', 'grape', 'rich purple', 'pale magenta', 'rose pink', 'dusty rose',
    'solmon', 'peachy pink', 'dusty orange', 'brown grey', 'dark khaki', 'greenish tan', 'light moss green', 'slime green', 'hospital green', 'pale olive',
    'seaweed green', 'algae green', 'duck egg blue', 'dull blue', 'soft blue', 'light periwinkle', 'candy pink', 'pig pink'
]

importance = 0.85
# MODEL_ROOT = '/home/hwx107m/Desktop/thesis/Model/DeepFashion'
# RESULTS_ROOT = '/home/hwx107m/Desktop/thesis/Result/DeepFashion'
# FEATURES_ROOT = '/home/hwx107m/Desktop/thesis/Feature/DeepFashion'


def feature_cluster(features, clusters=10, save=False):
    """
    concatenate three types features with PCA  as  inputs
    inputs : features
    outputs : cluster module
    """

    kmeans = KMeans(n_clusters=clusters, random_state=2, n_init=30)
    print("kmeans fitting...")
    start = datetime.datetime.now()
    kmeans.fit(features)
    end = datetime.datetime.now()
    print(f'kmeans using time : {end - start}')

    if save:
        joblib.dump(kmeans, f"{MODEL_ROOT}/fv_pca-{importance}_{clusters}-means.pkl")
        print(f'fv_pca-{importance}_{clusters}-means model saved!!!')
    return kmeans


def feature_visualize_DBSCAN(features, cluster):
    print("Silhouette score calculating...")
    start = datetime.datetime.now()
    # Get silhouette samples
    CLUSTERS = np.unique(cluster.labels_).shape[0]
    max_clust = np.max(cluster.labels_)
    avg_score = -1
    fig = plt.figure()
    ax3 = fig.add_subplot(121)
    y_lower, y_upper = 0, 0
    # Silhouette plot
    if CLUSTERS > 1:
        silhouette_vals = silhouette_samples(features, cluster.labels_)
        for i, k in enumerate(np.unique(cluster.labels_)):
            if k == -1:
                color = Color[61]
            elif k > 60:
                color = Color[60]
            else:
                color = Color[k]
            cluster_silhouette_vals = silhouette_vals[cluster.labels_ == k]
            cluster_silhouette_vals.sort()
            y_upper += len(cluster_silhouette_vals)

            ax3.barh(range(y_lower, y_upper), cluster_silhouette_vals, edgecolor='none', height=1, color=sns.xkcd_rgb[color])
            ax3.text(-0.03, (y_lower + y_upper) / 2, str(k))
            y_lower += len(cluster_silhouette_vals)

        # Get the average silhouette score and plot it
        avg_score = np.mean(silhouette_vals[cluster.labels_ != -1])
        end = datetime.datetime.now()
        print(f'Silhouette score using time : {end - start}')
        ax3.axvline(avg_score, linestyle='--', linewidth=2, color='red')
        ax3.set_yticks([])
        ax3.set_xlim([-0.1, 1])
        ax3.set_xlabel('Silhouette coefficient values')
        ax3.set_ylabel('Cluster labels')
        ax3.set_title('Silhouette plot for the various clusters', y=1.02)
        print('Average silhouette score = %f' % avg_score)

    # tmp = np.concatenate((cluster.cluster_centers_, features), axis=0)
    print("TSNE fitting...")
    start = datetime.datetime.now()
    tsne_fv = TSNE(n_components=2).fit_transform(features)
    end = datetime.datetime.now()
    print(f'TSNE using time : {end - start}')
    # tsne_fv_cen = tsne_fv[0:CLUSTERS, :]
    # tsne_fv = tsne_fv[CLUSTERS:, :]
    ax4 = fig.add_subplot(122)
    ax4.set_title('fisher vector clustering visualization')
    for i in range(-1, max_clust+1):
        if i > 60:
            color = Color[60]
        else:
            color = Color[i]
        index = np.argwhere(cluster.labels_ == i)
        ax4.scatter(tsne_fv[index[:], 0], tsne_fv[index[:], 1],
                    c=sns.xkcd_rgb[color], s=20, marker='.', label=i)
        # ax4.scatter(tsne_fv_cen[i, 0], tsne_fv_cen[i, 1], c=sns.xkcd_rgb[Color[i]],
        #             s=100, alpha=0.5, marker='*', label=i, edgecolors='black')
    ax4.legend(loc=(1, 0), ncol=3, fontsize=8)
    fig.set_size_inches(18.5, 10.5)

    return avg_score, fig, max_clust


def feature_visualize(features, kmeans, CLUSTERS, RESULTS_ROOT, draw=False, save=False):
    """ print("kmeans predicting...")
    start = datetime.datetime.now()
    kmeans.predict(features)
    end = datetime.datetime.now()
    print(f'kmeans using time : {end - start}') """
    # Silhouette analysis

    print("Silhouette score calculating...")
    start = datetime.datetime.now()
    # Get silhouette samples
    silhouette_vals = silhouette_samples(features, kmeans.labels_)

    # Silhouette plot
    fig2 = plt.figure()
    ax3 = fig2.add_subplot(121)
    y_lower, y_upper = 0, 0
    for i, cluster in enumerate(np.unique(kmeans.labels_)):
        cluster_silhouette_vals = silhouette_vals[kmeans.labels_ == cluster]
        cluster_silhouette_vals.sort()
        y_upper += len(cluster_silhouette_vals)
        ax3.barh(range(y_lower, y_upper), cluster_silhouette_vals,
                 edgecolor='none', height=1, color=sns.xkcd_rgb[Color[cluster]])
        ax3.text(-0.03, (y_lower + y_upper) / 2, str(i + 1))
        y_lower += len(cluster_silhouette_vals)
    
    # Get the average silhouette score and plot it
    avg_score = np.mean(silhouette_vals)
    end = datetime.datetime.now()
    print(f'Silhouette score using time : {end - start}')
    ax3.axvline(avg_score, linestyle='--', linewidth=2, color='red')
    ax3.set_yticks([])
    ax3.set_xlim([-0.1, 1])
    ax3.set_xlabel('Silhouette coefficient values')
    ax3.set_ylabel('Cluster labels')
    ax3.set_title('Silhouette plot for the various clusters', y=1.02)
    print('Average silhouette score = %f' % avg_score)

    tmp = np.concatenate((kmeans.cluster_centers_, features), axis=0)
    del features
    print("TSNE fitting...")
    start = datetime.datetime.now()
    tsne_fv = TSNE(n_components=2).fit_transform(tmp)
    end = datetime.datetime.now()
    print(f'TSNE using time : {end - start}')
    del tmp
    tsne_fv_cen = tsne_fv[0:CLUSTERS, :]
    tsne_fv = tsne_fv[CLUSTERS:, :]
    ax4 = fig2.add_subplot(122)
    ax4.set_title('fisher vector clustering visualization')
    for i in range(0, CLUSTERS):
        index = np.argwhere(kmeans.labels_ == i)
        ax4.scatter(tsne_fv[index[:], 0], tsne_fv[index[:], 1],
                    c=sns.xkcd_rgb[Color[i]], s=20, marker='.', label=i)
        ax4.scatter(tsne_fv_cen[i, 0], tsne_fv_cen[i, 1], c=sns.xkcd_rgb[Color[i]],
                    s=100, alpha=0.5, marker='*', label=i, edgecolors='black')
    ax4.legend(loc=(1, 0), ncol=3, fontsize=8)
    fig2.set_size_inches(18.5, 10.5)
    if save:
        fig2.savefig(f'{RESULTS_ROOT}/{CLUSTERS}_clusters_result.png', format='png')
    if draw:
        plt.show()
    return avg_score, fig2


def TSNE_visual(features, RESULTS_ROOT):
    fig = plt.figure()
    print("TSNE fitting...")
    start = datetime.datetime.now()
    tsne_fv = TSNE(n_components=2).fit_transform(features)
    end = datetime.datetime.now()
    print(f'TSNE using time : {end - start}')
    ax = fig.add_subplot(122)
    ax.set_title('fisher vector clustering visualization')
    ax.scatter(tsne_fv[:, 0], tsne_fv[:, 1], c=sns.xkcd_rgb[Color[0]], s=20, marker='.')
    ax.legend(loc=(1, 0), ncol=3, fontsize=8)
    fig.set_size_inches(18.5, 10.5)
    fig.savefig(f'{RESULTS_ROOT}/origine_feature.png', format='png')


def copy2label(result: DataFrame, clusters, save_img_path):
    '''
    copy each of input image to their corresponding label folder.

    '''
    dir_name = Path(save_img_path, f'{clusters}-result')
    os.makedirs(dir_name)
    # create folders of each cluster
    for i in range(clusters):
        os.makedirs(Path(dir_name, str(i)))
    # copy images
    for i in range(len(result)):
        curr_img = str(result.iloc[i, 0])
        label = result.iloc[i, 1]
        img_name = curr_img.split('\\')[-1]
        dest_img = Path(dir_name, str(label), img_name)
        shutil.copyfile(Path(curr_img), dest_img)


def save_result(labels: list, names: list, clusters, out_path):
    result = pd.DataFrame(columns=['file_name', 'label'])
    result['file_name'] = names
    result['label'] = labels
    copy2label(result, clusters, out_path)


if __name__ == "__main__":
    """ partial_features = np.concatenate((np.load(f'{FEATURES_ROOT}/one-fifteenth_material_fv_pca_{importance}.npy'), 
                                    np.load(f'{FEATURES_ROOT}/one-fifteenth_texture_fv_pca_{importance}.npy'),
                                    np.load(f'{FEATURES_ROOT}/one-fifteenth_color_fv_pca_{importance}.npy')), axis=1)
    print("partial feature concat!!!")
    print(f'partial feature shape: {partial_features.shape}')

    KMS = feature_cluster(partial_features, CLUSTERS)
    # label = KMS.labels_
    # print(KMS.labels_)
    del partial_features """

    # feature_path = 'C:/Users/TSSW/Desktop/clothing/clustering/Final Code/features/女裝/春夏/*/'
    feature_path = 'C:/Users/TSSW/Desktop/clothing/clustering/Final Code/features/男裝/春夏/'
    with open(f'../results/男裝/k-means-Average silhouette score.txt', 'a+') as f:
        f.writelines(['春夏\n', 'Clusters'])
        for c in range(5, 51, 5):
            f.write(f', {c}')
        f.write('\n')

    F = glob.glob(feature_path)
    for FEATURES_ROOT in F:
        s = FEATURES_ROOT.split('\\')
        # print(s[1], s[2])
        # MODEL_ROOT = f'../weights/男裝/春夏/{s[1]}'
        # RESULTS_ROOT = f'../results/男裝/春夏/{s[1]}'
        RESULTS_ROOT = f'../results/男裝/春夏'
        if not os.path.exists(RESULTS_ROOT):
            os.makedirs(RESULTS_ROOT)
        # all_features = np.concatenate((np.load(f'{FEATURES_ROOT}/material_fv_pca_{importance}.npy'),
        #                                np.load(f'{FEATURES_ROOT}/texture_fv_pca_{importance}.npy'),
        #                                np.load(f'{FEATURES_ROOT}/color_fv_pca_{importance}.npy')), axis=1)
        # all_features = np.concatenate((np.load(f'{FEATURES_ROOT}/material.npy'),
        #                                np.load(f'{FEATURES_ROOT}/texture.npy'),
        #                                np.load(f'{FEATURES_ROOT}/color.npy')), axis=1)
        all_features = np.load(f'{FEATURES_ROOT}/color.npy')
        # color_feature = np.load(f'{FEATURES_ROOT}/color_fv_pca_{importance}.npy')
        # all_features = color_feature
        print("all feature concat!!!")
        print(f'all feature shape: {all_features.shape}')

        # with open(f'../results/女裝/k-means-Average silhouette score.txt', 'a+') as f:
        #     f.write(f'{s[1]}')

        all_img_file = glob.glob(f'D:/時裝週原始資料-2021_crop/男裝/春夏/*/*.jpg')
        for i in range(5, 51, 5):
            KMS = feature_cluster(all_features, i)
            label = KMS.labels_
            save_result(label, names=all_img_file, clusters=i, out_path=RESULTS_ROOT)
            score = feature_visualize(all_features, KMS, i, RESULTS_ROOT)
            with open(f'../results/男裝/k-means-Average silhouette score.txt', 'a+') as f:
                f.write(f', {score}')
        with open(f'../results/男裝/k-means-Average silhouette score.txt', 'a+') as f:
            f.write('\n')

    # feature_path = 'C:/Users/TSSW/Desktop/clothing/clustering/Final Code/features/女裝/秋冬/*/'
    feature_path = 'C:/Users/TSSW/Desktop/clothing/clustering/Final Code/features/男裝/秋冬/'
    with open(f'../results/男裝/k-means-Average silhouette score.txt', 'a+') as f:
        f.write('秋冬\n')
    F = glob.glob(feature_path)
    for FEATURES_ROOT in F:
        s = FEATURES_ROOT.split('\\')
        # print(s[1])
        # MODEL_ROOT = f'../weights/女裝/秋冬/{s[1]}'
        # RESULTS_ROOT = f'../results/女裝/秋冬/{s[1]}'
        RESULTS_ROOT = f'../results/男裝/秋冬'
        if not os.path.exists(RESULTS_ROOT):
            os.makedirs(RESULTS_ROOT)
        # all_features = np.concatenate((np.load(f'{FEATURES_ROOT}/material_fv_pca_{importance}.npy'),
        #                                np.load(f'{FEATURES_ROOT}/texture_fv_pca_{importance}.npy'),
        #                                np.load(f'{FEATURES_ROOT}/color_fv_pca_{importance}.npy')), axis=1)
        # all_features = np.concatenate((np.load(f'{FEATURES_ROOT}/material.npy'),
        #                                np.load(f'{FEATURES_ROOT}/texture.npy'),
        #                                np.load(f'{FEATURES_ROOT}/color.npy')), axis=1)
        all_features = np.load(f'{FEATURES_ROOT}/color.npy')
        # color_feature = np.load(f'{FEATURES_ROOT}/color_fv_pca_{importance}.npy')
        # all_features = color_feature
        print("all feature concat!!!")
        print(f'all feature shape: {all_features.shape}')

        # with open(f'../results/女裝/k-means-Average silhouette score.txt', 'a+') as f:
        #     f.write(f'{s[1]}')

        all_img_file = glob.glob(f'D:/時裝週原始資料-2021_crop/男裝/秋冬/*/*.jpg')
        for i in range(5, 51, 5):
            KMS = feature_cluster(all_features, i)
            label = KMS.labels_
            save_result(label, names=all_img_file, clusters=i, out_path=RESULTS_ROOT)
            score = feature_visualize(all_features, KMS, i, RESULTS_ROOT)
            with open(f'../results/男裝/k-means-Average silhouette score.txt', 'a+') as f:
                f.write(f', {score}')
        with open(f'../results/男裝/k-means-Average silhouette score.txt', 'a+') as f:
            f.write('\n')
