import numpy as np
# import configure as cfg
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import joblib
import datetime
import glob

# FEATURES_ROOT = '/home/TSSW/Project/apparel/data/train_feature'
IMPORTANCE = 0.85


def feature_reduce(features_root, feature_type, importance=0.95, save_path=None, save=True):
# def feature_reduce(feature, feature_type, importance=cfg.PCA_Importance, save=True):
    """
    Feature Dimension Reduction 

    inputs : (features,feature_type)
        e.g. (features=material,feature_type='material')

    outputs : pca_feature
    """
    #print(f"start to dimension reduce")
    partial_feature = np.load(f'{features_root}/{feature_type}_fv_1.npy')
    print(partial_feature.shape)
    f_PCA = PCA(n_components=importance)
    start = datetime.datetime.now()
    print("start fitting...")
    partial_pca = f_PCA.fit_transform(partial_feature)
    joblib.dump(f_PCA, f'{save_path}/{feature_type}_pca.m')
    del partial_feature
    end = datetime.datetime.now()
    print(f'{feature_type} fitting time : {end - start}')
    np.save(f"{features_root}/{feature_type}_fv_pca_{IMPORTANCE}.npy", partial_pca)
    print(f'one-fifteenth {feature_type} shape: {partial_pca.shape}')
    del partial_pca

    # start = datetime.datetime.now()
    # print("start transform...")
    # all_pca = []
    # for i in range(total_part):
    #     feature = np.load(f'{features_root}/{feature_type}_fv_{i+1}.npy')
    #     split1, split2, split3, split4, split5 = np.array_split(feature, 5, axis=0)
    #     # split1, split2, split3, split4, split5, split6, split7, split8, split9, split10 = np.array_split(feature, 10, axis=0)
    #     del feature
    #     print("split finished")
    #     pca_feature = f_PCA.transform(split1)
    #     del split1
    #     pca_feature = np.concatenate((pca_feature, f_PCA.transform(split2)), axis=0)
    #     del split2
    #     pca_feature = np.concatenate((pca_feature, f_PCA.transform(split3)), axis=0)
    #     del split3
    #     pca_feature = np.concatenate((pca_feature, f_PCA.transform(split4)), axis=0)
    #     del split4
    #     pca_feature = np.concatenate((pca_feature, f_PCA.transform(split5)), axis=0)
    #     del split5
    #     # pca_feature = np.concatenate((pca_feature, f_PCA.transform(split6)), axis=0)
    #     # del split6
    #     # pca_feature = np.concatenate((pca_feature, f_PCA.transform(split7)), axis=0)
    #     # del split7
    #     # pca_feature = np.concatenate((pca_feature, f_PCA.transform(split8)), axis=0)
    #     # del split8
    #     # pca_feature = np.concatenate((pca_feature, f_PCA.transform(split9)), axis=0)
    #     # del split9
    #     # pca_feature = np.concatenate((pca_feature, f_PCA.transform(split10)), axis=0)
    #     # del split10
    #
    #     if i==0:
    #         all_pca = pca_feature
    #         del pca_feature
    #     else:
    #         all_pca = np.concatenate((all_pca, pca_feature), axis=0)
    #         del pca_feature
    #
    # end = datetime.datetime.now()
    # print(f'{feature_type} transform time : {end - start}')
    #
    # print(f'{feature_type} shape: {all_pca.shape}')
    #
    # if save:
    #     # joblib.dump(f_PCA, f"{cfg.Model_Root}/{feature_type}_PCA.pkl")
    #     # np.save(f'{cfg.Feature_Root}/{feature_type}_pca.npy', all_pca_feature)
    #     np.save(f"{features_root}/{feature_type}_fv_pca_{IMPORTANCE}.npy", all_pca)
    #
    # del all_pca


if __name__ == "__main__":
    feature_path = 'C:/Users/TSSW/Desktop/clothing/clustering/Final Code/features/女裝/*/*/'
    # save_path = '../weights/女裝/{}/{}'
    for FEATURES_ROOT in glob.glob(feature_path):
        s = FEATURES_ROOT.split('\\')
        save_path = f'../weights/女裝/{s[1]}/{s[2]}'
        # save_path = f'../weights/男裝/{s[1]}'
        feature_reduce(f'{FEATURES_ROOT}', 'material', IMPORTANCE, save_path)
        feature_reduce(f'{FEATURES_ROOT}', 'texture', IMPORTANCE, save_path)
        feature_reduce(f'{FEATURES_ROOT}', 'color', IMPORTANCE, save_path)
    
    
    """ material = np.load(f'{FEATURES_ROOT}/material_fv.npy')
    _ = feature_reduce(material, 'material')

    texture = np.load(f'{FEATURES_ROOT}/texture_fv.npy')
    _ = feature_reduce(texture, 'texture')

    color = np.load(f'{FEATURES_ROOT}/color_fv.npy')
    _ = feature_reduce(color, 'color') """