import os
import glob
import numpy as np
from fishervector import FisherVectorGMM
import datetime

# FEATURES_ROOT = '/home/TSSW/Project/apparel/data/train_feature'
# MODEL_ROOT = '/home/TSSW/Project/apparel/data/train_feature'
GMM_KERNELS = 5


def feature_optimize(features_root, feature_type, MODEL_ROOT, save=True):
    """
    feature_type : material (2048) , texture(2048) ,color(512)

    inputs : (features,feature_type)
            e.g. (features=material,feature_type='material')
    outputs : fisher vectors

    """

    # features = np.load(f'{features_root}/one-fifteenth_{feature_type}.npy')
    features = np.load(f'{features_root}/{feature_type}.npy')
    features_size = features.shape[1]
    features = np.reshape(features, (-1, 1, features_size))
    print("GMM training...")
    start = datetime.datetime.now()
    fv_gmm = FisherVectorGMM(n_kernels=GMM_KERNELS).fit(
            features, model_dump_path=MODEL_ROOT+'/'+feature_type + '_GMM', verbose=False)
    # fv_gmm = FisherVectorGMM(n_kernels=cfg.GMM_Kernel).fit(
    #           features, model_dump_path=cfg.Model_Root+'/'+feature_type + '_GMM', verbose=False)
    print(f"{feature_type} GMM training completed!!!")
    end = datetime.datetime.now()
    print(f'{feature_type} GMM training using time : {end - start}')

    print("Fisher Vector trasfering...")
    # part_fv = []
    # partial_features = np.load(f'{features_root}/one-fifteenth_{feature_type}.npy')
    # partial_features_size = partial_features.shape[1]
    # partial_features = np.reshape(partial_features, (-1, 1, partial_features_size))
    #
    # start = datetime.datetime.now()
    # for f in partial_features:
    #     fv = fv_gmm.predict(np.expand_dims(f, axis=0))
    #     fv = fv.flatten()
    #     part_fv.append(fv)
    # del partial_features
    # part_fv = np.asarray(part_fv)
    # print(f"one-fifteenth {feature_type} fisher vector shape: {part_fv.shape}")
    # if save:
    #     np.save(f"{features_root}/one-fifteenth_{feature_type}_fv.npy", part_fv)
    # del part_fv

    all_fv = []
    count = 0
    part = 1
    for f in features:
        fv = fv_gmm.predict(np.expand_dims(f, axis=0))
        fv = fv.flatten()
        all_fv.append(fv)
        count += 1

        if count % 50000 == 0:
            all_fv = np.asarray(all_fv)
            print(f"{feature_type} part {part} fisher vector shape: {all_fv.shape}")
            if save:
                # np.save(f"{cfg.Feature_Root}/{feature_type}_fv.npy", all_fv)
                np.save(f"{features_root}/{feature_type}_fv_{part}.npy", all_fv)
            all_fv = []
            part += 1
    all_fv = np.asarray(all_fv)
    print(f"{feature_type} part {part} fisher vector shape: {all_fv.shape}")
    if save:
        # np.save(f"{cfg.Feature_Root}/{feature_type}_fv.npy", all_fv)
        np.save(f"{features_root}/{feature_type}_fv_{part}.npy", all_fv)

    end = datetime.datetime.now()
    print(f'{feature_type} using time : {end - start}')


if __name__ == "__main__":

    # TYPE = ['color', 'material', 'texture']
    # feature_path = 'D:/時裝週原始資料-2021/features/男裝/*/'
    feature_path = 'C:/Users/TSSW/Desktop/clothing/clustering/Final Code/features/男裝/*/'
    # save_path = '../weights/女裝/{}/{}'
    for FEATURES_ROOT in glob.glob(feature_path):
        s = FEATURES_ROOT.split('\\')
        save_path = f'../weights/男裝/{s[1]}'
        # save_path = f'../weights/女裝/{s[1]}/{s[2]}'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        feature_optimize(FEATURES_ROOT, 'color', save_path)
        feature_optimize(FEATURES_ROOT, 'material', save_path)
        feature_optimize(FEATURES_ROOT, 'texture', save_path)



    # material = np.load(f'{FEATURES_ROOT}/one-fifteenth_material.npy')
    # _ = feature_optimize(material, 'material')

    # texture = np.load(f'{FEATURES_ROOT}/texture.npy')
    # _ = feature_optimize(texture, 'texture')

    # color = np.load(f'{FEATURES_ROOT}/color.npy')
    # _ = feature_optimize(color, 'color')