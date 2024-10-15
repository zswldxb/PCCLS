import os
import glob
import h5py
import numpy as np
from torch.utils.data import Dataset
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


def download_data(data_path="../data/ScanObjectNN"):
    if not os.path.exists(data_path):
        os.mkdir(data_path)
        # www = 'https://github.com/ma-xu/pointMLP-pytorch/releases/download/dataset/h5_files.zip'
        # zipfile = os.path.basename(www)
        # os.system(f'wget {www} --no-check-certificate')
        os.system(f'mc cp -r minio/perception/meicheng/DataSet/ScanObjectNN.zip {data_path}')
        os.system(f'unzip {data_path}/ScanObjectNN.zip -d {data_path}')
        os.system(f'mv {data_path}/h5_files/main_split/training_objectdataset_augmentedrot_scale75.h5 {data_path}/train.h5')
        os.system(f'mv {data_path}/h5_files/main_split/test_objectdataset_augmentedrot_scale75.h5 {data_path}/test.h5')
        os.system(f'rm -rf {data_path}/ScanObjectNN.zip')
        os.system(f'rm -rf {data_path}/h5_files')
        os.system(f'rm -rf {data_path}/__MACOSX')



def load_data(data_path='../data/ScanObjectNN', split='train', num_points=1024):
    download_data(data_path)
    h5_name = f'{data_path}/{split}.h5'
    f = h5py.File(h5_name, mode="r")
    all_data = f['data'][:].astype('float32')
    all_cls = f['label'][:].astype('int64')
    f.close()
    all_data = all_data[:, :num_points, :]
    return all_data, all_cls


def translate_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2./3., high=3./2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])
    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    return translated_pointcloud


class ScanObjectNN(Dataset):
    def __init__(self, data_path, split='train', num_points=1024):
        self.data, self.cls = load_data(data_path, split, num_points)
        self.split = split

    def __getitem__(self, item):
        data = self.data[item]
        cls = self.cls[item]
        if self.split == 'train':
            data = translate_pointcloud(data)
            np.random.shuffle(data)
        return data, cls

    def __len__(self):
        return self.data.shape[0]


if __name__ == '__main__':
    train = ScanObjectNN(f'../data/ScanObjectNN', split='train', num_points=1024)
    test = ScanObjectNN(f'../data/ScanObjectNN', split='test', num_points=1024)

    # save = []
    # for i, (data, cls) in enumerate(train):
    #     if i > 10: break
    #     save.append({'../data': data, 'cls': cls})
    # np.save(f'scanobjectnn_vis', save)