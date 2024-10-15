import os
import glob
import h5py
import numpy as np
from torch.utils.data import Dataset

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


def download(root_path='../data/ModelNet40'):
    if not os.path.exists(root_path):
        os.mkdir(root_path)
        # www = 'https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip'
        # zipfile = os.path.basename(www)
        # os.system(f'wget {www} --no-check-certificate')
        os.system(f'mc cp -r minio/perception/meicheng/DataSet/ModelNet40.zip {root_path}')
        os.system(f'unzip {root_path}/ModelNet40.zip -d {root_path}')
        os.system(f'mv {root_path}/modelnet40_ply_hdf5_2048/* {root_path}')
        os.system(f'rm -rf {root_path}/ModelNet40.zip')
        os.system(f'rm -rf {root_path}/modelnet40_ply_hdf5_2048')


def load_data(root_path='../data/ModelNet40', split='train', num_points=1024):
    download(root_path)
    all_data = []
    all_cls = []
    for h5_name in glob.glob(f'{root_path}/ply_data_{split}*.h5'):
        f = h5py.File(h5_name, mode="r")
        all_data.append(f['data'][:].astype('float32'))
        all_cls.append(f['label'][:].astype('int64'))
        f.close()
    all_data = np.concatenate(all_data, axis=0)
    all_cls = np.concatenate(all_cls, axis=0)
    all_data = all_data[:, :num_points, :]
    all_data = normalize_pointcloud(all_data)
    return all_data, all_cls


def normalize_pointcloud(pts):
    # pts: b_s, n_p, c_s
    pts = pts - np.mean(pts, axis=1, keepdims=True)
    pts = pts / np.max(np.sqrt(np.sum(pts ** 2, axis=2, keepdims=True)), axis=1, keepdims=True)
    return pts


def translate_pointcloud(pts):
    # pts: n_p, c_s
    scale = np.random.uniform(low=2./3., high=3./2., size=[3])
    translate = np.random.uniform(low=-0.2, high=0.2, size=[3])
    pts = np.add(np.multiply(pts, scale), translate).astype('float32')
    return pts


class ModelNet40(Dataset):
    def __init__(self, data_path='../data/ModelNet40', split='train', num_points=1024):
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
    from torch.utils.data import DataLoader
    train_loader = DataLoader(ModelNet40(f'data', split='train', num_points=1024), num_workers=4, batch_size=32, shuffle=True, drop_last=True)
    for batch_idx, (data, label) in enumerate(train_loader):
        print(f"batch_idx: {batch_idx}  | data shape: {data.shape} | lable shape: {label.shape}")

    train_set = ModelNet40(f'data', split='train', num_points=1024)
    test_set = ModelNet40(f'data', split='test', num_points=1024)
    print(f"train_set size {train_set.__len__()}")
    print(f"test_set size {test_set.__len__()}")