import os
from glob import glob
import numpy as np
from torch.utils.data import Dataset, DataLoader
from data_augment import augment_img
#对数据集进行封装
class ct_dataset(Dataset):
    def __init__(self, mode, load_mode,augment,saved_path,test_patient, patch_n=None, patch_size=None, transform=None):
        assert mode in ['train', 'test'], "mode is 'train' or 'test'"
        assert load_mode in [0,1], "load_mode is 0 or 1"
        self.augment = augment
        self.load_mode = load_mode
        self.patch_n = patch_n
        self.patch_size = patch_size
        self.transform = transform
        input_path_source1 = sorted(glob(os.path.join(saved_path, '*_input_source1.npy')))  # glob遍历文件夹下所有文件或文件夹；sorted对所有可迭代的对象进行排序操作
        target_path_source1 = sorted(glob(os.path.join(saved_path, '*_target_source1.npy')))
        input_path_source2 = sorted(glob(os.path.join(saved_path, '*_input_source2.npy')))  # glob遍历文件夹下所有文件或文件夹；sorted对所有可迭代的对象进行排序操作
        target_path_source2 = sorted(glob(os.path.join(saved_path, '*_target_source2.npy')))
        input_path_source3 = sorted(glob(os.path.join(saved_path, '*_input_source3.npy')))  # glob遍历文件夹下所有文件或文件夹；sorted对所有可迭代的对象进行排序操作
        target_path_source3 = sorted(glob(os.path.join(saved_path, '*_target_source3.npy')))
        input_source1 = [f for f in input_path_source1 if test_patient not in f]
        target_source1 = [f for f in target_path_source1 if test_patient not in f]
        input_source2 = [f for f in input_path_source2 if test_patient not in f]
        target_source2 = [f for f in target_path_source2 if test_patient not in f]
        input_source3 = [f for f in input_path_source3 if test_patient not in f]
        target_source3 = [f for f in target_path_source3 if test_patient not in f]

        if mode == 'train':

            if load_mode == 0:  # batch data load
                self.input_source1 = input_source1
                self.target_source1 = target_source1
                self.input_source2 = input_source2
                self.target_source2 = target_source2
                self.input_source3 = input_source3
                self.target_source3 = target_source3
        else:  # mode =='test'

            if load_mode == 0:  # batch data load
                self.input_source1 = input_source1
                self.target_source1 = target_source1
                self.input_source2 = input_source2
                self.target_source2 = target_source2
                self.input_source3 = input_source3
                self.target_source3 = target_source3


    def __len__(self):
        return len(self.target_source1)

    def __getitem__(self, idx):
        input_img_source1, target_img_source1 = self.input_source1[idx], self.target_source1[idx]
        input_img_source2, target_img_source2= self.input_source2[idx], self.target_source2[idx]
        input_img_source3, target_img_source3 = self.input_source3[idx], self.target_source3[idx]
        if self.load_mode == 0:
            input_img_source1, target_img_source1 = np.load(input_img_source1), np.load(target_img_source1)
            input_img_source2, target_img_source2 = np.load(input_img_source2), np.load(target_img_source2)
            input_img_source3, target_img_source3 = np.load(input_img_source3), np.load(target_img_source3)

        if self.augment:
            temp = np.random.randint(0, 8)
            input_img_source1, target_img_source1 = augment_img(input_img_source1, temp), augment_img(target_img_source1, temp)
            input_img_source2, target_img_source2 = augment_img(input_img_source2, temp), augment_img(target_img_source2, temp)
            input_img_source3, target_img_source3 = augment_img(input_img_source3, temp), augment_img(target_img_source3, temp)
        if self.transform:
            input_img_source1 = self.transform(input_img_source1)
            target_img_source1 = self.transform(target_img_source1)
            input_img_source2 = self.transform(input_img_source2)
            target_img_source2 = self.transform(target_img_source2)
            input_img_source3 = self.transform(input_img_source3)
            target_img_source3 = self.transform(target_img_source3)

        if self.patch_size:
            input_patches_source1, target_patches_source1 = get_patch(input_img_source1,target_img_source1,
                                                      self.patch_n,self.patch_size)
            input_patches_source2, target_patches_source2 = get_patch(input_img_source2,target_img_source2,
                                                      self.patch_n,self.patch_size)
            input_patches_source3, target_patches_source3 = get_patch(input_img_source3,target_img_source3,
                                                      self.patch_n,self.patch_size)
            return (input_patches_source1, target_patches_source1),(input_patches_source2,target_patches_source2),(input_patches_source3,target_patches_source3)
        else:
            return (input_img_source1, target_img_source1), (input_img_source2, target_img_source2), (input_img_source3, target_img_source3)


def get_patch(full_input_img, full_target_img, patch_n, patch_size): # 定义patch
    assert full_input_img.shape == full_target_img.shape
    patch_input_imgs = []
    patch_target_imgs = []
    h, w = full_input_img.shape
    new_h, new_w = patch_size, patch_size
    for _ in range(patch_n):
        top = np.random.randint(0, h-new_h)
        left = np.random.randint(0, w-new_w)
        patch_input_img = full_input_img[top:top+new_h, left:left+new_w]
        patch_target_img = full_target_img[top:top+new_h, left:left+new_w]
        patch_input_imgs.append(patch_input_img)
        patch_target_imgs.append(patch_target_img)
    return np.array(patch_input_imgs), np.array(patch_target_imgs)

def get_loader(mode, load_mode, augment,
               saved_path,test_patient,patch_n, patch_size,
               transform, batch_size, num_workers):
    dataset_ = ct_dataset(mode, load_mode, augment, saved_path, test_patient, patch_n, patch_size, transform)
    # shuffle将序列的所有元素随机排序
    data_loader = DataLoader(dataset=dataset_, batch_size=batch_size, shuffle=(True if mode=='Train' else False), num_workers=num_workers)   # shuffle=True
    return data_loader
