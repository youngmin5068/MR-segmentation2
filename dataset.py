import torch
import glob
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import nibabel as nib
import os
import numpy as np
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
import torchio as tio
from albumentations.pytorch import ToTensorV2
import torchvision.transforms as transforms
from custom_transforms import *
from monai.transforms import AsDiscrete

class tumor_dataset(Dataset):
    def __init__(self, csv_path, transform=None):

        self.csv_path = csv_path
        self.transforms = transform
        self.threshold = AsDiscrete(threshold=0.5)
        
        df = pd.read_csv(self.csv_path)

        # df_true = df[df["LABELS"] == 1]
        # df_false = df[df["LABELS"] == 0].sample(n=len(df_true))
        # half_false = df_false.sample(frac=0.7,random_state=42)
        
        # df = pd.concat([df_true,half_false]).reset_index(drop=True)
        # df = df.sample(frac=1.0,random_state=42)
        
        self.input_list = []
        self.mask_list = []
        self.label_list = []

        for index, row in df.iterrows():
            self.input_list.append(row['INPUT_PATH'])
            self.mask_list.append(row['TARGET_PATH'])
            self.label_list.append(row['LABELS'])

    def __len__(self):
        if len(self.input_list) == len(self.mask_list):
            return len(self.mask_list)
        else:
            return "error"
        
    def preprocess(self,train_path,mask_path):
        
        input_slice = pydicom.read_file(train_path)

        input_img = input_slice.pixel_array
        input_img = apply_voi_lut(input_img, input_slice)
        epsilon = 1e-10
        min_val = np.min(input_img)
        max_val = np.max(input_img)
        input_img = (input_img - min_val) / (max_val - min_val+epsilon)
        

        target_slice = pydicom.read_file(mask_path)
        target_img = target_slice.pixel_array
        epsilon = 1e-10
        min_val = np.min(target_img)
        max_val = np.max(target_img)
        target_img = (target_img - min_val) / (max_val - min_val+epsilon)


        input_img = Image.fromarray(input_img)
        target_img = Image.fromarray(target_img)


        return input_img, target_img



    def __getitem__(self, idx):

        input_path = self.input_list[idx]
        target_path = self.mask_list[idx]
        label = self.label_list[idx]

        input_img,mask_img = self.preprocess(input_path,target_path)

       # mask_img = np.where(mask_img > 0.5, 1, 0)

        # if self.transforms:
        #     transformed = self.transforms(image=input_img, mask = mask_img)
        #     input_img = transformed['image']
        #     mask_img = transformed['mask']
        
        
        if self.transforms:
            transform = transforms.Compose([transforms.ToTensor(),
                                                transforms.Resize((512,512)),
                                                customRandomRotate(degrees=10,SEED=idx),
                                                customRandomHorizontalFlip(p=0.5,SEED=idx),
                                                customRandomVerticalFlip(p=0.5,SEED=idx)
                                                #customRandomResizedCrop(SEED=idx, size=(256,256))
                                                ])
        else:
            transform = transforms.Compose([transforms.ToTensor(),
                                                    transforms.Resize((512,512)),
                                                    #customRandomResizedCrop(SEED=idx, size=(256,256))
                                                    ])
        input_img = transform(input_img)
        mask_img = transform(mask_img)

        # thresh = torch.zeros_like(mask_img)
        # thresh[mask_img > 0.5] = 1.0            

        thresh = self.threshold(mask_img)

        return input_img,thresh,label


if __name__ == '__main__':
    import albumentations as A
    train_transform = A.Compose([
        A.Resize(512,512),
        A.HorizontalFlip(p=0.5), 
        A.VerticalFlip(p=0.5), 
        A.RandomRotate90(p=0.5),
        ToTensorV2(),
    ])
    dataset = tumor_dataset(csv_path="/data/raw/train/train_meta_except.csv",transform=True)
    train_loader = DataLoader(dataset=dataset, batch_size=64, shuffle=True, num_workers=0)
    sample = next(iter(train_loader))
    print(len(dataset))
    print(sample[0].shape,sample[1].shape)