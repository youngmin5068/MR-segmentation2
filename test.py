import SimpleITK as sitk
import torch
import numpy as np
import torchvision.transforms as transforms
from pydicom.pixel_data_handlers.util import apply_voi_lut
from PIL import Image
from monai.networks.nets import SwinUNETR
import pydicom
from Metrics import *


# LabelOverlapMeasuresImageFilter를 사용하여 겹침 측정
def calculate_overlap_measures(truth_image_3d, prediction_image_3d):
    
    truth_image_3d = sitk.Cast(truth_image_3d, sitk.sitkUInt16)
    prediction_image_3d = sitk.Cast(prediction_image_3d,sitk.sitkUInt16)

    overlap_measures_filter = sitk.LabelOverlapMeasuresImageFilter()
    
    # 필터에 진실 이미지와 예측 이미지 설정
    overlap_measures_filter.Execute(truth_image_3d, prediction_image_3d)
    sensitivity = overlap_measures_filter.GetTruePositives() / (overlap_measures_filter.GetTruePositives() + overlap_measures_filter.GetFalseNegatives())

    
    return overlap_measures_filter.GetDiceCoefficient(), sensitivity
    
def combine_2d_slices_to_3d_itk(slices):
    # 2D 슬라이스들을 하나의 Numpy 배열로 쌓기
    volume_array = np.stack(slices)

    volume_array = volume_array.squeeze()
    image_3d = sitk.GetImageFromArray(volume_array)
    #image_3d.CopyInformation(reference_image)
    
    return image_3d


import pandas as pd
path = "/data/raw/test/test_meta.csv"
df = pd.read_csv(path)
patients = list(set(df["PatientID"].tolist()))
patients.sort()

model_path = "/data/save_model/model_240125.pth"
net = SwinUNETR(img_size=(512,512),in_channels=1,out_channels=1,spatial_dims=2)
net.load_state_dict(torch.load(model_path))

device = "cuda:0"
net = net.to(device)

transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Resize((512,512)),
                                        #customRandomResizedCrop(SEED=idx, size=(256,256))
                                        ])

test_result = []
cols = ["PatientID", "Dice_score", "Accuracy"]
for p in patients:

    df_p = df[df["PatientID"] == p]
    df_p.sort_values(by='INPUT_PATH')

    input_img_paths = []
    target_img_paths = []
    label_list = []
    
    for index, row in df_p.iterrows():
        input_img_paths.append(row['INPUT_PATH'])
        target_img_paths.append(row['TARGET_PATH'])
        label_list.append(row['LABELS'])

    total = 0.0
    correct = 0.0
    pred_stack = []
    true_stack = []
    for img_path,target_path,label in zip(input_img_paths,target_img_paths,label_list):
        input_slice = pydicom.read_file(img_path)

        input_img = input_slice.pixel_array
        input_img = apply_voi_lut(input_img, input_slice)
        epsilon = 1e-10
        min_val = np.min(input_img)
        max_val = np.max(input_img)
        input_img = (input_img - min_val) / (max_val - min_val+epsilon)


        target_slice = pydicom.read_file(target_path)
        target_img = target_slice.pixel_array
        epsilon = 1e-10
        min_val = np.min(target_img)
        max_val = np.max(target_img)
        target_img = (target_img - min_val) / (max_val - min_val+epsilon)
        mask_thresh = np.zeros_like(target_img)
        mask_thresh[target_img > 0.5] = 1.0
        mask_thresh = transform(mask_thresh)
        true_stack.append(mask_thresh.squeeze())

        input_img = Image.fromarray(input_img)
        input_img = transform(input_img)
        input_img = input_img.unsqueeze(0)
        input_img = input_img.to(device)

        output,predict = net(input_img)

        output = torch.sigmoid(output)
        pred_thresh = torch.zeros_like(output)
        pred_thresh[output > 0.5] = 1.0
        pred_thresh = pred_thresh.squeeze()
        pred_stack.append(pred_thresh.cpu().detach().numpy())

        
        predict = torch.sigmoid(predict)

        label_thresh = torch.zeros_like(predict)
        label_thresh[predict > 0.5] = 1.0
        
        correct += (label_thresh == label).sum().item()
    accuracy = correct / len(target_img_paths)
    
    pred_volume = combine_2d_slices_to_3d_itk(pred_stack)
    true_volume = combine_2d_slices_to_3d_itk(true_stack)
    dice,sensitivity = calculate_overlap_measures(true_volume, pred_volume)

    test_result.append([p, dice,accuracy])
    test_df = pd.DataFrame(test_result, columns=cols)
    test_df.to_csv("/data/raw/test/test_result.csv", index=False)
    print("patient : {}, dice : {:.4f}, sensitivity, {:.4f}, acc : {:.4f} ".format(p, dice, sensitivity, accuracy))

        
