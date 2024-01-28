import os
import torch
import random
import numpy as np
from Metrics import *
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import logging
import torch.optim as optim
from dataset import tumor_dataset
import albumentations as A
from loss import DiceLoss
from config import *
from model import UNet
import pandas as pd
from albumentations.pytorch import ToTensorV2
from monai.networks.nets import SwinUNETR
from nested_unet import UNetPlusPlus


def main(net,device,train_csv,valid_csv):

    cols = ["Epoch","Dice_loss","Dice_score","BCE_loss","Accuracy"]
    
    # dir_checkpoint = "/data/save_model"

    train_transform = A.Compose([
        A.Resize(512,512),
        A.HorizontalFlip(p=0.5), 
        A.VerticalFlip(p=0.5), 
        #A.RandomRotate90(p=0.5),
        ToTensorV2(),
    ])

    valid_transform = A.Compose([
        A.Resize(512,512),
        ToTensorV2()
    ])


    train_dataset = tumor_dataset(csv_path=train_csv,
                                    transform = True,
                                )
    val_dataset = tumor_dataset(csv_path=valid_csv,
                          transform = False,
                         )
    
    train_loader = DataLoader(train_dataset, num_workers=12, batch_size=BATCH_SIZE, shuffle=True,pin_memory=True)
    val_loader =  DataLoader(val_dataset, num_workers=12, batch_size=BATCH_SIZE, shuffle=False,pin_memory=True)
    
    logging.basicConfig(level=logging.INFO)

    logging.info(f'''Starting training:
        Epochs:          {EPOCH}
        Batch size:      {BATCH_SIZE}
        Train size:      {len(train_dataset)}
        Tuning size:     {len(val_dataset)}
        Learning rate:   {LEARNING_RATE}        
        Device:          {device}
    ''')

    optimizer = optim.AdamW(net.parameters(),betas=(0.9,0.999),lr=LEARNING_RATE,weight_decay=1e-5) # weight_decay : prevent overfitting
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=10,T_mult=1,eta_min=0.0001,last_epoch=-1)
    diceloss = DiceLoss()
    bceloss = nn.BCEWithLogitsLoss()
    
    best_dice = 0.0
    best_epoch = 1
    
    results = []
    alpha=0.8
    for epoch in range(EPOCH):
        net.train()
        
        i=1
        for imgs, true_masks,_ in train_loader:
            imgs = imgs.to(device=device, dtype=torch.float32)
            true_masks = true_masks.to(device=device,dtype=torch.float32)
            

            optimizer.zero_grad()

            #label_pred
            masks_preds = net(imgs)

            loss = diceloss(torch.sigmoid(masks_preds),true_masks)

            loss.backward()
            optimizer.step()

            nn.utils.clip_grad_value_(net.parameters(), 0.1)     

            if i*BATCH_SIZE%500 == 0:
                print('epoch : {}, index : {}/{},total loss: {:.4f}'.format(
                                                                                epoch+1, 
                                                                                i*BATCH_SIZE,
                                                                                len(train_dataset),
                                                                                loss.detach())
                                                                                ) 
            i += 1

        del imgs
        del true_masks

        with torch.no_grad():
            print("--------------Validation start----------------")
            net.eval()      

            dice_loss = 0.0
            bce_loss = 0.0
            dice = 0.0
            total = 0.0
            correct = 0.0
            for imgs, true_masks,_ in val_loader:
                imgs = imgs.to(device=device,dtype=torch.float32)
                true_masks = true_masks.to(device=device,dtype=torch.float32)
                

                #label_pred
                mask_pred = net(imgs)
                loss1 = diceloss(torch.sigmoid(mask_pred), true_masks)

                dice_loss += loss1.item()

                mask_pred = torch.sigmoid(mask_pred)
                threshold = torch.zeros_like(mask_pred)
                threshold[mask_pred>0.5] = 1.0
                
                dice += dice_score(threshold, true_masks)
            print("dice sum : {:.4f}, length : {}".format(dice,len(val_loader)))

            mean_dice_score = dice/len(val_loader)
            mean_dice_loss = dice_loss/len(val_loader)

            mean_bce_loss,acc =0,0

            results.append([epoch+1,mean_dice_loss,mean_dice_score.item(),mean_bce_loss,acc])
            df = pd.DataFrame(results,columns=cols)
            df.to_csv(f"/data/youngmin/UNetPlusPlus_result.csv",index=False)


            print("current dice : {:.4f}".format(mean_dice_score))

            if mean_dice_score > best_dice:
                print("UPDATE dice, loss")
                best_epoch = epoch
                best_dice = mean_dice_score

                try:
                    os.mkdir(DIR_CHECKPOINT)
                    logging.info("Created checkpoint directory")
                except OSError:
                    pass
                torch.save(net.module.state_dict(), DIR_CHECKPOINT + f'/UNetPlusPlus_result.pth') # d-> custom_UNet_V2 /// e -> att swinUNetr  ////f -> custom unet v2
                logging.info(f'Checkpoint {epoch + 1} saved !')

            print("best epoch : {}, best dice : {:.4f}".format(best_epoch+1,best_dice))
        #scheduler.step()




 
if __name__ == "__main__":

    def set_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(seed)
        random.seed(seed)

    set_seed(MODEL_SEED)
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    torch.backends.cudnn.benchmark = True
    

    device = torch.device(f'cuda:0' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    #net = att_UNet(1,1)c
    #net = UNet(1,1)
    net = UNetPlusPlus(1,1)
    if torch.cuda.device_count() > 1:
        print("Let's use mutli GPUs!")
        net = nn.DataParallel(net,device_ids=[0,1,2,3,4,5]) 
    net.to(device=device)
    main(net=net,device=device,train_csv="/data/raw/train/train_meta_final.csv",valid_csv="/data/raw/tuning/tuning_meta_final.csv")

    # np.save("/workspace/IITP/task_3D/dir_checkpoint/swinunetr_losses.npy",losses.cpu().numpy())
    # np.save("/workspace/IITP/task_3D/dir_checkpoint/swinunetr_dices.npy",dices.cpu().numpy())