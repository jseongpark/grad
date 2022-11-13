import os
import torch
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') #GPU 할당

print("###")
CFG = {
    'IMG_SIZE':32, #이미지 사이즈
    'EPOCHS':50, #에포크
    'LEARNING_RATE':2e-2, #학습률
    'BATCH_SIZE':12, #배치사이즈
    'SEED':41, #시드
}

from glob import glob

def get_test_data(data_dir):
    img_path_list = []
    
    # get image path
    img_path_list.extend(glob(os.path.join(data_dir, '*.jpg')))
    img_path_list.sort(key=lambda x:int(x.split('/')[-1].split('.')[0]))
    
    return img_path_list

data_dir = "C:\grad/output/test"
test_img_path = get_test_data(data_dir)

import torchvision.datasets as datasets # 이미지 데이터셋 집합체
import torchvision.transforms as transforms # 이미지 변환 툴

from torch.utils.data import DataLoader # 학습 및 배치로 모델에 넣어주기 위한 툴
from torch.utils.data import DataLoader, Dataset


import torch.nn.functional as F

import torch.nn as nn # 신경망들이 포함됨
#import torch.nn.init as init # 텐서에 초기값을 줌

class CNNclassification(nn.Module):
    def __init__(self):
        super(CNNclassification, self).__init__()
        self.layer1 = nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1)

        self.layer2 = nn.Conv2d(8, 32, kernel_size=3, stride=1, padding=1)
        
        self.layer3 = torch.nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2))
        
        self.layer4 = torch.nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2))
        
        self.layer5 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)

        self.layer6 = torch.nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2)) 

        self.layer7 = torch.nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())

        self.layer8 = torch.nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1), 
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.fc_layer = nn.Sequential( 
            nn.Linear(2048, 1030)
        )    

    def forward(self, x):
        
        x = self.layer1(x) #1층
        
        x = self.layer2(x) #2층
         
        x = self.layer3(x) #3층
        
        x = self.layer4(x) #4층

        x = self.layer5(x) #5층

        x = self.layer6(x) #6층

        x = self.layer7(x) #7층

        x = self.layer8(x) #8층

        x = torch.flatten(x, start_dim=1) # N차원 배열 -> 1차원 배열
        
        out = self.fc_layer(x)
        return out

def predict(model, test_loader, device):
    model.eval()
    model_pred = []
    with torch.no_grad():
        for img in tqdm(iter(test_loader)):
            img = img.to(device)

            pred_logit = model(img)
            pred_logit = pred_logit.argmax(dim=1, keepdim=True).squeeze(1)

            model_pred.extend(pred_logit.tolist())
    return model_pred

checkpoint = torch.load('C:\grad/saved/best_model.pth')
model = CNNclassification().to(device)
model.load_state_dict(checkpoint)

test_dataset = CustomDataset(test_img_path, None, train_mode=False, transforms=test_transform)
test_loader = DataLoader(test_dataset, batch_size = 20, shuffle=False, num_workers=0)
preds = predict(model, img, device)
print(preds)