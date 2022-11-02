import os
import torch
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') #GPU 할당

CFG = {
    'IMG_SIZE':32, #이미지 사이즈
    'EPOCHS':50, #에포크
    'LEARNING_RATE':2e-2, #학습률
    'BATCH_SIZE':12, #배치사이즈
    'SEED':41, #시드
}

labels = []
f = open("/gdrive/MyDrive/grad_proj/labels.txt", 'r', encoding='utf-8')
while True:
    line = f.readline()
    if not line: break
    labels.append(line[-2])
f.close()
print(labels[0], labels[1])

def invert_dictionary(obj):
  return {value: key for key, value in obj.items()}

word_to_num = dict()

f = open("/gdrive/MyDrive/grad_proj/words/ko.txt", 'r', encoding='utf-8')
index = 0
while True:
    line = f.readline()
    if not line: break
    word_to_num[line[:-1]] = index
    index = index + 1
f.close()

num_to_word = invert_dictionary(word_to_num)

from glob import glob


def get_train_data(data_dir):
    img_path_list = []
    label_list = []

    # get image path
    img_path_list.extend(glob(os.path.join(data_dir, '*.jpg')))
    img_path_list.sort(key=lambda x: int(x.split('/')[-1].split('.')[0]))
    label_list = list(map(lambda path: word_to_num[labels[int(path.split('/')[-1].split('.')[0])]], img_path_list))
    return img_path_list, label_list


def get_test_data(data_dir):
    img_path_list = []

    # get image path
    img_path_list.extend(glob(os.path.join(data_dir, '*.jpg')))
    img_path_list.sort(key=lambda x: int(x.split('/')[-1].split('.')[0]))

    return img_path_list

data_dir = "/gdrive/MyDrive/grad_proj/img/train"
all_img_path, all_label = get_train_data(data_dir)
test_img_path = get_test_data("/gdrive/MyDrive/grad_proj/img/test")

import torchvision.datasets as datasets  # 이미지 데이터셋 집합체
import torchvision.transforms as transforms  # 이미지 변환 툴

from torch.utils.data import DataLoader  # 학습 및 배치로 모델에 넣어주기 위한 툴
from torch.utils.data import DataLoader, Dataset

class CustomDataset(Dataset):
    def __init__(self, img_path_list, label_list, train_mode=True, transforms=None):  # 필요한 변수들을 선언
        self.transforms = transforms
        self.train_mode = train_mode
        self.img_path_list = img_path_list
        self.label_list = label_list

    def __getitem__(self, index):  # index번째 data를 return
        img_path = self.img_path_list[index]
        # Get image data
        image = cv2.imread(img_path)
        if self.transforms is not None:
            image = self.transforms(image)

        if self.train_mode:
            label = self.label_list[index]
            return image, label
        else:
            return image

    def __len__(self):  # 길이 return
        return len(self.img_path_list)

# Train : Validation = 0.8 : 0.25 Split
train_len = int(len(all_img_path)*0.8)
vali_len = int(len(all_img_path)*0.2)

train_img_path = all_img_path[:train_len]
train_label = all_label[:train_len]

vali_img_path = all_img_path[train_len:]
vali_label = all_label[train_len:]

train_transform = transforms.Compose([
    transforms.ToPILImage(),  # Numpy배열에서 PIL이미지로
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize([CFG['IMG_SIZE'], CFG['IMG_SIZE']]),  # 이미지 사이즈 변형
    transforms.ToTensor(),  # 이미지 데이터를 tensor
    transforms.Normalize(mean=0.5, std=0.5)  # 이미지 정규화

])

test_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize([CFG['IMG_SIZE'], CFG['IMG_SIZE']]),
    transforms.ToTensor(),
    transforms.Normalize(mean=0.5, std=0.5)
])
# Get Dataloader

#CustomDataset class를 통하여 train dataset생성
train_dataset = CustomDataset(train_img_path, train_label, train_mode=True, transforms=train_transform)
#만든 train dataset를 DataLoader에 넣어 batch 만들기
train_loader = DataLoader(train_dataset, batch_size = CFG['BATCH_SIZE'], shuffle=True, num_workers=0) #BATCH_SIZE : 24

#vaildation 에서도 적용
vali_dataset = CustomDataset(vali_img_path, vali_label, train_mode=True, transforms=test_transform)
vali_loader = DataLoader(vali_dataset, batch_size = CFG['BATCH_SIZE'], shuffle=False, num_workers=0)

train_batches = len(train_loader)
vali_batches = len(vali_loader)

print('total train imgs :',train_len,'/ total train batches :', train_batches)
print('total valid imgs :',vali_len, '/ total valid batches :', vali_batches)

from tqdm.auto import tqdm
import torch.nn.functional as F

import torch.nn as nn  # 신경망들이 포함됨


# import torch.nn.init as init # 텐서에 초기값을 줌

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
        x = self.layer1(x)  # 1층

        x = self.layer2(x)  # 2층

        x = self.layer3(x)  # 3층

        x = self.layer4(x)  # 4층

        x = self.layer5(x)  # 5층

        x = self.layer6(x)  # 6층

        x = self.layer7(x)  # 7층

        x = self.layer8(x)  # 8층

        x = torch.flatten(x, start_dim=1)  # N차원 배열 -> 1차원 배열

        out = self.fc_layer(x)
        return out

import torch.optim as optim # 최적화 알고리즘들이 포함힘

model = CNNclassification().to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params = model.parameters(), lr = CFG["LEARNING_RATE"])
scheduler = None


def train(model, optimizer, train_loader, scheduler, device):
    model.to(device)
    n = len(train_loader)

    # Loss Function 정의
    criterion = nn.CrossEntropyLoss().to(device)
    best_acc = 0

    for epoch in range(1, CFG["EPOCHS"] + 1):  # 에포크 설정
        model.train()  # 모델 학습
        running_loss = 0.0

        for img, label in tqdm(iter(train_loader)):
            img, label = img.to(device), torch.tensor(label).to(device)  # 배치 데이터
            optimizer.zero_grad()  # 배치마다 optimizer 초기화

            # Data -> Model -> Output
            logit = model(img)  # 예측값 산출
            loss = criterion(logit, label)  # 손실함수 계산

            # 역전파
            loss.backward()  # 손실함수 기준 역전파
            optimizer.step()  # 가중치 최적화
            running_loss += loss.item()

        print('[%d] Train loss: %.10f' % (epoch, running_loss / len(train_loader)))

        if scheduler is not None:
            scheduler.step()

        # Validation set 평가
        model.eval()  # evaluation 과정에서 사용하지 않아야 하는 layer들을 알아서 off 시키도록 하는 함수
        vali_loss = 0.0
        correct = 0
        with torch.no_grad():  # 파라미터 업데이트 안하기 때문에 no_grad 사용
            for img, label in tqdm(iter(vali_loader)):
                img, label = img.to(device), torch.tensor(label).to(device)

                logit = model(img)
                vali_loss += criterion(logit, label)
                pred = logit.argmax(dim=1, keepdim=True)  # 1001개의 class중 가장 값이 높은 것을 예측 label로 추출
                correct += pred.eq(label.view_as(pred)).sum().item()  # 예측값과 실제값이 맞으면 1 아니면 0으로 합산
        vali_acc = 100 * correct / len(vali_loader.dataset)
        print('Vail set: Loss: {:.4f}, Accuracy: {}/{} ( {:.0f}%)\n'.format(vali_loss / len(vali_loader), correct,
                                                                            len(vali_loader.dataset),
                                                                            100 * correct / len(vali_loader.dataset)))

        # 베스트 모델 저장
        if best_acc < vali_acc:
            best_acc = vali_acc
            torch.save(model.state_dict(),
                       '/gdrive/pMyDrive/grad_proj/saved/best_model.pth')  # 이 디렉토리에 best_model.pth을 저장
            print('Model Saved.')

train(model, optimizer, train_loader, scheduler, device)

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

def predict_test(model, test_loader, device):
    model.eval()
    model_pred = []
    test_imgs = []
    with torch.no_grad():
        for img in tqdm(iter(test_loader)):
            test_imgs.append(img)
            img = img.to(device)

            pred_logit = model(img)
            pred_logit = pred_logit.argmax(dim=1, keepdim=True).squeeze(1)

            model_pred.extend(pred_logit.tolist())
    return model_pred, test_imgs

