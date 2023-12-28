import os
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.cuda.amp import autocast, GradScaler
from torch.optim import lr_scheduler

import numpy as np
from PIL import Image, ImageFile
from torch.utils.data import Dataset, DataLoader
import json
from time import sleep
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import random
import sys
import matplotlib.ticker as plticker
from datetime import datetime

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torchvision.ops import box_iou 

import torch.nn.functional as F
from torch.utils.data import IterableDataset

class COCODatasetOD(Dataset):
    def __init__(self, filepath, device="cpu", image_size=416, cells=10, Shuffle=False, testdataset=False):
        if testdataset:
            self.apath = os.path.join(filepath, "annotations", "instances_val2017.json")
            self.ipath = os.path.join(filepath, "val2017")
        else:
            self.apath = os.path.join(filepath, "annotations", "instances_train2017.json")
            self.ipath = os.path.join(filepath, "train2017")
        
        self.image_size = (image_size, image_size)
        self.cells = cells
        self.cell_size = 1 / self.cells
        self.classes = 10
        self.device = device
        
        with open(self.apath, "r") as f:
            data = json.load(f)
            newdata = dict()

            for image in data['images']:
                image_id = image['id']
                if image_id not in newdata:
                    newdata[image_id] = image
            for annotation in data['annotations']:
                image_id = annotation['image_id']
                classification = annotation['category_id']

                if image_id in newdata:
                    if 'annotations' in newdata[image_id] and classification <= self.classes: # only get the first 10 classes
                        newdata[image_id]['annotations'].append(annotation)
                    elif classification <= 10 and 'annotations' not in newdata[image_id]:
                        newdata[image_id]['annotations'] = [annotation]
            self.annotations = list(dict({key: newdata[key] for key in newdata if 'annotations' in newdata[key]}).values())
            if Shuffle:
                random.shuffle(self.annotations)
                
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, index):
        annotation = self.annotations[index]
        image_width, image_height = annotation['width'], annotation['height']
        
        #Load the image
        image = Image.open(os.path.join(self.ipath, annotation['file_name'])).convert('RGB').resize(self.image_size)
        
        #Prepare the annotations
        processed_annotations = np.zeros((self.cells, self.cells, self.classes+5))
        
        for a in annotation['annotations']:
            bbox = a['bbox'] # (x, y, w, h)
            rbbox = [bbox[0]/image_width, bbox[1]/image_height, bbox[2]/image_width, bbox[3]/image_height] #relative position
            
            mx, my = (rbbox[0]+(rbbox[0]+rbbox[2]))/2, (rbbox[1]+(rbbox[1]+rbbox[3]))/2
            rcellX, rcellY = mx/self.cell_size, my/self.cell_size
            rcellw, rcellh = rbbox[2]/self.cell_size, rbbox[3]/self.cell_size
            
            posX, posY = int(np.floor(rcellX)), int(np.floor(rcellY))
            offsetX, offsetY = rcellX % 1, rcellY % 1
            
            ohe_classes = np.zeros(self.classes)
            ohe_classes[a['category_id'] - 1] = 1
            
            # [x_position, y_position, width, height, confidence... one hot encoded classes confidence]
            insert = np.concatenate(([offsetX, offsetY, rcellw, rcellh, 1], ohe_classes))
            processed_annotations[posX][posY] = insert
            
            flattened_annotations = processed_annotations.reshape(processed_annotations.shape[0]*processed_annotations.shape[1]*processed_annotations.shape[2])
            
            totensor = T.ToTensor()
            
        return totensor(image).to(self.device), torch.tensor(flattened_annotations.astype(float), dtype=torch.float).to(self.device)

def displayImageWithLabels(img, label):
    fig, ax = plt.subplots(1)

    image = np.transpose(img, (1, 2, 0))
    ax.imshow(image)
    
    for x in range(len(label)):
        for y in range(len(label[x])):
            if label[x][y][0] != 0:
                cell_size_x, cell_size_y = len(image) / len(label), len(image[0]) / len(label[0])
                mx, my = ((x)*cell_size_x)+(cell_size_x*label[x][y][0]), ((y)*cell_size_y)+(cell_size_y*label[x][y][1])
                w, h = (cell_size_x*label[x][y][2]), (cell_size_y*label[x][y][3])

                circle = patches.Circle((mx,my), 2, color="orange")
                rect = patches.Rectangle(((mx-(w/2)),(my-(h/2))), w, h, edgecolor='r', facecolor="None")
                ax.add_patch(circle)            
                ax.add_patch(rect)

class YOLOLoss(nn.Module):
    def __init__(self, cells, classes):
        super(YOLOLoss, self).__init__()
        self.cells = cells
        self.classes = classes
        
    def forward(self, output, target): 
        cel = nn.CrossEntropyLoss()
        totalLoss = 0
        
        for batch in range(len(output)):
            for index in range(self.cells**2):
                if torch.is_nonzero(output[batch][index*15]):
                    #coordinate loss
                    coordinate_loss = ((output[batch][index*15:(index*15)+1] - target[batch][index*15:(index*15)+1])**2) + (((torch.sqrt(output[batch][(index*15)+2]) - torch.sqrt(target[batch][(index*15)+2]))**2)+((torch.sqrt(output[batch][(index*15)+3]) - torch.sqrt(target[batch][(index*15)+3]))**2))
                    
                    #confidence loss
                    confidence_loss = -(target[batch][(index*15)+4] * torch.log(output[batch][(index*15)+4] + 1e-9) + (1 - target[batch][(index*15)+4]) * torch.log(1 - output[batch][(index*15)+4] + 1e-9))

                    #class loss
                    class_loss = torch.sum(torch.sub( target[batch][(index*15)+5:(index*15)+14], output[batch][(index*15)+5:(index*15)+14] )**2)
                    #print(coordinate_loss, confidence_loss, class_loss)
                    totalLoss += (coordinate_loss + confidence_loss + class_loss)
        return totalLoss

class YOLOActivation(nn.Module):
    def __init__(self, classes, cells=7):
        super(YOLOActivation, self).__init__()
        self.classes = classes
        self.cells = cells
        self.softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, input_):
        for batch in range(len(input_)):
            for i in range((self.cells**2)-1):
                input_[batch][(i*(self.classes+5))+5:(i*(self.classes+5))+5+self.classes] = self.softmax(input_[batch][(i*(self.classes+5))+5:(i*(self.classes+5))+5+self.classes])
                input_[batch][(i*(self.classes+5)):(i*(self.classes+5))+2] = self.sigmoid(input_[batch][(i*(self.classes+5)):(i*(self.classes+5))+2])
                input_[batch][(i*(self.classes+5))+4] = self.sigmoid(input_[batch][(i*(self.classes+5))+4])
        return input_     

class YOLOv1(nn.Module):
    def __init__(self, cells=7, classes=10):
        super(YOLOv1, self).__init__()
        self.normalisation = nn.BatchNorm2d(3)
        self.dropout = nn.Dropout(p=0.5)
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2)
        self.max1 = nn.MaxPool2d(2, stride=2)
            
        self.conv2 = nn.Conv2d(64, 192, 3)
        self.max2 = nn.MaxPool2d(2, stride=2)
            
        self.conv3 = nn.Conv2d(192, 128, 1)
        self.conv4 = nn.Conv2d(128, 256, 3)
        self.conv5 = nn.Conv2d(256, 256, 1)
        self.conv6 = nn.Conv2d(256, 512, 3)
        self.max3 = nn.MaxPool2d(2, stride=2)
            
        self.conv7 = nn.Conv2d(512, 256, 1)
        self.conv8 = nn.Conv2d(256, 512, 3)
        self.conv9 = nn.Conv2d(512, 256, 1)
        self.conv10 = nn.Conv2d(256, 512, 3)
        self.conv11 = nn.Conv2d(512, 512, 1)
        self.conv12 = nn.Conv2d(512, 1024, 3)
        self.max4 = nn.MaxPool2d(2, stride=2)
            
        self.conv13 = nn.Conv2d(1024, 512, 1)
        self.conv14 = nn.Conv2d(512, 1024, 3)
        self.conv15 = nn.Conv2d(1024, 512, 1)
        self.conv16 = nn.Conv2d(512, 1024, 3)
        self.conv17 = nn.Conv2d(1024, 1024, 3)
        self.conv18 = nn.Conv2d(1024, 1024, 3, stride=2)
        self.flatten = nn.Flatten()
            
        self.linear1 = nn.Linear(1024, 4096)
        self.linear2 = nn.Linear(4096, 7*7*(classes+5))
        self.yoloactivation = YOLOActivation(cells=cells, classes=classes)
        
    def forward(self, x):
        x = self.normalisation(x)
        x = self.dropout(x)
        
        x = self.conv1(x)
        x = self.max1(x)
        
        x = self.conv2(x)
        x = self.max2(x)
        
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.max3(x)
        
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        x = self.conv11(x)
        x = self.conv12(x)
        x = self.max4(x)
        
        x = self.conv13(x)
        x = self.conv14(x)
        x = self.conv15(x)
        x = self.conv16(x)
        x = self.conv17(x)
        x = self.conv18(x)
        
        x = self.flatten(x)
        
        x = self.dropout(x)
        
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.yoloactivation(x)
        
        return x

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    train_ds = COCODatasetOD("./coco2017", Shuffle=False, cells=7, image_size=448, device=device)
    test_ds = COCODatasetOD("./coco2017", Shuffle=False, cells=7, image_size=448, testdataset=True, device=device)

    train_dataloader = DataLoader(dataset=train_ds, batch_size=5, shuffle=True, num_workers=4)
    test_dataloader = DataLoader(dataset=test_ds, batch_size=5, shuffle=True, num_workers=4)
    
    # define the model
    model = YOLOv1().to(device)

    # define the loss function and optimizer
    loss_fn = YOLOLoss(cells=7, classes=10)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    # defining Tensorboard writer
    writer = SummaryWriter('data/runs')

    # training specific parameters
    num_epochs = 50

    # training function
    for epoch in range(num_epochs):
        running_loss = 0
        running_accuracy = 0
        starttime = datetime.now()

        print(f"COMMENCING EPOCH {epoch+1}")
        training_data = tqdm(train_dataloader, leave=True)

        for batch_idx, (images, annotations) in enumerate(training_data):
            model.train(True)
            outputs = model(images)

            loss = loss_fn(outputs, annotations)
            running_loss += loss.item()

            running_accuracy += torch.mean(torch.sub(annotations, outputs)).item()

            for param in model.parameters():
                param.grad = None

            loss.backward()
            optimizer.step()

        model.train(False)
        model.eval()

        testing_loss = 0
        testing_accuracy = 0
        # evaluate testing data
        for i, (img, ann) in enumerate(test_dataloader):
            if i < 10:
                test_output = model(img)
                test_loss = loss_fn(test_output, ann)
                testing_accuracy += torch.mean(torch.sub(ann, test_outputs)).item()
            else:
                testing_loss = testing_loss / 10
                testing_accuracy = testing_accuracy / 10
                break

        print (f'Epoch [{epoch+1}/{num_epochs}], Average Loss: {running_loss/len(training_data)}, Average Accuracy: {running_accuracy/len(training_data)}')
        writer.add_scalar('training loss', running_loss/len(training_data), epoch+1)
        writer.add_scalar('training accuracy', running_accuracy/len(training_data), epoch+1)
        writer.add_scalar('testing loss', testing_loss, epoch+1)
        writer.add_scalar('testing accuracy', testing_accuracy, epoch+1)

        scheduler.step()
        torch.save(model.state_dict(f"data/saves/{starttime}_{datetime.now()}_EPOCH{epoch+1}"))   

if __name__ == '__main__':
    main()