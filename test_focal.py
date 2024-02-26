import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from model import UNET
from utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs,
)
import os 
from dataset import Dataset
from metrics import *
import pandas as pd
from model import model_smp, model_unet,model_dunet, preprocessing_fn
import segmentation_models_pytorch as smp
from utils import * 
import sys
import torch
from tqdm import tqdm as tqdm
import matplotlib.pyplot as plt
import cv2
import numpy as np
from torch.nn import functional as F


DATA_DIR = 'Dataset_1000'
this_dir_path = os.path.abspath(os.getcwd())


x_train_dir = os.path.join(DATA_DIR , 'images')
y_train_dir = os.path.join(DATA_DIR , 'masks')
x_train_dir = f"{this_dir_path}/rgb_images"
y_train_dir = f"{this_dir_path}/semantic_annotations/rgbLabels"
x_valid_dir = os.path.join(DATA_DIR, 'rgb252')
y_valid_dir = os.path.join(DATA_DIR, 'mask252')
x_test_dir = os.path.join(DATA_DIR , 'test50_rgb')
y_test_dir = os.path.join(DATA_DIR , 'test50_mask')

print(len(os.listdir(x_train_dir)))
print(len(os.listdir(y_train_dir)))
print(len(os.listdir(x_valid_dir)))
print(len(os.listdir(y_valid_dir)))
print(len(os.listdir(x_test_dir)))
print(len(os.listdir(y_test_dir)))


print(x_test_dir)

class_dict = pd.read_csv("label_class_dict.csv")
# Get class names
class_names = class_dict['name'].tolist()
# Get class RGB values
class_rgb_values = class_dict[['r','g','b']].values.tolist()
# class_rgb_dict = {cls: rgb for cls, rgb in zip(class_names, class_rgb_values)}
class_rgb_dict = {cls_name: rgb for cls_name, rgb in zip(class_names, class_rgb_values)}

print('All dataset classes and their corresponding RGB values in labels:')
print('Class Names: ', class_names)
print('Class RGB values: ', class_rgb_values)


# CLASSES = [ 'road', 'lanemarks', 'curb', 'person', 'rider', 'vehicles', 'bicycle', 'motorcycle', 'traffic sign']
CLASSES = [ 'background','road', 'lanemarks', 'curb', 'person', 'rider', 'vehicles', 'bicycle', 'motorcycle', 'traffic sign']


# class_colors_bgr = {
#     "background": [0, 0, 0],        # Black
#     "road": [255, 0, 255],          # Magenta
#     "lanemarks": [255, 0, 0],       # Blue
#     "curb": [0, 255, 0],            # Green
#     "person": [0, 0, 255],          # Red
#     "rider": [255, 255, 255],       # White
#     "vehicles": [255, 255, 0],      # Cyan
#     "bicycle": [0, 255, 255],       # Yellow
#     "motorcycle": [128, 128, 255],  # Lighter Red/Rose
#     "traffic sign": [0, 128, 128]   # Teal
# }




class_colors_bgr = [
    [0,0,0],         # background
    [255, 0, 255],   # road
    [255, 0, 0],     # lanemarks
    [0, 255, 0],     # curb
    [0, 0, 255],     # person
    [255, 255, 255], # rider
    [255, 255, 0],   # vehicles
    [0, 255, 255],   # bicycle
    [128, 128, 255], # motorcycle
    [0, 128, 128]    # traffic sign
]

# Convert BGR to RGB
class_colors_rgb = [list(reversed(color)) for color in class_colors_bgr]

from torch.utils.data import DataLoader


import numpy as np
 
 
class Meter(object):
    """Meters provide a way to keep track of important statistics in an online manner.
    This class is abstract, but provides a standard interface for all meters to follow.
    """
 
    def reset(self):
        """Reset the meter to default settings."""
        pass
 
    def add(self, value):
        """Log a new value to the meter
        Args:
            value: Next result to include.
        """
        pass
 
    def value(self):
        """Get the value of the meter in the current state."""
        pass
 
 
class AverageValueMeter(Meter):
    def __init__(self):
        super(AverageValueMeter, self).__init__()
        self.reset()
        self.val = 0
 
    def add(self, value, n=1):
        self.val = value
        self.sum += value
        self.var += value * value
        self.n += n
 
        if self.n == 0:
            self.mean, self.std = np.nan, np.nan
        elif self.n == 1:
            self.mean = 0.0 + self.sum  # This is to force a copy in torch/numpy
            self.std = np.inf
            self.mean_old = self.mean
            self.m_s = 0.0
        else:
            self.mean = self.mean_old + (value - n * self.mean_old) / float(self.n)
            self.m_s += (value - self.mean_old) * (value - self.mean)
            self.mean_old = self.mean
            self.std = np.sqrt(self.m_s / (self.n - 1.0))
 
    def value(self):
        return self.mean, self.std
 
    def reset(self):
        self.n = 0
        self.sum = 0.0
        self.var = 0.0
        self.val = 0.0
        self.mean = np.nan
        self.mean_old = 0.0
        self.m_s = 0.0
        self.std = np.nan
 

 
class Epoch:
    def __init__(self, model, loss, metrics, stage_name, device="gpu", verbose=True):
        self.model = model
        self.loss = loss
        self.metrics = metrics
        self.stage_name = stage_name
        self.verbose = verbose
        self.device = device
 
        self._to_device()
 
    def _to_device(self):
        self.model.to(self.device)
        self.loss.to(self.device)
        for metric in self.metrics:
            metric.to(self.device)
 
    def _format_logs(self, logs):
        str_logs = ["{} - {:.4}".format(k, v) for k, v in logs.items()]
        s = ", ".join(str_logs)
        return s
 
    def batch_update(self, x, y):
        raise NotImplementedError
 
    def on_epoch_start(self):
        pass
 
    def run(self, dataloader):
 
        self.on_epoch_start()
 
        logs = {}
        loss_meter = AverageValueMeter()
        metrics_meters = {metric.__name__: AverageValueMeter() for metric in self.metrics}
 
        with tqdm(
            dataloader,
            desc=self.stage_name,
            file=sys.stdout,
            disable=not (self.verbose),
        ) as iterator:
            for x, y in iterator:
                x, y = x.to(self.device), y.to(self.device)
                loss, y_pred = self.batch_update(x, y)
 
                # update loss logs
                loss_value = loss.cpu().detach().numpy()
                loss_meter.add(loss_value)
                # loss_logs = {self.loss.__name__: loss_meter.mean}
                loss_logs = {self.loss.__class__.__name__: loss_meter.mean}
                logs.update(loss_logs)
 
                # update metrics logs
                for metric_fn in self.metrics:
                    metric_value = metric_fn(y_pred, y).cpu().detach().numpy()
                    metrics_meters[metric_fn.__name__].add(metric_value)
                metrics_logs = {k: v.mean for k, v in metrics_meters.items()}
                logs.update(metrics_logs)
 
                if self.verbose:
                    s = self._format_logs(logs)
                    iterator.set_postfix_str(s)
 
        return logs
 
 
class TrainEpoch(Epoch):
    def __init__(self, model, loss, metrics, optimizer, device="cpu", verbose=True):
        super().__init__(
            model=model,
            loss=loss,
            metrics=metrics,
            stage_name="train",
            device=device,
            verbose=verbose,
        )
        self.optimizer = optimizer
 
    def on_epoch_start(self):
        self.model.train()
        self.model.to(self.device)
 
    def batch_update(self, x, y):
        x, y = x.to(self.device), y.to(self.device)
        self.optimizer.zero_grad()
        prediction = self.model.forward(x)
        loss = self.loss(prediction, y)
        loss.backward()
        self.optimizer.step()
        return loss, prediction
 
 
class ValidEpoch(Epoch):
    def __init__(self, model, loss, metrics, device="cpu", verbose=True):
        super().__init__(
            model=model,
            loss=loss,
            metrics=metrics,
            stage_name="valid",
            device=device,
            verbose=verbose,
        )
 
    def on_epoch_start(self):
        self.model.eval()
        self.model.to(self.device)
 
    def batch_update(self, x, y):
        x, y = x.to(self.device), y.to(self.device)
        with torch.no_grad():
            prediction = self.model.forward(x)
            y_class_indices = torch.argmax(y, dim=1)
            prediction = F.softmax(prediction, dim=1)
            loss = self.loss(prediction, y_class_indices)
        return loss, prediction



train_dataset = Dataset(
    x_train_dir, 
    y_train_dir, 
    augmentation=get_training_augmentation(), 
    preprocessing=get_preprocessing(preprocessing_fn),
    classes=CLASSES,
)
 
valid_dataset = Dataset(
    x_valid_dir, 
    y_valid_dir, 
    augmentation=get_validation_augmentation(), 
    preprocessing=get_preprocessing(preprocessing_fn),
    classes=CLASSES,
)
 
# train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=1)
# valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=0)

# model_dunet = model_smp#model_unet # keep changing this 


optimizer = torch.optim.Adam([ 
    dict(params=model_dunet.parameters(), lr=0.0001),
])


loss = DiceLoss()
loss = nn.CrossEntropyLoss()


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print("here her ehere herhehehrehhrehr ehr", DEVICE)
# train_epoch = TrainEpoch(
#     model_dunet, 
#     loss=loss, 
#     metrics=metrics, 
#     optimizer=optimizer,
#     device=DEVICE,
#     verbose=True,
# )
 
# valid_epoch = ValidEpoch(
#     model_dunet, 
#     loss=loss, 
#     metrics=metrics, 
#     device=DEVICE,
#     verbose=True,
# )


max_score = 0
 
# for i in range(0, 10):
    
#     print('\nEpoch: {}'.format(i))
#     train_logs = train_epoch.run(train_loader)
#     valid_logs = valid_epoch.run(valid_loader)
    
#     # Save the model with best iou score
#     if max_score < valid_logs['iou_score']:
#         max_score = valid_logs['iou_score']
#         torch.save(model_dunet, "UnetTypes.pth")
#         print('Model saved!')
        
#     if i == 50:
#         optimizer.param_groups[0]['lr'] = 1e-5
#         print('Decrease decoder learning rate to 1e-5!')


# create test dataset
test_dataset = Dataset(
    x_test_dir, 
    y_test_dir, 
    augmentation=get_training_augmentation(), 
    preprocessing=get_preprocessing(preprocessing_fn),
    #    augmentation=None, 
#    preprocessing=get_preprocessing(preprocessing_fn),
    classes=CLASSES,
) 
 
test_dataloader = DataLoader(test_dataset,batch_size=1, shuffle=False, num_workers=0)
 
metrics = [
    IoU(threshold=0.5),
    Accuracy(threshold=0.5),
    Fscore(threshold=0.5),
    Recall(threshold=0.5),
    Precision(threshold=0.5),
]


Trained_model = torch.load('UnetTypes.pth')
 
# Evaluate model on test set
test_epoch = ValidEpoch(
    model=Trained_model,
    loss=loss,
    metrics=metrics,
    device=DEVICE,
)
 
# logs = test_epoch.run(test_dataloader)
# print(logs)
idx = np.random.choice(np.arange(len(test_dataset)))
image, gt_mask = test_dataset[idx]  
print("test image agd gt shape", image.shape, gt_mask.shape)
x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
predicted_mask = Trained_model(x_tensor)
# changes based on loss estimation 
# predicted_mask =  F.softmax(predicted_mask, dim=1)

gt_mask_copy = gt_mask.copy()
gt_mask_copy_for_compute = gt_mask_copy.copy()

# pr_mask = (predicted_mask.squeeze().cpu().numpy().round()) 
pr_mask = (predicted_mask.squeeze().detach().cpu().numpy().round())
print("predicted mask shape", pr_mask.shape)
print("gt mask shape", gt_mask.shape)
pr_mask = pr_mask[1,:,:]
gt_mask = gt_mask[1,:,:]
image_t = image.transpose(1, 2, 0)
# print(image.shape, image_t.shape)


def visualizeData(image, ground_truth_mask, predicted_mask):
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(image)
    ax[0].set_title(f'Original Image test image:{idx}')
    ax[1].imshow(ground_truth_mask)
    ax[1].set_title('Ground Truth Mask')
    ax[2].imshow(predicted_mask)
    ax[2].set_title('Predicted Mask')
    plt.savefig('test_original_mask_and_preds.png')
    plt.show()
    



predicted_output = torch.argmax(predicted_mask.squeeze(), dim=0).detach().cpu().numpy()


if isinstance(gt_mask_copy, np.ndarray):
    gt_mask_copy = torch.from_numpy(gt_mask_copy)


gt_output = torch.argmax(gt_mask_copy.squeeze(), dim=0).detach().cpu().numpy()
Indices = np.unique(predicted_output)
 
for i in Indices:
  print(CLASSES[i])



def decode_segmentation_map(image, class_colors_rgb):
    if image.ndim != 2:
        raise ValueError("Image for segmentation map must be a 2D array")

    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)

    for l in range(len(class_colors_rgb)):
        idx = (image == l)
        r[idx] = class_colors_rgb[l][0]
        g[idx] = class_colors_rgb[l][1]
        b[idx] = class_colors_rgb[l][2]

    rgb = np.stack([r, g, b], axis=2)
    return rgb


rgb_gt = decode_segmentation_map(gt_output, class_colors_rgb)

# Decode the predicted mask
rgb_map = decode_segmentation_map(predicted_output, class_colors_rgb) 
# plt.imshow(rgb_map)
# plt.show()


visualizeData(image_t, rgb_gt, rgb_map)


# Convert RGB to BGR for OpenCV
bgr_map = cv2.cvtColor(rgb_map, cv2.COLOR_RGB2BGR)
cv2.imwrite('rgb_predicted_map.png', bgr_map)

unique_classes = np.unique(predicted_output)

class_accuracies = {}
for cls in unique_classes:
    pr_cls_mask = (predicted_output == cls)
    gt_cls_mask = (gt_mask_copy_for_compute[cls, :, :])
    pr_cls_mask = pr_cls_mask.astype(bool)
    gt_cls_mask = gt_cls_mask.astype(bool)


    correct_predictions = np.sum(pr_cls_mask & gt_cls_mask)
    total_gt = np.sum(gt_cls_mask)
    
    if total_gt > 0:
        accuracy = correct_predictions / total_gt
        class_accuracies[cls] = accuracy
    else:
        class_accuracies[cls] = np.nan  

for cls, acc in class_accuracies.items():
    if np.isnan(acc):
        print(f"Class {CLASSES[cls]}: div by zero, and opt did something funny")
    else:
        print(f"Class {CLASSES[cls]}: {acc * 100:.2f}% accuracy")


num_classes = gt_mask_copy.shape[0]

gt_classes_present = []
for cls in range(num_classes):
    if np.any(gt_mask_copy_for_compute[cls, :, :]):
        gt_classes_present.append(cls)
        print(f"Class {cls} present in ground truth: {CLASSES[cls]}")
