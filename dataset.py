import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import numpy as np
import cv2
import matplotlib.pyplot as plt

import pandas as pd
from torch.utils.data import Dataset as BaseDataset
import torchvision.transforms as T
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
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


CLASSES = [ 'background','road', 'lanemarks', 'curb', 'person', 'rider', 'vehicles', 'bicycle', 'motorcycle', 'traffic sign']


class_colors_bgr = [
    [0,0,0], #background
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


transform = T.Resize(size = (512,512))

class Dataset(BaseDataset):
    def __init__(self, images_dir, masks_dir, classes=None, augmentation=None, preprocessing=None):
        a = os.listdir(images_dir)
        b = os.listdir(masks_dir)
        self.ids_x = sorted(a[:100])
        self.ids_y = sorted(os.listdir(b[:100]))
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids_x]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids_y]

        self.class_rgb_values = {cls: np.array(rgb) for cls, rgb in zip(classes, class_rgb_values)}
        self.class_values = [CLASSES.index(cls.lower()) for cls in classes]
        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):
        # Read data
        image = cv2.imread(self.images_fps[i],cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
        
        # # Read mask in BGR color space
        mask_bgr = cv2.imread(self.masks_fps[i], cv2.IMREAD_COLOR)
        mask_bgr = cv2.cvtColor(mask_bgr, cv2.COLOR_BGR2RGB)
        mask_bgr = cv2.resize(mask_bgr, (224, 224), interpolation=cv2.INTER_NEAREST)


        binary_masks = [np.all(mask_bgr == np.array(color), axis=-1).astype(np.float32) for color in class_colors_rgb]

        # Stack binary masks into a multi-channel mask
        final_mask = np.stack(binary_masks, axis=-1)

        # Apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=final_mask)
            image, final_mask = sample['image'], sample['mask']

        # Apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=final_mask)
            image, final_mask = sample['image'], sample['mask']


        return image, final_mask

    def __len__(self):
        return len(self.ids_x)



def visualize(image, mask, class_rgb_values, class_idx, label=None):
    plt.figure(figsize=(14, 10))

    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title("Original Image")

    plt.subplot(1, 2, 2)
    single_class_mask = mask[:, :, class_idx]
    rgb_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    rgb_mask[single_class_mask == 1] = class_rgb_values[class_idx]

    plt.imshow(rgb_mask)
    plt.title(f"Mask for {label}" if label else f"Mask for class {class_idx}")

    plt.show()

# # Looping through each class to visualize
# for idx, label in enumerate(CLASSES):
#     dataset = Dataset(x_test_dir, y_test_dir, classes=[label])
#     image, mask = dataset[2]  # Assuming you want to visualize the third image in the dataset
#     visualize(image=image, mask=mask, class_rgb_values=class_colors_rgb, class_idx=idx, label=label)


# binary mask data, simple test
class CarvanaDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index].replace(".jpg", "_mask.gif"))
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        mask[mask == 255.0] = 1.0

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        return image, mask





