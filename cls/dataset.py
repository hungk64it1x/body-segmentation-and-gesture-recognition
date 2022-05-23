import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
from utils import *
from config import CFG
import os
import cv2



class GestureDataset(Dataset):
    
    def __init__(self, data_path, mask_path, folders, frames, transform=None):
        self.data_path = data_path
        self.mask_path = mask_path
        self.folders = folders
        self.transform = transform
        self.frames = frames

    def __len__(self):
        return len(self.folders)

    def read_images(self, image_path, mask_path, selected_folder):
        images = []
        for i in self.frames:
            image = cv2.imread(os.path.join(image_path, selected_folder, '{}{}.jpg'.format(selected_folder, i)))[:,:,::-1]
            image = cv2.resize(image, (CFG.IMAGE_SIZE, CFG.IMAGE_SIZE))
            mask = cv2.imread(os.path.join(mask_path, selected_folder, '{}{}.jpg'.format(selected_folder, i)))[:,:,::-1]
            mask = cv2.resize(mask, (CFG.IMAGE_SIZE, CFG.IMAGE_SIZE))
            concat = cv2.bitwise_and(image, mask)
            images.append(concat)
        
        if self.transform is not None:
            images_dict = dict()
            for i in range(len(images)):
                if i==0:
                    images_dict['image'] = images[i]
                else:
                    images_dict[f'image{i-1}'] = images[i]
            augmented = self.transform(**images_dict)
           
            transformed_images = []
            for i in range(len(images)):
                if(i==0):
                    transformed_images.append(augmented['image'])
                else:
                    transformed_images.append(augmented[f'image{i-1}'])
                    
            transformed_images = torch.stack(transformed_images, axis=0)
            return transformed_images
        else:
            images = torch.stack(images, axis=0)
            return images

    def __getitem__(self, index):
        folder = self.folders[index]
        label = str(folder)[0]
        # Load data
        X = self.read_images(f'{self.data_path}/{label}', f'{self.mask_path}/{label}', folder)   
        
        if(X.shape[0] < CFG.SEQ_LEN):
            n_pad = CFG.SEQ_LEN - X.shape[0]
            pad_matrix = np.zeros(shape=(n_pad, X.shape[1], X.shape[2], X.shape[3]))
            X = torch.cat([X, pad_matrix], axis=0)
        y = torch.tensor(int(label)).long()            
        return X, y
    
    
class GestureTestDataset(Dataset):
    
    def __init__(self, data_path, mask_path, folders, frames, transform=None):
        self.data_path = data_path
        self.mask_path = mask_path
        self.folders = folders
        self.transform = transform
        self.frames = frames

    def __len__(self):
        return len(self.folders)

    def read_images(self, image_path, mask_path, selected_folder):
        images = []
        for i in self.frames:
            image = cv2.imread(os.path.join(image_path, str(selected_folder), '{}.jpg'.format(i)))[:,:,::-1]
            image = cv2.resize(image, (CFG.IMAGE_SIZE, CFG.IMAGE_SIZE))
            mask = cv2.imread(os.path.join(mask_path, str(selected_folder), '{}.jpg'.format(i)))[:,:,::-1]
            mask = cv2.resize(mask, (CFG.IMAGE_SIZE, CFG.IMAGE_SIZE))
            concat = cv2.bitwise_and(image, mask)
            images.append(concat)
        
        if self.transform is not None:
            images_dict = dict()
            for i in range(len(images)):
                if i==0:
                    images_dict['image'] = images[i]
                else:
                    images_dict[f'image{i-1}'] = images[i]
            augmented = self.transform(**images_dict)
           
            transformed_images = []
            for i in range(len(images)):
                if(i==0):
                    transformed_images.append(augmented['image'])
                else:
                    transformed_images.append(augmented[f'image{i-1}'])
                    
            transformed_images = torch.stack(transformed_images, axis=0)
            return transformed_images
        else:
            images = torch.stack(images, axis=0)
            return images

    def __getitem__(self, index):
        folder = self.folders[index]
        # Load data
        X = self.read_images(f'{self.data_path}', f'{self.mask_path}', folder)   
        if(X.shape[0] < CFG.SEQ_LEN):
            n_pad = CFG.SEQ_LEN - X.shape[0]
            pad_matrix = np.zeros(shape=(n_pad, X.shape[1], X.shape[2], X.shape[3]))
            X = torch.cat([X, pad_matrix], axis=0)
         
        return X

class GesturePseudoDataset(Dataset):
    
    def __init__(self, data_path, mask_path, labels, folders, frames, transform=None):
        self.data_path = data_path
        self.mask_path = mask_path
        self.folders = folders
        self.transform = transform
        self.frames = frames
        self.labels = labels

    def __len__(self):
        return len(self.folders)

    def read_images(self, image_path, mask_path, selected_folder):
        images = []
        for i in self.frames:
            image = cv2.imread(os.path.join(image_path, str(selected_folder), '{}.jpg'.format(i)))[:,:,::-1]
            image = cv2.resize(image, (CFG.IMAGE_SIZE, CFG.IMAGE_SIZE))
            mask = cv2.imread(os.path.join(mask_path, str(selected_folder), '{}.jpg'.format(i)))[:,:,::-1]
            mask = cv2.resize(mask, (CFG.IMAGE_SIZE, CFG.IMAGE_SIZE))
            concat = cv2.bitwise_and(image, mask)
            images.append(concat)
        
        if self.transform is not None:
            images_dict = dict()
            for i in range(len(images)):
                if i==0:
                    images_dict['image'] = images[i]
                else:
                    images_dict[f'image{i-1}'] = images[i]
            augmented = self.transform(**images_dict)
           
            transformed_images = []
            for i in range(len(images)):
                if(i==0):
                    transformed_images.append(augmented['image'])
                else:
                    transformed_images.append(augmented[f'image{i-1}'])
                    
            transformed_images = torch.stack(transformed_images, axis=0)
            return transformed_images
        else:
            images = torch.stack(images, axis=0)
            return images

    def __getitem__(self, index):
        folder = self.folders[index]
        # Load data
        X = self.read_images(f'{self.data_path}', f'{self.mask_path}', folder)   
        if(X.shape[0] < CFG.SEQ_LEN):
            n_pad = CFG.SEQ_LEN - X.shape[0]
            pad_matrix = np.zeros(shape=(n_pad, X.shape[1], X.shape[2], X.shape[3]))
            X = torch.cat([X, pad_matrix], axis=0)
         
        y = torch.tensor(int(self.labels[index])).long()
         
        return X, y