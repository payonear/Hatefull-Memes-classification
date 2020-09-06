import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd

class HatefulMemesDataset(Dataset):
    def __init__(self,
                data_path,
                img_path,
                img_transform,
                txt_transform,
                balance=False,
                random_state=0):
        
        self.sample_frame = pd.read_json(data_path, lines=True)
        self.img_path = img_path
        self.img_transform = img_transform
        self.txt_transform = txt_transform

        if balance:
            pos = self.sample_frame[self.sample_frame.label.eq(1)]
            neg = self.sample_frame[self.sample_frame.label.eq(0)]
            self.sample_frame = pd.concat([neg.sample(pos.shape[0],
                                                random_state=random_state)
                                        ,pos])

        self.sample_frame = self.sample_frame.reset_index(drop=True)
        self.sample_frame.img = self.sample_frame.img.apply(lambda x: img_path/x)

    def __len__(self):
        return self.sample_frame.shape[0]

    def class_distr(self):
        pos = self.sample_frame.label.sum()
        print(f'Positive class make up {pos/self.sample_frame.shape[0]*100}% of sample')

    def __getitem__(self, idx):
        img_id = self.sample_frame.loc[idx, 'id']
        image = Image.open(self.sample_frame.loc[idx,'img']).convert('RGB')
        image = self.img_transform(image)
        text = self.sample_frame.loc[idx, 'text']
        text = self.txt_transform(text)
        if 'label' in self.sample_frame.columns:
            label = [self.sample_frame.loc[idx, 'label']]
            label = torch.Tensor(label).long().squeeze()
            sample = {
                    'id': img_id,
                    'image': image,
                    'text': text,
                    'label': label 
            }
        else:
             sample = {
                    'id': img_id,
                    'image': image,
                    'text': text
            }
        
        return sample

