import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from utils.dataset import *
from pathlib import Path
from tqdm import tqdm
import numpy as np
import random
from sklearn. metrics import roc_auc_score, accuracy_score

class HatefulMemesModel(pl.LightningModule):
    def __init__(self, hparams):
        keys = ['model', 'train_path', 'dev_path', 'test_path', 
                'img_path', 'img_transform', 'txt_transform']
        for key in keys:
            if key not in hparams.keys():
                raise KeyError(f'{key} is required parameter!')
        
        super(HatefulMemesModel,self).__init__()
        self.hparams = hparams
        self.model = self.hparams.get('model')
        self.model.to(self.hparams.get('device', 'cpu'))
        self.img_transform = self.hparams.get('img_transform')
        self.txt_transform = self.hparams.get('txt_transform')
        self.train_dataset = self.__build_dataset('train_path')
        self.val_dataset = self.__build_dataset('dev_path')
        self.output_path = Path(
            self.hparams.get('output_path', 'model-outputs')
        )
        self.output_path.mkdir(exist_ok = True)
        self.trainer_params = self.__get_trainer_params()

    def forward(self, img, txt, label):
        return self.model(img, txt, label)

    def training_step(self, batch, batch_idx):
        device = self.hparams.get('device', 'cpu')
        _, loss = self.forward(
            #batch['image'].to(device),
            batch['text'].to(device),
            batch['label'].to(device)
        )
        return {'loss': loss}

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        return {'avg_train_loss': avg_loss,
                'progress_bar': {'avg_train_loss': avg_loss},
                'log': {'avg_train_loss': avg_loss}
            }
    def validation_step(self, batch, batch_idx):
        device = self.hparams.get('device', 'cpu')
        _, loss = self.eval().forward(
            batch['image'].to(device),
            batch['text'].to(device),
            batch['label'].to(device)
        )
        return {'val_loss': loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        return {'avg_val_loss': avg_loss,
                'progress_bar': {'avg_val_loss': avg_loss},
                'log': {'avg_val_loss': avg_loss}
        }

    def configure_optimizers(self):
        optimizer = [optim.Adam(self.model.parameters(), 
                    lr=self.hparams.get('lr', 1e-3))]
        scheduler = [optim.lr_scheduler.CosineAnnealingLR(optimizer[0], 
                                               T_max=self.hparams.get('max_epochs',10))]
        return optimizer, scheduler

    def train_dataloader(self):
        loader = DataLoader(self.train_dataset,
            batch_size = self.hparams.get('batch_size', 8),
            shuffle = self.hparams.get('shuffle', True),
            num_workers = self.hparams.get('num_workers', 4),
            pin_memory= self.hparams.get('pin_memory', False)
        )
        return loader

    def val_dataloader(self):
        loader = DataLoader(self.val_dataset,
            batch_size = self.hparams.get('batch_size', 8),
            shuffle = False,
            num_workers = self.hparams.get('num_workers', 4),
            pin_memory= self.hparams.get('pin_memory', False)
        )
        return loader

    def __set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def __build_dataset(self, key):
        data_path = self.hparams.get(key)
        img_path = self.hparams.get('img_path')
        balance = (self.hparams.get('balance', False) 
                    if 'train' in str(key) else False)
        random_state = self.hparams.get('random_state', 0)
        dataset = HatefulMemesDataset(data_path,
                                    img_path,
                                    self.img_transform,
                                    self.txt_transform,
                                    balance,
                                    random_state
        )
        return dataset    

    def fit(self):
        self.__set_seed(self.hparams.get('random_state', 17))
        self.trainer = pl.Trainer(**self.trainer_params)
        self.trainer.fit(self)

    def __get_trainer_params(self):
        checkpoint_callback = ModelCheckpoint(
                            filepath=self.output_path,
                            verbose=self.hparams.get('verbose', True),
                            monitor=self.hparams.get('monitor', 'avg_val_loss'),
                            mode='min',
                            prefix=self.hparams.get('prefix', ''),
                        )
        early_stop_callback = EarlyStopping(
                            monitor=self.hparams.get(
                                'early_stop_monitor', 'avg_val_loss'),
                            min_delta=self.hparams.get(
                                'early_stop_min_delta', 0.001),
                            patience=self.hparams.get(
                                'early_stop_patience', 3),
                            verbose=self.hparams.get('verbose', True)
                        )

        trainer_params = {
            'checkpoint_callback': checkpoint_callback,
            'early_stop_callback': self.hparams.get(
                'early_stop_callback', early_stop_callback),
            'gradient_clip_val': self.hparams.get(
                "gradient_clip_val", 1),
            'gpus': self.hparams.get('gpus', 1),
            'overfit_pct': self.hparams.get(
                'overfit_pct', None),
            'max_epochs': self.hparams.get(
                'max_epochs', 10),
            'default_root_dir': self.output_path,
            'fast_dev_run': self.hparams.get(
                'fast_dev_run', False),
            'logger': self.hparams.get(
                'logger', False)
        }
        return trainer_params
    
    @torch.no_grad()
    def val_metrics(self):
        device = self.hparams.get('device', 'cpu')
        df = pd.DataFrame(index = self.val_dataset.sample_frame.id,
                                columns = ['proba','label', 'pred'])
        loader = self.val_dataloader()
        for batch in tqdm(loader, total = len(loader)):
            preds, _ = self.model.eval().to(device)(
                    batch['image'].to(device), batch['text'].to(device))
            if device=='cuda':
                preds = preds.cpu().detach().numpy()
            df.loc[batch['id'], 'proba'] = preds[:,1]
            df.loc[batch['id'], 'label'] = batch['label']
            df.loc[batch['id'], 'pred'] = preds.argmax(1)
        df.proba = df.proba.astype(float)
        df.label = df.label.astype(int)
        df.pred = df.pred.astype(int)
        auc_roc = roc_auc_score(df.label, df.proba)
        acc = accuracy_score(df.label, df.pred)
        print(f'AUC_ROC: {auc_roc}, Accuracy: {acc}')

    @torch.no_grad()
    def make_submission(self, path, device):
        device = self.hparams.get('device', 'cpu')
        test_dataset = self.__build_dataset('test_path')
        loader = DataLoader(test_dataset,
            batch_size = self.hparams.get('batch_size', 8),
            shuffle = False,
            num_workers = self.hparams.get('num_workers ', 4)
        )
        submission = pd.DataFrame(index = test_dataset.sample_frame.id,
                                columns = ['proba','label'])
        for batch in tqdm(loader, total = len(loader)):
            preds, _ = self.model.eval()(
                batch['image'].to(device), batch['text'].to(device)
            )
            if device=='cuda':
                preds = preds.cpu().detach().numpy()
            submission.loc[batch['id'], 'proba'] = preds[:,1]
            submission.loc[batch['id'], 'label'] = preds.argmax(1)
        submission.proba = submission.proba.astype(float)
        submission.label = submission.label.astype(int)
        submission.to_csv((path), index=True)
        return submission