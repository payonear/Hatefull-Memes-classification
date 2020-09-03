import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from utils.dataset import *
from pathlib import Path
from tqdm import tqdm

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
        _, loss = self.forward(
            batch['image'],
            batch['text'],
            batch['label']
        )
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        _, loss = self.eval().forward(
            batch['image'],
            batch['text'],
            batch['label']
        )
        return {'val_loss': loss}

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        return {'avg_val_loss': avg_loss,
                'progress_bar': {'avg_val_loss': avg_loss},
                'log': {'avg_val_loss': avg_loss}
        }

    def optim_config(self):
        optimizer = [optim.Adam(self.model.parameters(), 
                    lr=self.hparams.get('lr', 1e-3))]
        scheduler = [optim.lr_scheduler.CosineAnnealingLR(optimizer[0], 
                                                T_max=self.hparams.get('max_epochs',10))]
        return optimizer, scheduler

    def train_dataloader(self):
        loader = DataLoader(self.train_dataset,
            batch_size = self.hparams.get('batch_size', 8),
            shuffle = True,
            num_workers = self.hparams.get('num_workers ', 4)
        )
        return loader

    def val_dataloader(self):
        loader = DataLoader(self.val_dataset,
            batch_size = self.hparams.get('batch_size', 8),
            shuffle = False,
            num_workers = self.hparams.get('num_workers ', 4)
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
        self._set_seed(self.hparams.get('random_state', 17))
        self.trainer = pl.Trainer(**self.trainer_params)
        self.trainer.fit(self)

    def __get_trainer_params(self):
        checkpoint_callback = ModelCheckpoint(
                            filepath=self.output_path,
                            save_best_only=self.hparams.get('save_best_only', True),
                            verbose=self.hparams.get('verbose', True),
                            monitor='avg_val_loss',
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
            'early_stop_callback': early_stop_callback,
            'gradient_clip_val': self.hparams.get(
                "gradient_clip_value", 1),
            'gpus': self.hparams.get('gpus', 1),
            'overfit_batches': self.hparams.get(
                'overfit_batches', 0.0),
            'max_epochs': self.hparams.get(
                'max_epochs', 10),
            'default_save_path': self.output_path
        }
        return trainer_params

    @torch.no_grad()
    def make_submission(self, path):
        test_dataset = self.__build_dataset('test_path')
        loader = DataLoader(self.test_dataset,
            batch_size = self.hparams.get('batch_size', 8),
            shuffle = False,
            num_workers = self.hparams.get('num_workers ', 4)
        )
        submission = pd.DataFrame(index = test_dataset.sample_frame.id,
                                columns = ['proba','label'])
        for batch in tqdm(loader, total = len(loader)):
            preds, _ = self.model.eval().to('cpu')(
                batch['image'], batch['text']
            )
            submission.loc[batch['id'], 'proba'] = preds[:,1]
            submission.loc[batch['id'], 'label'] = preds.argmax(dim=1)
            submission.proba = submission.proba.astype(float)
            submission.label = submission.proba.astype(int)
            return submission