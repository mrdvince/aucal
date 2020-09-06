import pytorch_lightning as pl
import config
import numpy as np
from torch.utils.data import random_split, DataLoader
from torch.utils.data import Dataset
import os
import random
from text import text_to_tensor
import torch
from utils.audio import load_wav, melspectrogram


def files_to_list(data_dir):
    file_list = []
    with open(os.path.join(data_dir, 'metadata.csv'), encoding='utf-8')as f:
        for line in f:
            parts = line.strip().split('|')
            wav_path = os.path.join(data_dir, 'wavs', '%s.wav' % parts[0])
            file_list.append([wav_path, parts[1]])
    return file_list


class ljdataset(Dataset):
    def __init__(self, data_dir=config.hparams.data_dir):
        self.file_list = files_to_list(data_dir)
        random.shuffle(self.file_list)

    def get_text(self, text):
        text_norm = torch.IntTensor(text_to_tensor(
            text, config.hparams.text_cleaners))
        return text_norm

    def get_mel(self, filename):
        wav = load_wav(filename)
        mel = melspectrogram(wav).astype(np.float32)
        return torch.Tensor(mel)

    def get_mel_text_pair(self, fname_and_text):
        filename, text = fname_and_text[0], fname_and_text[1]
        text = self.get_text(text)
        mel = self.get_mel(filename)
        return (text, mel)

    def __getitem__(self, index):
        return self.get_mel_text_pair(self.file_list[index])

    def __len__(self):
        return len(self.file_list)

class TacotronModule(pl.LightningDataModule):
    def __init__(self):
        super().__init__()
        self.params = config.hparams
        self.data_dir = self.params.data_dir

    def prepare_data(self):
        trainset = ljdataset()

    def train_dataloader(self):
        pass
