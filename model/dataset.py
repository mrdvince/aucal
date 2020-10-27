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


class ljcollate():
    def __init__(self, n_frames_per_step=config.hparams.n_frames_per_step):
        self.n_frames_per_step = n_frames_per_step

    def __call__(self, batch):
        input_lenghts, ids_sorted = torch.sort(
            torch.LongTensor([len(x) for x in batch]),
            dim=0, descending=True)

        max_input_len = input_lenghts[0]
        text_padded = torch.LongTensor(len(batch), max_input_len)
        text_padded.zero_()
        for i in range(len(ids_sorted)):
            text = batch[ids_sorted[i]][0]
            text_padded[i, :text.size(0)] = text

        num_mels = batch[0][1].size(0)
        max_target_len = max([x[1].size for x in batch])

        if max_target_len % self.n_frames_per_step != 0:
            max_input_len += self.n_frames_per_step - \
                max_target_len % self.n_frames_per_step
            assert max_target_len % self.n_frames_per_step == 0

        mel_padded = torch.FloatTensor(len(batch), num_mels, max_input_len)
        mel_padded.zero_()
        gate_padded = torch.FloatTensor(len(batch), max_target_len)
        gate_padded.zero_()
        output_lenghths = torch.LongTensor(len(batch))

        for i in range(len(ids_sorted)):
            mel = batch[ids_sorted[i]][1]
            mel_padded[i, :, :mel.size(1)] = mel
            gate_padded[i, mel.size(1)-1:] = 1
            output_lenghths[i] = mel.size(1)

        return text_padded, input_lenghts, mel_padded, gate_padded, output_lenghths


class TacotronModule(pl.LightningDataModule):
    def __init__(self):
        super().__init__()
        self.params = config.hparams
        self.data_dir = self.params.data_dir

    def prepare_data(self):
        trainset = ljdataset()

    def train_dataloader(self):
        pass
