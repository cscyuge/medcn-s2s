PAD, CLS = '[PAD]', '[CLS]'
SEP = '[SEP]'
from tqdm import tqdm
import re
import nltk
import pickle
import torch

def build_dataset(config, src_ids_path, src_masks_path, tar_ids_path, tar_masks_path, tar_txt_path):
    token_ids_srcs = pickle.load(open(src_ids_path, 'rb'))
    seq_len_src = config.pad_size
    mask_srcs = pickle.load(open(src_masks_path, 'rb'))

    token_ids_tars = pickle.load(open(tar_ids_path, 'rb'))
    seq_len_tar = config.pad_size
    mask_tars = pickle.load(open(tar_masks_path,'rb'))

    tar_txts = pickle.load(open(tar_txt_path,'rb'))

    dataset = []
    for token_ids_src, mask_src, token_ids_tar, mask_tar, tar_txt in zip(token_ids_srcs, mask_srcs, token_ids_tars, mask_tars,tar_txts):
        dataset.append((token_ids_src, int(0), seq_len_src, mask_src, token_ids_tar, int(0), seq_len_tar, mask_tar, tar_txt))
    return dataset


class DatasetIterater(object):
    def __init__(self, batches, batch_size, device):
        self.batch_size = batch_size
        self.batches = batches
        self.n_batches = len(batches) // batch_size
        self.residue = False  # 记录batch数量是否为整数
        if len(batches) % self.n_batches != 0:
            self.residue = True
        self.index = 0
        self.device = device

    def _to_tensor(self, datas):
        x_src = torch.LongTensor([_[0] for _ in datas]).to(self.device)
        seq_len_src = torch.LongTensor([_[2] for _ in datas]).to(self.device)
        mask_src = torch.LongTensor([_[3] for _ in datas]).to(self.device)
        x_tar = torch.LongTensor([_[4] for _ in datas]).to(self.device)
        seq_len_tar = torch.LongTensor([_[6] for _ in datas]).to(self.device)
        mask_tar = torch.LongTensor([_[7] for _ in datas]).to(self.device)
        tar_txt =[_[8] for _ in datas]
        return (x_src, seq_len_src, mask_src), (x_tar, seq_len_tar, mask_tar), tar_txt

    def __next__(self):
        if self.residue and self.index == self.n_batches:
            batches = self.batches[self.index * self.batch_size: len(self.batches)]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches


def build_iterator(dataset, config):
    iter = DatasetIterater(dataset, config.batch_size, config.device)
    return iter


