
import pickle
import torch
from tqdm import tqdm
from transformers import BertTokenizer
from keras.preprocessing.sequence import pad_sequences


def main():
    with open('./data/src_all.pkl', 'rb') as f:
        src_all = pickle.load(f)
    with open('./data/tar_all.pkl', 'rb') as f:
        tar_all = pickle.load(f)

    # bert_model = 'trueto/medbert-base-wwm-chinese'
    bert_model = 'bert-base-chinese'
    tokenizer = BertTokenizer.from_pretrained(bert_model)

    src_ids = []
    tar_ids = []
    len_cnt = [0 for _ in range(2600)]
    for src in tqdm(src_all):
        tokens = tokenizer.tokenize(src)
        tokens = ['[CLS]'] + tokens + ['[SEP]']
        ids = tokenizer.convert_tokens_to_ids(tokens)
        src_ids.append(ids)
        len_cnt[len(ids)] += 1

    for tar in tqdm(tar_all):
        tokens = tokenizer.tokenize(tar)
        tokens = ['[CLS]'] + tokens + ['[SEP]']
        ids = tokenizer.convert_tokens_to_ids(tokens)
        tar_ids.append(ids)
        len_cnt[len(ids)] += 1
    print(len(src_ids))
    src_ids_smaller = []
    tar_ids_smaller = []
    tar_txts = []
    max_len = 256
    for src, tar, txt in zip(src_ids, tar_ids, tar_all):
        if len(src)<max_len and len(tar)<max_len:
            src_ids_smaller.append(src)
            tar_ids_smaller.append(tar)
            tar_txts.append(txt)
    src_ids = src_ids_smaller
    tar_ids = tar_ids_smaller
    print(len(src_ids))
    src_ids = pad_sequences(src_ids, maxlen=max_len, dtype="long", value=0, truncating="post", padding="post")
    tar_ids = pad_sequences(tar_ids, maxlen=max_len, dtype="long", value=0, truncating="post", padding="post")


    src_masks = [[float(i != 0.0) for i in ii] for ii in src_ids]
    tar_masks = [[float(i != 0.0) for i in ii] for ii in tar_ids]

    with open('./data/src_ids.pkl', 'wb') as f:
        pickle.dump(src_ids, f)
    with open('./data/tar_ids.pkl', 'wb') as f:
        pickle.dump(tar_ids, f)
    with open('./data/src_masks.pkl', 'wb') as f:
        pickle.dump(src_masks, f)
    with open('./data/tar_masks.pkl', 'wb') as f:
        pickle.dump(tar_masks, f)
    with open('./data/tar_txts.pkl', 'wb') as f:
        pickle.dump(tar_txts, f)


if __name__ == '__main__':
    main()