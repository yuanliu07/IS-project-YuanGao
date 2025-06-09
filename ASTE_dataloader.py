import torch
import numpy as np
from collections import Counter
from torch.utils.data import Dataset

from vocab import *
from span_tagging import form_raw_table, map_raw_table_to_id


# 继承自PyTorch的Dataset类的自定义数据集类
class ASTE_Dataset(Dataset):
    def __init__(self, file_name, vocab=None, version='3D', tokenizer=None, max_len=128, lower=True, is_clean=True):
        super().__init__()

        self.max_len = max_len
        self.lower = lower
        self.version = version
        self.tokenizer = tokenizer


        if type(file_name) is str:
            with open(file_name, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                self.raw_data = [line2dict(l, is_clean=is_clean) for l in lines]
        else:
            self.raw_data = file_name

        self.data = self.preprocess(self.raw_data, vocab=vocab, version=version)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


    def text2bert_id(self, token):
        # 将文本转换为BERT的token ID，同时生成word_mapback列表，用于记录原始单词与token之间的关系。
        re_token = []
        word_mapback = []
        word_split_len = []
        for idx, word in enumerate(token):
            temp = self.tokenizer.tokenize(word)
            re_token.extend(temp)
            word_mapback.extend([idx] * len(temp))
            word_split_len.append(len(temp))
        re_id = self.tokenizer.convert_tokens_to_ids(re_token)
        return re_id, word_mapback, word_split_len

    def preprocess(self, data, vocab, version):

        token_vocab = vocab['token_vocab']
        label2id = vocab['label_vocab']['label2id']
        processed = []
        max_len = self.max_len
        CLS_id = self.tokenizer.convert_tokens_to_ids([self.tokenizer.cls_token])
        SEP_id = self.tokenizer.convert_tokens_to_ids([self.tokenizer.sep_token])

        for d in data:
            golden_label = map_raw_table_to_id(form_raw_table(d), label2id) if 'triplets' in d else None
            # tok
            tok = d['token']
            if self.lower:
                tok = [t.lower() for t in tok]

            text_raw_bert_indices, word_mapback, _ = self.text2bert_id(tok)
            text_raw_bert_indices = text_raw_bert_indices[:max_len]
            word_mapback = word_mapback[:max_len]

            length = word_mapback[-1] + 1
            assert (length == len(tok))
            bert_length = len(word_mapback)

            tok = tok[:length]
            tok = [token_vocab.stoi.get(t, token_vocab.unk_index) for t in tok]

            temp = {
                'token': tok,
                'token_length': length,
                'bert_token': CLS_id + text_raw_bert_indices + SEP_id,
                'bert_length': bert_length,
                'bert_word_mapback': word_mapback,
                'golden_label': golden_label
            }



            processed.append(temp)
            # print("processed", processed)

        return processed


# 批处理函数,将多个数据样本（一个batch）组合成一个批次的张量，以便进行批处理。
def ASTE_collate_fn(batch):

    # batch = [b for b in batch if b is not None]  # 过滤空样本

    batch_size = len(batch)

    re_batch = {}

    # 提取每个样本中的xxx，转为长整型张量
    token = get_long_tensor([batch[i]['token'] for i in range(batch_size)])
    token_length = torch.tensor([batch[i]['token_length'] for i in range(batch_size)])
    bert_token = get_long_tensor([batch[i]['bert_token'] for i in range(batch_size)])
    bert_length = torch.tensor([batch[i]['bert_length'] for i in range(batch_size)])
    bert_word_mapback = get_long_tensor([batch[i]['bert_word_mapback'] for i in range(batch_size)])

    golden_label = np.zeros((batch_size, token_length.max(), token_length.max()), dtype=np.int64)

    if batch[0]['golden_label'] is not None:
        for i in range(batch_size):
            golden_label[i, :token_length[i], :token_length[i]] = batch[i]['golden_label']

    golden_label = torch.from_numpy(golden_label)

    re_batch = {
        'token': token,
        'token_length': token_length,
        'bert_token': bert_token,
        'bert_length': bert_length,
        'bert_word_mapback': bert_word_mapback,
        'golden_label': golden_label
    }

    return re_batch


def get_long_tensor(tokens_list, max_len=None):
    # 将一个由列表组成的列表（即二维列表，其中每个内部列表代表一个样本的token序列）转换为一个填充了0的PyTorch长整型张量（LongTensor）
    """ Convert list of list of tokens to a padded LongTensor. """
    batch_size = len(tokens_list)
    token_len = max(len(x) for x in tokens_list) if max_len is None else max_len
    tokens = torch.LongTensor(batch_size, token_len).fill_(0)
    for i, s in enumerate(tokens_list):
        tokens[i, : min(token_len, len(s))] = torch.LongTensor(s)[:token_len]
    return tokens


############################################################################
# data preprocess 数据预处理
def clean_data(l):
    # 清洗数据，移除重复的三元组，并重新格式化字符串（句子+####+三元组）
    token, triplets = l.strip().split('####')
    temp_t = list(set([str(t) for t in eval(triplets)]))
    return token + '####' + str([eval(t) for t in temp_t]) + '\n'


def line2dict(l, is_clean=False):
    if is_clean:
        l = clean_data(l)
    #     清洗后，转为字典，包含句子的单词列表和排序后的三元组列表。
    sentence, triplets = l.strip().split('####')
    start_end_triplets = []
    for t in eval(triplets):
        start_end_triplets.append(tuple([[t[0][0], t[0][-1]], [t[1][0], t[1][-1]], t[2]]))
    start_end_triplets.sort(key=lambda x: (x[0][0], x[1][-1]))  # sort ?
    return dict(token=sentence.split(' '), triplets=start_end_triplets)


#############################################################################
# vocab 词汇
def build_vocab(dataset):
    # 从数据集中构建词汇表
    tokens = []

    files = ['train_triplets.txt', 'dev_triplets.txt', 'test_triplets.txt']
    for file_name in files:
        file_path = dataset + '/' + file_name
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        for l in lines:
            # 提取句子，分割出单词
            cur_token = l.strip().split('####')[0].split()
            tokens.extend(cur_token)
    return tokens


def load_vocab(dataset_dir, lower=True):
    # 加载词汇表，创建并返回一个包含词汇表的字典。
    tokens = build_vocab(dataset_dir)
    if lower:
        tokens = [w.lower() for w in tokens]
    token_counter = Counter(tokens)
    token_vocab = Vocab(token_counter, specials=["<pad>", "<unk>"])
    vocab = {'token_vocab': token_vocab}
    return vocab
