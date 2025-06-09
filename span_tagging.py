# span tagging
# 根据给定的数据字典d和版本version构建一个原始的情感分析表格（raw table）
def form_raw_table(d):
    # 初始化一个二维列表，用于存储每个单词或词组的标签；提取a、o索引
    raw_table = [['' for _ in range(len(d['token']))] for _ in range(len(d['token']))]
    aspect_index = list(set([(x[0][0], x[0][-1]) for x in d['triplets']]))
    opinion_index = list(set([(x[1][0], x[1][-1]) for x in d['triplets']]))

    # schema，创建字典，用于存储每个单词或词组的情感极性。
    candidate_senti_aspect_opinion_same = {(min(t[0][0], t[1][0]), max(t[0][1], t[1][1])): t[2] for t in d['triplets']}

    candidate_senti = candidate_senti_aspect_opinion_same

    for i in range(len(d['token'])):
        for j in range(i, len(d['token'])):
            raw_table[i][j] = 'A-' if (i, j) in aspect_index else 'N-'
            raw_table[i][j] += ('O-' if (i, j) in opinion_index else 'N-')
            raw_table[i][j] += candidate_senti[(i, j)] if (i, j) in candidate_senti else 'N'
    return raw_table


def form_label_id_map():
    # 根据不同的版本（version）构建标签到ID的映射（label2id）和ID到标签的映射（id2label）。
    # 这些映射在训练模型时用于将文本标签转换为数值表示，以及在模型预测后将数值结果转换回文本标签。
    label_list = []
    for ifA in ['N', 'A']:
        for ifO in ['N', 'O']:
            for ifP in ['N', 'NEG', 'NEU', 'POS']:
                label_list.append(ifA + '-' + ifO + '-' + ifP)

    label2id = {x: idx for idx, x in enumerate(label_list)}
    id2label = {idx: x for idx, x in enumerate(label_list)}
    return label2id, id2label


def form_sentiment_id_map():
    # 创建情感标签到ID的映射和ID到情感标签的映射
    label_list = ['N', 'NEG', 'NEU', 'POS']
    label2id = {x: idx for idx, x in enumerate(label_list)}
    id2label = {idx: x for idx, x in enumerate(label_list)}
    return label2id, id2label


def map_raw_table_to_id(raw_table, label2id):
    # 将原始的情感分析表格（raw table）转换为ID表示
    return [[label2id.get(x, 0) for x in y] for y in raw_table]


def map_id_to_raw_table(raw_table_id, id2label):
    # 将ID表示的情感分析表格转换回原始的情感分析表格
    return [[id2label[x] for x in y] for y in raw_table_id]
