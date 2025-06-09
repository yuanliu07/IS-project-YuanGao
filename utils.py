import os
import torch
import random
import logging
import numpy as np
from datetime import datetime


# 确保结果目录存在，不存在就创建
def ensure_dir(d, verbose=True):
    if not os.path.exists(d):
        if verbose:
            print("Directory {} do not exist; creating...".format(d))
        os.makedirs(d)


# 设置随机种子，确保实验可重复（？）
def set_seed(random_seed):
    os.environ['PYTHONHASHSEED'] = str(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    random.seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


# 结果显示
def show_results(saved_results):
    all_str = ''
    for version in ['1D', '2D', '3D']:
        all_str += 'STAGE' + '-' + version + '\t'
        for dataset in ['14lap', '14res', '15res', '16res']:
            # 数据集遍历（上），数据集、版本；p、r、f1（下）
            k = '{}-{}-True'.format(dataset, version)
            all_str += '|{:.2f}\t{:.2f}\t{:.2f}|\t'.format(saved_results[k]['precision'], saved_results[k]['recall'],
                                                           saved_results[k]['f1'])
        all_str += '\n'

    # logger.info("results:{:.5f}").format(all_str)
    return all_str


def create_logger(args, log_pkg):
    """
    :param logger_file_path:
    :return:
    """
    # 获取当前的时间
    current_time = datetime.now()
    # 将时间格式化为字符串，这里使用的格式是 年月日时分秒
    timestamp_str = current_time.strftime('%Y_%m_%d_%H_%M_%S')

    # 创建日志文件的文件名
    # log_pkg = os.path.join(destination_folder, args.configs, args.dataset)
    if not os.path.exists(log_pkg):
        os.makedirs(log_pkg)
    log_filename = os.path.join(log_pkg, f'log_{timestamp_str}.log')

    logger = logging.getLogger()  # 设定日志对象
    logger.setLevel(logging.INFO)  # 设定日志等级

    file_handler = logging.FileHandler(log_filename)  # 文件输出
    console_handler = logging.StreamHandler()  # 控制台输出

    # 输出格式
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s: %(message)s "
    )

    file_handler.setFormatter(formatter)  # 设置文件输出格式
    console_handler.setFormatter(formatter)  # 设施控制台输出格式
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


import torch


# Algorithm 1: Greedy Inference
def loop_version_from_tag_table_to_triplets(tag_table, id2senti, version='3D'):
    raw_table_id = torch.tensor(tag_table)
    # 3D: {N,A} - {N,O} - {N, NEG, NEU, POS}
    if_aspect = (raw_table_id & torch.tensor(8)) > 0
    if_opinion = (raw_table_id & torch.tensor(4)) > 0
    if_triplet = (raw_table_id & torch.tensor(3))

    m = if_triplet.nonzero()
    senti = if_triplet[m[:, 0], m[:, 1]].unsqueeze(dim=-1)
    candidate_triplets = torch.cat([m, senti, m.sum(dim=-1, keepdim=True)], dim=-1).tolist()
    candidate_triplets.sort(key=lambda x: (x[-1], x[0]))

    valid_triplets = []
    valid_triplets_set = set([])

    # line 5 to line 24 (look into every sentiment snippet)
    for r_begin, c_end, p, _ in candidate_triplets:

        #####################################################################################################
        # CASE-1: aspect-opinion
        aspect_candidates = guarantee_list(
            (if_aspect[r_begin, r_begin:(c_end + 1)].nonzero().squeeze() + r_begin).tolist())  # line 7
        opinion_candidates = guarantee_list(
            (if_opinion[r_begin:(c_end + 1), c_end].nonzero().squeeze() + r_begin).tolist())  # line 8

        if len(aspect_candidates) and len(opinion_candidates):  # line 9
            select_aspect_c = -1 if (len(aspect_candidates) == 1 or aspect_candidates[-1] != c_end) else -2  # line 10
            select_opinion_r = 0 if (len(opinion_candidates) == 1 or opinion_candidates[0] != r_begin) else 1  # line 11

            # line 12
            a_ = [r_begin, aspect_candidates[select_aspect_c]]
            o_ = [opinion_candidates[select_opinion_r], c_end]
            s_ = id2senti[p]  # id2label[p]

            # line 13
            if str((a_, o_, s_)) not in valid_triplets_set:
                valid_triplets.append((a_, o_, s_))
                valid_triplets_set.add(str((a_, o_, s_)))

        #####################################################################################################
        # CASE-2: opinion-aspect
        opinion_candidates = guarantee_list(
            (if_opinion[r_begin, r_begin:(c_end + 1)].nonzero().squeeze() + r_begin).tolist())  # line 16
        aspect_candidates = guarantee_list(
            (if_aspect[r_begin:(c_end + 1), c_end].nonzero().squeeze() + r_begin).tolist())  # line 17

        if len(aspect_candidates) and len(opinion_candidates):  # line 18
            select_opinion_c = -1 if (
                        len(opinion_candidates) == 1 or opinion_candidates[-1] != c_end) else -2  # line 19
            select_aspect_r = 0 if (len(aspect_candidates) == 1 or aspect_candidates[0] != r_begin) else 1  # line 20

            # line 21
            o_ = [r_begin, opinion_candidates[select_opinion_c]]
            a_ = [aspect_candidates[select_aspect_r], c_end]
            s_ = id2senti[p]  # id2label[p]

            # line 22
            if str((a_, o_, s_)) not in valid_triplets_set:
                valid_triplets.append((a_, o_, s_))
                valid_triplets_set.add(str((a_, o_, s_)))
    return {
        'aspects': if_aspect.nonzero().squeeze().tolist(),  # for ATE
        'opinions': if_opinion.nonzero().squeeze().tolist(),  # for OTE
        'triplets': sorted(valid_triplets, key=lambda x: (x[0][0], x[0][-1], x[1][0], x[1][-1]))  # line 25
    }


def guarantee_list(l):
    if type(l) != list:
        l = [l]
    return l