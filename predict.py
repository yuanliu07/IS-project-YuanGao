import os
import time
import torch
from transformers import BertTokenizer
from torch.utils.data import DataLoader

from ASTE_dataloader import ASTE_Dataset, ASTE_collate_fn, load_vocab
from scheme.span_tagging import form_label_id_map, form_sentiment_id_map
from evaluate_0423 import evaluate_model, print_evaluate_dict


def predict(model_path, version='3D', dataset='14lap', saved_file=None,
            batch_size=16, device='cuda',
            pretrained_model='bert-base-uncased',
            dataset_dir='./data/ASTE-Data-V2-EMNLP2020'):
    tokenizer = BertTokenizer.from_pretrained(pretrained_model)
    dataset_dir = dataset_dir + '/' + dataset
    vocab = load_vocab(dataset_dir=dataset_dir)

    label2id, id2label = form_label_id_map(version)
    senti2id, id2senti = form_sentiment_id_map()
    vocab['label_vocab'] = dict(label2id=label2id, id2label=id2label)
    vocab['senti_vocab'] = dict(senti2id=senti2id, id2senti=id2senti)

    base_model = torch.load(model_path).to(device)

    test_dataset = ASTE_Dataset(file_name=os.path.join(dataset_dir, 'test_triplets.txt'),
                                vocab=vocab,
                                tokenizer=tokenizer)

    test_dataloader = DataLoader(test_dataset,
                                 batch_size=batch_size,
                                 collate_fn=ASTE_collate_fn,
                                 shuffle=False)

    start_time = time.time()

    _, test_results = evaluate_model(base_model, test_dataset, test_dataloader,
                                     id2senti=id2senti,
                                     device=device,
                                     version=version,
                                     saved_file=saved_file)
    print('Time in {:.3f}s'.format(time.time() - start_time))
    print_evaluate_dict(test_results)
    return test_results


if __name__ == '__main__':

    # Example here
    # model_path = 'saved_models/16res/16res_1D_True_best.pkl'
    # version = '1D'
    # dataset = '16res'
    #
    # predict(model_path, version=version, dataset= dataset,
    #             saved_file = None,
    #             batch_size = 16,
    #             device = 'cuda',
    #             pretrained_model = 'bert-base-uncased',
    #             dataset_dir = './data/ASTE-Data-V2-EMNLP2020')

    datasets = ['15res', '16res', '14lap']
    versions = ['1D', '2D', '3D']
    batch_size = 16
    device = 'cuda'
    pretrained_model = 'bert-base-uncased'
    dataset_dir = './data/ASTE-Data-V2-EMNLP2020'

    results = {}

    # Iterate over each dataset and version
    for dataset in datasets:
        for version in versions:
            model_path = f'saved_models/{dataset}/{dataset}_{version}_True_best.pkl'
            saved_file = f'results_{dataset}_{version}.txt'

            print(f"Predicting for {dataset}_{version}...")
            predict(model_path, version=version, dataset=dataset,
                    saved_file=saved_file,
                    batch_size=batch_size,
                    device=device,
                    pretrained_model=pretrained_model,
                    dataset_dir=dataset_dir)
