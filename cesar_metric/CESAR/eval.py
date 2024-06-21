import torch
import pandas as pd
import sys

from transformers import BertTokenizer
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

from preprocessing import get_test_tokens, get_train_tokens
from cesar import CESAR
from training_utils import evaluate

if __name__ == "__main__":
    if len(sys.argv) == 3:
        PATH_VAL = sys.argv[1]
        PATH_TEST = sys.argv[2]
    else:
        PATH_VAL = 'data/CS_metric_copa.csv'
        PATH_TEST = 'data/CS_metric_test.csv'

    batch_size = 16
    tokenizer = BertTokenizer.from_pretrained('bert-large-uncased', 
                                            do_lower_case=True)

    # validiation set
    input_ids, attention_masks, token_type_ids, labels = get_train_tokens(tokenizer, path=PATH_VAL)
    copa_data = pd.read_csv(PATH_VAL)
    ids = torch.tensor(copa_data['id'])
    val_data = TensorDataset(input_ids, attention_masks, token_type_ids, labels, ids)
    val_sampler = RandomSampler(val_data)
    val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=batch_size)

    # test set
    input_ids, attention_masks, token_type_ids = get_test_tokens(tokenizer, path=PATH_TEST)
    test_data = TensorDataset(input_ids, attention_masks, token_type_ids)
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)

    # find available device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # load CESAR model
    model = CESAR("bert-large-uncased", causal_attention=True)
    model.to(device)
    model.load_state_dict(torch.load('models/cesar_model.pt'))

    print('-' * 25)
    evaluate(model, val_dataloader, test_dataloader, verbose=True, device=device)

    