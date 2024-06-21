import torch
import pandas as pd

from transformers import BertTokenizer
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

from preprocessing import get_test_tokens, get_train_tokens
from cesar import CESAR
from training_utils import set_seed, training, evaluate

if __name__ == "__main__":
    batch_size = 16
    tokenizer = BertTokenizer.from_pretrained('bert-large-uncased', 
                                            do_lower_case=True)
    
    # train set
    input_ids, attention_masks, token_type_ids, labels = get_train_tokens(tokenizer, path='data/CS_metric_train_full.csv')
    train_data = TensorDataset(input_ids, attention_masks, token_type_ids, labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    # validiation set
    input_ids, attention_masks, token_type_ids, labels = get_train_tokens(tokenizer, path='data/CS_metric_copa.csv')
    copa_data = pd.read_csv('data/CS_metric_copa.csv')
    ids = torch.tensor(copa_data['id'])
    val_data = TensorDataset(input_ids, attention_masks, token_type_ids, labels, ids)
    val_sampler = RandomSampler(val_data)
    val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=batch_size)

    # test set
    input_ids, attention_masks, token_type_ids = get_test_tokens(tokenizer)
    test_data = TensorDataset(input_ids, attention_masks, token_type_ids)
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)

    # find available device
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

    # load CESAR model
    model = CESAR("bert-large-uncased", causal_attention=True)
    model.to(device)
    

    # training loop
    set_seed(42) 
    model = training(model, train_dataloader, epochs=2, val_dataloader=val_dataloader, test_dataloader=test_dataloader, device=device)

    print('-' * 25)
    print('Evaluate final model')
    evaluate(model, val_dataloader, test_dataloader, verbose=True, device=device)

    print('Save model...')
    torch.save(model.state_dict(), 'models/model.pt')

    