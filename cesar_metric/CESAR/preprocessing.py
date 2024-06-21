import torch
import pandas as pd 

MAX_LEN = 128

def get_tokens(sentences, tokenizer):
    encoded_inputs = tokenizer(sentences, 
                    add_special_tokens=True, 
                    truncation=True,
                    padding="max_length",
                    max_length=MAX_LEN,
                    return_tensors="pt")
    
    input_ids = encoded_inputs['input_ids']
    attention_masks = encoded_inputs['attention_mask']
    return input_ids, attention_masks    

def get_train_tokens(tokenizer, path='data/CS_metric_train.csv', type_ids=True):
    train_data = pd.read_csv(path)

    sentences = list(train_data['cause'].to_numpy())
    second_sentences = list(train_data['effect'].to_numpy())
    encoded_inputs = tokenizer(sentences, 
        second_sentences, 
        add_special_tokens=True, 
        truncation=True,
        padding="max_length",
        max_length=MAX_LEN,
        return_tensors="pt")

    input_ids = encoded_inputs['input_ids']
    attention_masks = encoded_inputs['attention_mask']    
    labels = torch.tensor(train_data['label'], dtype=torch.float32)

    if type_ids:
        token_type_ids = encoded_inputs['token_type_ids']
        return input_ids, attention_masks, token_type_ids, labels
    
    return input_ids, attention_masks, labels


def get_test_tokens(tokenizer, path='data/CS_metric_test.csv', type_ids=True):
    test_data = pd.read_csv(path)

    sentences = list(test_data['cause'].to_numpy())
    second_sentences = list(test_data['effect'].to_numpy())
    encoded_inputs = tokenizer(sentences, 
        second_sentences, 
        add_special_tokens=True, 
        truncation=True,
        padding="max_length",
        max_length=MAX_LEN,
        return_tensors="pt")
    
    sentences = list(test_data['C+A'].to_numpy())
    second_sentences = list(test_data['effect'].to_numpy())
    encoded_inputs_assumption = tokenizer(sentences, 
        second_sentences, 
        add_special_tokens=True, 
        truncation=True,
        padding="max_length",
        max_length=MAX_LEN,
        return_tensors="pt")

    sentences = list(test_data['C+D'].to_numpy())
    second_sentences = list(test_data['effect'].to_numpy())
    encoded_inputs_defeater = tokenizer(sentences, 
        second_sentences, 
        add_special_tokens=True, 
        truncation=True,
        padding="max_length",
        max_length=MAX_LEN,
        return_tensors="pt")
    
    input_ids = torch.stack([encoded_inputs['input_ids'], encoded_inputs_assumption['input_ids'], encoded_inputs_defeater['input_ids']], dim=1)
    attention_masks = torch.stack([encoded_inputs['attention_mask'], encoded_inputs_assumption['attention_mask'], encoded_inputs_defeater['attention_mask']], dim=1)

    if type_ids:
        token_type_ids = torch.stack([encoded_inputs['token_type_ids'], encoded_inputs_assumption['token_type_ids'], encoded_inputs_defeater['token_type_ids']], dim=1)
        return input_ids, attention_masks, token_type_ids
    
    return input_ids, attention_masks
