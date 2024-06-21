import torch
import sys

from transformers import BertTokenizer

from cesar import CESAR

if __name__ == "__main__":
    if len(sys.argv) == 3:
        s1 = sys.argv[1]
        s2 = sys.argv[2]
    else:
        raise ValueError(f'Two statements expected as arguments, {len(sys.argv) - 1} given.')

    tokenizer = BertTokenizer.from_pretrained('bert-large-uncased', 
                                            do_lower_case=True)

    # find available device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # load CESAR model
    model = CESAR("bert-large-uncased", causal_attention=True)
    model.to(device)
    model.load_state_dict(torch.load('models/cesar_model.pt'))

    encoded_inputs = tokenizer(s1, 
        s2, 
        add_special_tokens=True, 
        truncation=True,
        padding="max_length",
        max_length=128,
        return_tensors="pt")

    input_ids = encoded_inputs['input_ids'].to(device)
    attention_masks = encoded_inputs['attention_mask'].to(device)
    token_type_ids = encoded_inputs['token_type_ids'].to(device)

    model.eval()
    score = model(input_ids, attention_masks, token_type_ids)
    print(f'Causal strength between C="{s1}" and E="{s2}" is {score.cpu().detach().item()}')