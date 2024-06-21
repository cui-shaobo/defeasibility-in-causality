import torch
import matplotlib.pyplot as plt
import seaborn as sns

from transformers import BertTokenizer
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

from preprocessing import get_test_tokens, get_train_tokens
from cesar import CESAR


if __name__ == "__main__":
    batch_size = 16
    tokenizer = BertTokenizer.from_pretrained('bert-large-uncased', 
                                            do_lower_case=True)

    # test set
    input_ids, attention_masks, token_type_ids = get_test_tokens(tokenizer)
    test_data = TensorDataset(input_ids, attention_masks, token_type_ids)
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)

    # find available device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # load CESAR model
    model = CESAR("bert-large-uncased", causal_attention=True)
    model.to(device)
    model.load_state_dict(torch.load('models/cesar_model.pt'))

    model.eval()
    scores = {0: [], 1: [], 2: []}
    for batch in test_dataloader:
        b_input_ids, b_attn_mask, b_token_type_ids = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            base_scores = model(b_input_ids[:, 0, :], b_attn_mask[:, 0, :], b_token_type_ids[:, 0, :])
            assumption_scores = model(b_input_ids[:, 1, :], b_attn_mask[:, 1, :], b_token_type_ids[:, 1, :])
            defeater_scores = model(b_input_ids[:, 2, :], b_attn_mask[:, 2, :], b_token_type_ids[:, 2, :])
        
        scores[0].append(base_scores)
        scores[1].append(assumption_scores)
        scores[2].append(defeater_scores)

    base_scores = torch.concatenate(scores[0])
    assumption_scores = torch.concatenate(scores[1])
    defeater_scores = torch.concatenate(scores[2])

    defeater_scores = defeater_scores.detach().cpu().numpy()
    assumption_scores = assumption_scores.detach().cpu().numpy()
    base_scores = base_scores.detach().cpu().numpy()

    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot the kernel density estimates for each dataset
    sns.kdeplot(data=base_scores, shade=True, label='Cause -> Effect', ax=ax)
    sns.kdeplot(data=assumption_scores, shade=True, label='Cause + Assumption -> Effect', ax=ax)
    sns.kdeplot(data=defeater_scores, shade=True, label='Cause + Defeater -> Effect', ax=ax)

    # Add labels and legend
    ax.set_xlabel('Causal strength given by CESAR', fontsize=18)
    ax.set_ylabel('Density', fontsize=18)
    ax.set_title('')
    legend = ax.legend(bbox_to_anchor=(0.01, 1), loc='upper left', fontsize=17)
    fig.tight_layout()

    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)

    plt.savefig('figures/causal_strength_shift_cesar.pdf')