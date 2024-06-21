import torch
import matplotlib.pyplot as plt
import seaborn as sns

from transformers import BertTokenizer

from cesar import CESAR

if __name__ == "__main__":
    tokenizer = BertTokenizer.from_pretrained('bert-large-uncased', 
                                            do_lower_case=True)

    # find available device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # load CESAR model
    model = CESAR("bert-large-uncased", causal_attention=True, return_inner_scores=True)
    model.to(device)
    model.load_state_dict(torch.load('models/cesar_model.pt'))
    model.eval()

    s1 = "Fire starts."
    s2 = "House burns."

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

    score1, attn_map1, score_map1 = model(input_ids, attention_masks, token_type_ids)
    result1 = score_map1 * attn_map1
    
    s3 = "Fire does not start."
    s4 = "House burns."

    encoded_inputs = tokenizer(s3, 
        s4, 
        add_special_tokens=True, 
        truncation=True,
        padding="max_length",
        max_length=128,
        return_tensors="pt")

    input_ids = encoded_inputs['input_ids'].to(device)
    attention_masks = encoded_inputs['attention_mask'].to(device)
    token_type_ids = encoded_inputs['token_type_ids'].to(device)

    score2, attn_map2, score_map2 = model(input_ids, attention_masks, token_type_ids)
    result2 = score_map2 * attn_map2

    attn_map1[attn_map1 < 1e-3] = 0
    attn_map2[attn_map2 < 1e-3] = 0

    score_map1[score_map1 < 1e-3] = 0
    score_map2[score_map2 < 1e-3] = 0

    result1[result1 < 1e-3] = 0
    result2[result2 < 1e-3  ] = 0

    fig, axs = plt.subplots(2, 3, figsize=(10, 5.5))
    sns.heatmap(score_map1.detach().cpu().numpy(), yticklabels=['[CLS]', 'fire', 'starts', '.' ,'[SEP]'], xticklabels=['house', 'burns', '.', '[SEP]'],
            center=0.5, cmap=sns.color_palette("viridis", as_cmap=True), ax=axs[0, 0], annot=True).set_title("Embedding cosine similarities", fontsize=10)
    sns.heatmap(attn_map1.detach().cpu().numpy(), yticklabels=['[CLS]', 'fire', 'starts', '.', '[SEP]'], xticklabels=['house', 'burns', '.', '[SEP]'],
            center=0.5, cmap=sns.color_palette("viridis", as_cmap=True), ax=axs[0, 1], annot=True).set_title("Attention scores", fontsize=10)
    sns.heatmap(result1.detach().cpu().numpy(), yticklabels=['[CLS]', 'fire', 'starts', '.', '[SEP]'], xticklabels=['house', 'burns', '.', '[SEP]'],
                center=0.5, cmap=sns.color_palette("viridis", as_cmap=True), ax=axs[0, 2], annot=True).set_title(f"Causal strength: {score1.item():.2f}", fontsize=10)

    axs[0, 0].tick_params(axis='y', rotation=0)
    axs[0, 1].tick_params(axis='y', rotation=0)
    axs[0, 2].tick_params(axis='y', rotation=0)

    sns.heatmap(score_map2.detach().cpu().numpy(), yticklabels=['[CLS]', 'fire', 'does', 'not', 'start', '.', '[SEP]'], xticklabels=['house', 'burns', '.', '[SEP]'],
            center=0.5, cmap=sns.color_palette("viridis", as_cmap=True), ax=axs[1, 0], annot=True).set_title("Embedding cosine similarities", fontsize=10)
    sns.heatmap(attn_map2.detach().cpu().numpy(), yticklabels=['[CLS]', 'fire', 'does', 'not', 'start', '.', '[SEP]'], xticklabels=['house', 'burns', '.', '[SEP]'],
            center=0.5, cmap=sns.color_palette("viridis", as_cmap=True), ax=axs[1, 1], annot=True).set_title("Attention scores", fontsize=10)
    sns.heatmap(result2.detach().cpu().numpy(), yticklabels=['[CLS]', 'fire', 'does', 'not', 'start', '.', '[SEP]'], xticklabels=['house', 'burns', '.', '[SEP]'],
                center=0.5, cmap=sns.color_palette("viridis", as_cmap=True), ax=axs[1, 2], annot=True).set_title(f"Causal strength: {score2.item():.2f}", fontsize=10)

    fig.tight_layout()
    plt.savefig('figures/causal_heatmaps_with_dot.pdf')