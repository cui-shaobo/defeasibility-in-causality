import torch
import torch.nn as nn
import numpy as np
import random
import copy

from collections import defaultdict
from transformers import get_linear_schedule_with_warmup

def set_seed(seed_value=42):
    """Set seed for reproducibility.
    """
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)

def evaluate(model, val_dataloader, test_dataloader, device='cpu', verbose=True):
    corr_assumption = 0
    corr_defeater = 0
    test_size = 0
    model.eval()
    for batch in test_dataloader:
        b_input_ids, b_attn_mask, b_token_type_ids = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            base_scores = model(b_input_ids[:, 0, :], b_attn_mask[:, 0, :], b_token_type_ids[:, 0, :])
            assumption_scores = model(b_input_ids[:, 1, :], b_attn_mask[:, 1, :], b_token_type_ids[:, 1, :])
            defeater_scores = model(b_input_ids[:, 2, :], b_attn_mask[:, 2, :], b_token_type_ids[:, 2, :])

        corr_assumption += torch.sum(assumption_scores > base_scores)
        corr_defeater += torch.sum(defeater_scores < base_scores)
        test_size += len(assumption_scores)
        
    res = defaultdict(lambda : [0, 0])
    for batch in val_dataloader:
        b_input_ids, b_attn_mask, b_token_type_ids, b_labels, b_ids = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            scores = model(b_input_ids, b_attn_mask, b_token_type_ids)
            for score, label, id in zip(scores, b_labels, b_ids):
                res[id.item()][int(label.item())] = score.item()

    corr = 0
    for score1, score2 in res.values():
        corr += score1 < score2
    val_score = corr / len(res)
    
    assumption = (corr_assumption / test_size).item()
    defeater = (corr_defeater / test_size).item()
    
    if verbose:
        print(f'COPA accuracy: {val_score * 100:.1f}%')
        print(f'Assumption: {assumption * 100:.1f}%')
        print(f'Defeater: {defeater * 100:.1f}%')

    return assumption, defeater, np.sqrt(assumption * defeater)



def training(model, train_dataloader, epochs=4, val_dataloader=None, test_dataloader=None, device='cpu', best_res=0.0):
    print("Start training...\n")
    best_model_wts = copy.deepcopy(model.state_dict())
    best_gm_mean = best_res

    # loss function
    loss_fn = nn.MSELoss()

    optimizer = torch.optim.AdamW(model.parameters(),
                    lr=1e-5,    # Default learning rate
                    eps=1e-8    # Default epsilon value
                  )

    # Total number of training steps
    total_steps = len(train_dataloader) * epochs

    # Set up the learning rate scheduler
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                              num_warmup_steps=0, # Default value
                                              num_training_steps=total_steps)

    for epoch_i in range(1, epochs + 1):
        model.train()

        total_loss, batch_loss, batch_counts = 0, 0, 0
        for step, batch in enumerate(train_dataloader):
            batch_counts += 1
            # Load batch to GPU
            b_input_ids, b_attn_mask, b_token_type_ids, b_labels = tuple(t.to(device) for t in batch)

            # compute scores that model predicts
            scores = model(b_input_ids, b_attn_mask, b_token_type_ids)
            # Compute loss and accumulate the loss values
            loss = loss_fn(scores, b_labels)
                
            batch_loss += loss.item()
            total_loss += loss.item()

            # Perform a backward pass to calculate gradients
            loss.backward()

            # Clip the norm of the gradients to 1.0 to prevent "exploding gradients"
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update parameters and the learning rate
            optimizer.step()
            scheduler.step()

            # Print the loss values and time elapsed for every 20 batches
            if (step % 50 == 0 and step != 0) or (step == len(train_dataloader) - 1):
                print(f"Epoch: {epoch_i} Step: {step} Loss: {batch_loss / batch_counts:.6f}")
                batch_loss, batch_counts = 0, 0
    
        avg_train_loss = total_loss / len(train_dataloader)
        print("-" * 70)
        print(f"End Epoch: {epoch_i} Average loss {avg_train_loss:.6f}")

        if (test_dataloader is not None) and (val_dataloader is not None):
            assumption_acc, defeater_acc, gm_mean = evaluate(model, val_dataloader, test_dataloader, device=device, verbose=True)
            if best_gm_mean < gm_mean:
                best_gm_mean = gm_mean
                best_model_wts = copy.deepcopy(model.state_dict())

        print("-" * 70)
        print()

    print(f'Best GM score: {best_gm_mean:.4f}')
    model.load_state_dict(best_model_wts)
    return model