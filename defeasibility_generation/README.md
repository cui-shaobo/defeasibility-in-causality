# Defeasibility Generation

The code is written in Python 3.8.16 and uses libraries in the `requirements.txt`.

## Generate Supporter or Defeater

To generate the supporter or the defeater, use the following command:

```bash
python defeasibility_causality_generation.py\
    -r supporter\
    --train_path data/shuffled_train.csv\
    --val_path data/shuffled_val_extended.csv\
    --test_path data/shuffled_test_extended.csv\
    -m bart\
    -bsz 8\
    -p experiments/defeasibility_generation/supporter/bart/predictions.csv\
    -o output_bart_a
    
rm -rf output_bart_a
```

, where `-r` is the role of the generated defeasibility (`supporter` or `defeater`), `-m` is the model name (`bart`, `gpt2`, `t5`, or `t5l` T5-large), `-bsz` is the batch size, `-p` is the path to the predictions file, and `-o` is the output directory. Note that if you use T5-large, you may add a `-pp` to do the parallel training to reduce the memory load.

## Evaluate the Generated Defeasibility

To evaluate the generated defeasibility, use the following command:

```bash
python run_eval.py\
    --file experiments/defeasibility_generation/{reason_type}/{model_name}/predictions.csv\
    --hypos-col prediction\
    --refs-cols ground_truth
```

Replace `{reason_type}` with `supporter` or `defeater`, and `{model_name}` with `bart`, `gpt2`, `t5`, or `t5l` T5-large.