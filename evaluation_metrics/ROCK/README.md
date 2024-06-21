# ROCK Evaluation

The code is written in Python 3.8.16 and uses libraries in the `requirements.txt`.

## Prepare the Data

We use and adapt the code from the [original repository](https://github.com/zjiayao/ccr_rock). The difference is that we use the GPT-2 model instead of GPT-J-6B for event sampler because GPT-J is too big.

First run the following command to prepare the data:

```bash
python rock_preprocessing.py\
    --data_path ./data/shuffled_train.csv\
    --output_path ./experiments/rock/
```

# Evaluate the Causal Strength Metric

Then run the cells of the notebook `rock_evaluation.ipynb` to evaluate this causal strength metric.
