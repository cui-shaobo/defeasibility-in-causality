from collections import defaultdict

import pandas as pd
import pickle
from tqdm import tqdm

from CEQ import cs_sent, tokenize


ALPHA = 0.66
LAMBDA = 1

def calculate_ceq_score():
    f = open("data/causes.pkl", 'rb')
    causes = pickle.load(f)
    f.close()

    f = open("data/effects.pkl", 'rb')
    effects = pickle.load(f)
    f.close()

    df = pd.read_csv('CS_metric_copa.csv')
    ceq_scores = []
    for i in tqdm(range(df.shape[0])):
        cause_txt = tokenize(df.loc[i, 'cause'])
        effect_txt = tokenize(df.loc[i, 'effect'])
        cs_tmp_1 = cs_sent(cause_txt, effect_txt, causes, effects, ALPHA=ALPHA, LAMBDA=LAMBDA)
        ceq_scores.append(cs_tmp_1)
    df['ceq'] = ceq_scores
    df.to_csv('copa_ceq_scores.csv', index=False)

if __name__ == '__main__':
    res = defaultdict(lambda: [0, 0])
    df = pd.read_csv('copa_ceq_scores.csv')
    scores, labels, ids = df['ceq'].tolist(), df['label'].tolist(), df['id'].tolist()
    for score, label, id in zip(scores, labels, ids):
        res[id][int(label)] = score

    corr = 0
    for score1, score2 in res.values():
        corr += score1 < score2

    print(corr / 1000)
