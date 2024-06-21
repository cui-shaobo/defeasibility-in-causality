"""
@Project  : t-conan
@File     : plot_line.py
@Author   : Shaobo Cui
@Date     : 28.02.23 19:23
"""
import collections
import re
from collections import Counter

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def obtain_length_distributions(df, column_name):
    str_list = df[column_name].tolist()
    lens = [len(re.findall(r'\w+', test_string)) for test_string in str_list]
    ratio_dict = dict(Counter(lens))
    ratio_dict = collections.OrderedDict(sorted(ratio_dict.items()))
    x = list(ratio_dict.keys())
    y = list(ratio_dict.values())
    y = np.array(y) / sum(y)
    # print(len(x), len(y))

    return x, y

def obtain_average_length(lengths, ratios):
    # print(len(lengths), len(ratios))
    return np.multiply(lengths, ratios).sum()


if __name__ == "__main__":
    df = pd.read_csv('td-ccr-final.csv')
    x_cause, y_cause = obtain_length_distributions(df, 'cause')
    x_short, y_short = obtain_length_distributions(df, 'short_term_effect')
    x_long, y_long = obtain_length_distributions(df, 'long_term_effect')
    x_defeater, y_defeater = obtain_length_distributions(df, 'defeater')
    x_assumption, y_assumption = obtain_length_distributions(df, 'assumption')

    cause_average_length = obtain_average_length(x_cause, y_cause)
    short_average_length = obtain_average_length(x_short, y_short)
    long_average_length = obtain_average_length(x_long, y_long)
    defeater_average_length = obtain_average_length(x_defeater, y_defeater)
    assumption_average_length = obtain_average_length(x_assumption, y_assumption)

    print('cause: {}\nshort: {}\nlong: {}\ndefeater: {}\nassumption: {}'.format(cause_average_length,
                                                                                short_average_length,
                                                                                long_average_length,
                                                                                defeater_average_length,
                                                                                assumption_average_length))
    # plt.plot(x_cause, y_cause, label='cause')
    plt.figure(figsize=(10, 6.18))

    # plt.plot(x_short, y_short, label='short-term effect', linewidth=4)
    # plt.plot(x_long, y_long, label='long-term effect', linewidth=4)

    plt.plot(x_defeater, y_defeater, label='defeater', linewidth=4)
    plt.plot(x_assumption, y_assumption, label='assumption', linewidth=4)
    plt.legend(fontsize=20)
    plt.xlim([0, 30])
    plt.xlabel('Sentence length', fontsize=20)
    plt.ylabel('Density', fontsize=20)


    plt.savefig('defeater_assumption.pdf')


