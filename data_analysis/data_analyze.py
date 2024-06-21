"""
@Project  : t-conan
@File     : data_analyze.py
@Author   : Shaobo Cui
@Date     : 25.11.22 16:38
"""

import matplotlib.pyplot as plt
import pandas as pd

if __name__ == "__main__":
    # create data
    domains = ['Environment', 'Business', 'SciTech', 'Health', 'Work', 'Politics',
               'Education', 'Sports', 'Entertainment', 'Travel']
    x = domains
    y_instantly = [10, 20, 10, 30, 10, 10, 10, 10, 20, 10]
    y_seconds = [10, 20, 10, 30, 10, 10, 10, 10, 20, 10]
    y_minutes = [10, 20, 10, 30, 10, 10, 10, 10, 20, 10]
    y_hours = [10, 20, 10, 30, 10, 10, 10, 10, 20, 10]
    y_days = [10, 20, 10, 30, 10, 10, 10, 10, 20, 10]
    y_months = [10, 20, 10, 30, 10, 10, 10, 10, 20, 10]
    y_years = [10, 20, 10, 30, 10, 10, 10, 10, 20, 10]
    y_decades = [10, 20, 10, 30, 10, 10, 10, 10, 20, 10]
    y_centuries = [10, 20, 10, 30, 10, 10, 10, 10, 20, 10]

    time_domains = {'instantly': y_instantly,
                    'seconds': y_seconds,
                    'minutes': y_minutes,
                    'hours': y_hours,
                    'days': y_days,
                    'months': y_months,
                    'years': y_years,
                    'decades': y_decades,
                    'centuries': y_centuries}
    df = pd.DataFrame(time_domains, index=domains)
    ax = df.plot.bar(stacked=True)
    plt.xticks(rotation=75, fontsize=16)
    plt.yticks(fontsize=16)


    fig = plt.gcf()
    fig.set_size_inches(10, 6.18)
    plt.tight_layout()
    fig.savefig('time_domain.pdf')
