"""
@Project  : t-conan
@File     : align_domain.py
@Author   : Shaobo Cui
@Date     : 10.02.23 06:37
"""
import pandas as pd

if __name__ == "__main__":
    filename = 'final-td-ccr-8080.csv'
    df = pd.read_csv(filename)
    domains = [['Environment', 'environment'], ['Business', 'business'],
               ['Science/Technology', 'scitech'], ['Health', 'health'],
               ['Work', 'work'], ['Politics', 'politics'],
               ['Education', 'education'], ['Sports', 'sports'],
               ['Entertainment', 'entertainment'], ['Travel', 'travel']]
    domains_dict = {}
    for domain in domains:
        domains_dict[str(domain)] = domain[0]

    for i, row in df.iterrows():
        domain_name = row['domain']
        for domain in domains:
            if domain_name in domain:
                df.at[i, 'domain'] = domain[0]

    df.to_csv('td-ccr-domain_aligned.csv', index=False)
