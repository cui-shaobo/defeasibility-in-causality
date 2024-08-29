# Exploring Defeasibility in Causal Reasoning

This is the official implementation for our paper in ACL: Exploring Defeasibility in Causal Reasoning

Defeasibility in causal reasoning implies that the causal relationship between cause and effect can be strengthened or weakened. Namely, the causal strength between cause and effect should increase or decrease with the incorporation of strengthening arguments (supporters) or weakening arguments (defeaters), respectively. However, existing works ignore defeasibility in causal reasoning and fail to evaluate existing causal strength metrics in defeasible settings. In this work, we present $\delta$-CAUSAL, the first benchmark dataset for studying defeasibility in causal reasoning. $\delta$-CAUSAL includes around 11K events spanning ten domains, featuring defeasible causality pairs, namely, cause-effect pairs accompanied by supporters and defeaters. We further show that current causal strength metrics fail to reflect the change of causal strength with the incorporation of supporters or defeaters in $\delta$-CAUSAL. To this end, we propose CESAR (Causal Embedding aSsociation with Attention Rating), a metric that measures causal strength based on token-level causal relationships. CESAR achieves a significant 69.7{\%} relative improvement over existing metrics, increasing from 47.2{\%} to 80.1{\%} in capturing the causal strength change brought by supporters and defeaters. We further demonstrate even Large Language Models (LLMs) like GPT-3.5 still lag 4.5 and 10.7 points behind humans in generating supporters and defeaters, emphasizing the challenge posed by $\delta$-CAUSAL.


## Code Structure
+ `delta-causal-dataset/*`: the final dataset. 
+ `data_analysis/*`: analysis for our collected dataset.
+ `cesar_metric/*`: the metric for causal strength. 
+ `evaluation_metrics/*`: existing evaluation metrics. 
+ `metrics_outputs/*`: the outputs of existing metrics on causal strength and CESAR.
+ `defeasibility_generation/*`: code for defeasibility generation.

## Citation

If you want to cite our  dataset and paper, you can use this BibTex:
```bibtext
@inproceedings{cui-etal-2024-exploring,
    title = "Exploring Defeasibility in Causal Reasoning",
    author = "Cui, Shaobo  and
      Milikic, Lazar  and
      Feng, Yiyang  and
      Ismayilzada, Mete  and
      Paul, Debjit  and
      Bosselut, Antoine  and
      Faltings, Boi",
    editor = "Ku, Lun-Wei  and
      Martins, Andre  and
      Srikumar, Vivek",
    booktitle = "Findings of the Association for Computational Linguistics ACL 2024",
    month = aug,
    year = "2024",
    address = "Bangkok, Thailand and virtual meeting",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.findings-acl.384",
    pages = "6433--6452",
}

```
