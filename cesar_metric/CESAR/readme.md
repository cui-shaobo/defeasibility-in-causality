# CESAR metrics

The computation of the CESAR metric can be refered to our paper. 

In order to use the CESAR metric first load the CESAR model into the `models` directory from:

```
[Google Drive]: 
```
or train your own version by running the command:
```
python training.py
```

To evaluate the CESAR metric on the `COPA` and `TIDE` datasets, start the command:
```
python eval.py
```

To evaluate the CESAR metric on your own datasets give paths to the datasets on which testing the metric as arguments to the `eval.py`:
```
python eval.py [PATH_1] [PATH_2]
```

To evaluate the causal strength between to statements `C` and `E`, run the following command:
```
python eval_example.py [C] [E]
```
For instance, 
```
python eval_example.py "I am tired." "I am going to bed."
```
outputs: *Causal strength between C="I am tired." and E="I am going to bed." is 0.7213023900985718*

To reproduce the histogram demonstrating the causal strength shift from our paper, use the command:
```
python histogram_CS_shift.py    
```
Finally, to obtain the heatmap we presented during the case study in the paper, run the command:
```
python heat_map.py      
```


