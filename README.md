# GAR (Gradient Aligned Regression)
Official implementation for 'Gradient Aligned Regression via Pairwise Losses'.

Python version: 3.9.19

Prerequisite: torch==2.0.0 

## Common usage examples:
### Easy to apply on your own code:

```python
from loss import FAR

# define loss function with alpha hyper-parameter.
criterion = FAR(alpha=0.2)

# ground truths: [bs, label_dim]
truths = ...
# predictions: [bs, label_dim]
preds = ...

# compute FAR loss
loss = criterion(preds, truths)
```
### On tabular datasets:
- <code> python3 main.py --loss=FAR --dataset=wine_quality --lr=1e-2 --decay=1e-4 </code>
- <code> python3 main.py --loss=MAE --dataset=wine_quality --lr=1e-2 --decay=1e-4 </code>

### On Image dataset (AgeDB Scratch or Linear Probe Based on RNC):
Make sure you have AgeDB data and pass it to the code by '--data_folder'.
- From scratch:  <code> python3 ageDB_scratch.py --alpha=0.1 --learning_rate=0.5 --weight_decay=1e-4 --loss=FAR --data_folder='your-AgeDB-folder' </code>
- Linear probe:  <code> python3 ageDB_linear.py --alpha=0.1 --learning_rate=0.05 --weight_decay=1e-4 --loss=FAR --data_folder='your-AgeDB-folder' --ckpt='path-to-pretrained-model' </code>

We thank the <a href="https://github.com/kaiwenzha/Rank-N-Contrast">previous work</a> that provides general experimental settings for AgeDB.

## Synthetic Experiments:
Please check synthetic.ipynb for how to run on the two synthetic (Sine and Squared Sine) datasets. 


