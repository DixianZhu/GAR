# FAR
Function Aligned Regression

FAR motivated by the idea that Regression loss function should not only learns the function value but also function derivatives. The conventional regression loss only focuses on function values.


## Common usage examples:
### On tabular datasets:
python3 main.py --loss=FAR --dataset=wine_quality --lr=1e-2 --decay=1e-4
python3 main.py --loss=MAE --dataset=wine_quality --lr=1e-2 --decay=1e-4

### On Image dataset (AgeDB Scratch or Linear Probe Based on RNC)
Make sure you have AgeDB data under your data folder.
- From scratch:  python3 ageDB_scratch.py --alpha=0.1 --learning_rate=0.5 --weight_decay=1e-4 --loss=FAR
- Linear probe:  python3 ageDB_linear.py --alpha=0.1 --learning_rate=0.05 --weight_decay=1e-4 --loss=FAR --ckpt='path-to-pretrained-model'

