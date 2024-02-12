# FAR 
Function Aligned Regression.

FAR motivated by the idea that Regression loss function should not only learns the function value but also function derivatives. The conventional regression loss only focuses on function values.

Prerequisite: torch==2.0.0 

## Common usage examples:
### On tabular datasets:
- python3 main.py --loss=FAR --dataset=wine_quality --lr=1e-2 --decay=1e-4
- python3 main.py --loss=MAE --dataset=wine_quality --lr=1e-2 --decay=1e-4

### On Image dataset (AgeDB Scratch or Linear Probe Based on RNC):
Make sure you have AgeDB data and pass it to the code by '--data_folder'.
- From scratch:  python3 ageDB_scratch.py --alpha=10 --learning_rate=0.5 --weight_decay=1e-4 --loss=FAR --data_folder='your-AgeDB-folder'
- Linear probe:  python3 ageDB_linear.py --alpha=0.1 --learning_rate=0.05 --weight_decay=1e-4 --loss=FAR --data_folder='your-AgeDB-folder' --ckpt='path-to-pretrained-model'

We thank the <a href="https://github.com/kaiwenzha/Rank-N-Contrast">previous work</a> that provides general experimental settings for AgeDB.

## Synthetic Experiments:
Please check synthetic.ipynb for how to run on the two synthetic (Sine and Squared Sine) datasets. 

Alternatively, there are google colab files for <a href="https://colab.research.google.com/drive/1czRNpSGAkQtFksK0JmToQLa0_csmfBmC?usp=drive_link">sine dataset</a> and <a href="https://colab.research.google.com/drive/1XeGWBWLs05mHzD67GItn47l-lB_7V-bb?usp=drive_link">squared sine dataset</a>. Please make sure upload FAR related code (loss.py, models.py, utils.py) and synthetic data (sine.npz, sq_sine.npz) to colab folder.

<p align="center">
  <img src="figures/sin_all_mean_cmp.jpg" width="800" title="Sine">
</p>
<p align="center">
  <img src="figures/sin_MAE.jpg" width="400" title="Sine">
  <img src="figures/sin_RNC.jpg" width="400" title="Sine">
</p>

<p align="center">
  <img src="figures/sin_APearson.jpg" width="400" title="Sine">
  <img src="figures/sin_FAR.jpg" width="400" title="Sine">
</p>

<p align="center">
  <img src="figures/sq_sin_all_mean_cmp.jpg" width="800" title="Squared Sine">
</p>
<p align="center">
  <img src="figures/sq_sin_MAE.jpg" width="400" title="Squared Sine">
  <img src="figures/sq_sin_RNC.jpg" width="400" title="Squared Sine">
</p>

<p align="center">
  <img src="figures/sq_sin_APearson.jpg" width="400" title="Squared Sine">
  <img src="figures/sq_sin_FAR.jpg" width="400" title="Squared Sine">
</p>


:page_with_curl: Citation
---------
If you find FAR useful in your work, please cite the following paper:
```
@misc{zhu2024function,
      title={Function Aligned Regression: A Method Explicitly Learns Functional Derivatives from Data.},
      author={Dixian Zhu, Livnat Jerby-Arnon},
      year={2024}
     }
```
