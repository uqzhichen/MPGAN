*****MPGAN*****
---------------
code for the paper in ACM Multimedia 2020: Rethinking Generative Zero-Shot Learning: An Ensemble Learning Perspective for Recognising Visual Patches

*dataset:*
---------------
You can download the dataset [CUBird and NABird](https://drive.google.com/open?id=1YUcYHgv4HceHOzza8OGzMp092taKAAq1)
put the uncompressed dta to the folder "data"

***Raw wikipedia artical data:***
---------------
Raw wikipedia article data of CUBird and NABird, as well as detailed merging information of NABird, can be obtained [here](https://drive.google.com/open?id=0B_8vkk7CF-pwckxLQTVkcDBadGc).


***Reproduce results***
---------------
****CUBird SCS mode && SCE mode****

python run_trainer.py --splitmode easy --dataset CUB2011

python run_trainer.py --splitmode hard --dataset CUB2011


****NABird SCS mode && SCE mode****
python run_trainer.py --splitmode easy --dataset NABird

python run_trainer.py --splitmode hard --dataset NABird
