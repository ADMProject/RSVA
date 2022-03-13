# RSVA
RSVA: Recommender System using Variational Autoencoders

This project is a part of the course CS 247: Advanced Data Mining.

To run the code, refer to the notebook experiments.ipynb.

## Dataset:

The required datasets for running the notebook can be downloaded from:
MovieLens-20M (ML-20M) : https://drive.google.com/drive/folders/1wl-YGjVhY7S9fbswdu8moWkoidguMC1O?usp=sharing

Million Songs Dataset: https://drive.google.com/drive/folders/1zpVz-ojS-B0O7e2AAB1JY4DoSTdJr8Ei?usp=sharing

## Dataset Preparation:

The first step towards running the code, is dataset preparation. Refer to data.py and make the following mentioned changes.

1. Change the directory DATA_DIR mentioned on line 116 to the folder where your data is located.
2. Uncomment line 138 if you want to keep data with ratings greater than 3.5
3. Change the min_uc and min_sc parameters on line 70 as desired. min_uc refers to the users having atleast min_uc number of items in their watch history. min_sc refers to the items used/watched by atleast min_sc users. 

The data obtained after using data.py on the 2 datasets is present in the pro_sg folder present in the corresponding dataset folders shared above. For ML-20M dataset we had set, min_uc = 5 and min_sc = 4. For Million Songs Dataset, we used min_uc = 20 and min_sc = 200.

You need not run dataset.py again for these datasets.

Command for running data.py:
python data.py


## Training and Inferencing:

Refer to main.py script for training and inferencing from the pipeline.

The important parameters taken by the main.py script are as follows:
- cuda: For enabling operations using a GPU
- anneal_cap: Cap on the annealing value
- total_anneal_steps: Total number of steps on which we divide the anneal_cap
- plots: Folder where we store the data obtained after training the model. To get the plots refer to tensorboard documentation. You can use the folder generated in 'runs' folder to get the graphs and analyze data.
- lr: Learning rate
- epochs: Number of epochs

Update 'lossf' variable present on line 212 in models.py for chaging the metric to calculate divergence. By default the value is set to 'KLD' corresponding to KL Divergence. The other options are 'MMD', 'SWD' and 'CWD'.

Once, the parameters are set, you can run the code.

For example: python main.py --cuda --anneal_cap 1 --total_anneal_steps 20000 --plots swd-ml20m-epoch100-anneal1 --lr 1e-3 --epochs 100
