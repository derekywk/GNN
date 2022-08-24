## Enhancing Fraudulent Review Detection Using Graph Neural Network with Gist Extraction
### Install the Environment
Install the necessary Python packages using Anaconda CLI

`conda env create -f environment.yml`
### Download and Process Dataset
#### Amazon Customer Review Dataset
###### (Target Project Dataset)
Watches/Video_Games/Shoes Datasets will be downloaded by function `load_dataset()` in `feature_process.py` using package `tensorflow_datasets`

Process the features by running `feature_process.py`.

The process includes extracting gist (keywords) by `important_keywords()`, training Word2Vec model by `train_word_2_vec_model()`
#### YelpChi Dataset
###### (Dataset for Verifying Our Implementation)
YelpChi Dataset are stored in `./data`

Process the features by running `data_process.py`
### Train a Classification Model
Train a GNN/Care-GNN/Random-forest Model by running `train.py`

Configure the hyper-parameters by providing necessary arguments and modifying variables
* `NORMALIZATION` ( None / row / max_column / sum_column )
* `USING_GIST_AS` ( None / feature / relation / feature and relation )
* `NUMBER_OF_GIST['Feature']`
* `NUMBER_OF_GIST['Relation']`
* `TOTAL_VOTES_GT_1`
* `GIST_ONLY`
* `GENUINE_THRESHOLD`
* `FRAUDULENT_THRESHOLD`
### Plot the Results
Plot the accuracy vs epoch by setting `PLOT = True` in `train.py`

Plot the comparison of different models and the statistics distribution of gist features by `plots.py`