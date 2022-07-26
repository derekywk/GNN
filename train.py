import time
import os
import random
import argparse
from sklearn.model_selection import train_test_split

from utils import *
from model import *
from layers import *
from graphsage import *

import pandas as pd
from utils import tprint
from collections import defaultdict
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

"""
	Training CARE-GNN
	Paper: Enhancing Graph Neural Network-based Fraud Detectors against Camouflaged Fraudsters
	Source: https://github.com/YingtongDou/CARE-GNN
"""

PLOT = True
DATASET = "Watches_v1_00"
# DATASET = "Shoes_v1_00"
DATASET_SIZE = -1 # -1 refers to whole dataset
USING_GIST_AS = [None, 'feature', 'relation'][0]
DF_FILE_NAME = f"df_{DATASET}_size_{DATASET_SIZE}.pkl.gz"
DF_FILE_NAME_WITH_FEATURES = f"df_{DATASET}_size_{DATASET_SIZE}_with_features.pkl.gz"

GENUINE_THRESHOLD = 0.7
FRAUDULENT_THRESHOLD = 0.3

WORD_2_VEC_MODEL_NAME = f"Word2Vec_{DATASET}_size_{DATASET_SIZE}"
df = pd.read_pickle(DF_FILE_NAME_WITH_FEATURES, compression={"method": "gzip", "compresslevel": 1})

parser = argparse.ArgumentParser()

# dataset and model dependent args
parser.add_argument('--data', type=str, default='yelp', help='The dataset name. [yelp, amazon, watch]')
parser.add_argument('--model', type=str, default='CARE', help='The model name. [CARE, SAGE]')
parser.add_argument('--inter', type=str, default='GNN', help='The inter-relation aggregator type. [Att, Weight, Mean, GNN]')
parser.add_argument('--batch-size', type=int, default=1024, help='Batch size 1024 for yelp, 256 for amazon.')

# hyper-parameters
parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
parser.add_argument('--lambda_1', type=float, default=2, help='Simi loss weight.')
parser.add_argument('--lambda_2', type=float, default=1e-3, help='Weight decay (L2 loss weight).')
parser.add_argument('--emb-size', type=int, default=64, help='Node embedding size at the last layer.')
parser.add_argument('--num-epochs', type=int, default=31, help='Number of epochs.')
parser.add_argument('--test-epochs', type=int, default=3, help='Epoch interval to run test set.')
parser.add_argument('--under-sample', type=int, default=1, help='Under-sampling scale.')
parser.add_argument('--step-size', type=float, default=2e-2, help='RL action step size')

# other args
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=5, help='Random seed.')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
print(f'run on {args.data}; USING_GIST_AS {USING_GIST_AS}; Random seed {args.seed}')

# load graph, feature, and label
if args.data in ['amazon', 'yelp']:
	[homo, relation1, relation2, relation3], feat_data, labels = load_data(args.data)
	relation_list = [relation1, relation2, relation3]
	tprint(f"Feature size", feat_data.shape)
elif args.data == 'watch':
	feature_list = ['f_user_MNR',
		   'f_user_PR', 'f_user_NR', 'f_user_avgRD', 'f_user_WRD', 'f_user_BST',
		   'f_user_ERD', 'f_user_ETG', 'f_user_RL', 'f_user_ACS', 'f_user_MCS',
		   'f_product_MNR', 'f_product_PR', 'f_product_NR', 'f_product_avgRD',
		   'f_product_WRD', 'f_product_ERD', 'f_product_ETG', 'f_product_RL',
		   'f_product_ACS', 'f_product_MCS', 'f_RANK', 'f_RD', 'f_EXT', 'f_DEV',
		   'f_ETF', 'f_ISR', 'f_L', 'f_PC', 'f_PCW', 'f_PP1', 'f_RES',
		   'f_SW', 'f_OW']
	if USING_GIST_AS == 'feature':
		feature_list.extend([col for col in df.columns if 'f_gist_' in col])

	target_df_indices = (~pd.isna(df['genuine'])) & (df['total_votes'] > 1)
	feat_data = df.loc[target_df_indices, feature_list].to_numpy(dtype=float)
	labels = (1 - df.loc[target_df_indices, 'genuine']).to_numpy(dtype=float)
	tprint(f"Feature size", feat_data.shape)

	# Build Relations
	tprint('Computing RUR...')
	customer_id_to_index_list = defaultdict(list)
	relation_RUR = defaultdict(set)
	for idx, customer_id in enumerate(df.loc[target_df_indices, 'customer_id'].to_list()):
		customer_id_to_index_list[customer_id].append(idx)
	for index_list in customer_id_to_index_list.values():
		s = set(index_list)
		for index in index_list:
			relation_RUR[index] = s
	tprint('Computing RSR...')
	pisr_to_index_list = defaultdict(list)
	relation_RSR = defaultdict(set)
	for idx, (product_id, star_rating) in enumerate(df.loc[target_df_indices, ['product_id', 'star_rating']].itertuples(index=False, name=None)):
		pisr_to_index_list[(product_id, star_rating)].append(idx)
	for index_list in pisr_to_index_list.values():
		s = set(index_list)
		for index in index_list:
			relation_RSR[index] = s
	tprint('Computing RTR...')
	piym_to_index_list = defaultdict(list)
	relation_RTR = defaultdict(set)
	for idx, (product_id, review_date) in enumerate(df.loc[target_df_indices, ['product_id', 'review_date']].itertuples(index=False, name=None)):
		piym_to_index_list[(product_id, review_date.year, review_date.month)].append(idx)
	for index_list in piym_to_index_list.values():
		s = set(index_list)
		for index in index_list:
			relation_RTR[index] = s
	relation_list = [relation_RUR, relation_RSR, relation_RTR]

	if USING_GIST_AS == 'relation':
		for col in [col for col in df.columns if 'f_gist_' in col]:
			tprint(f'Computing Relation for gist "{col[7:]}"...')
			gist_relation = defaultdict(set)
			rounded_distance = ((df.loc[target_df_indices, col].reset_index(drop=True)*2).round(1))/2
			for rounded_value in rounded_distance.unique():
				index_list = rounded_distance.index[rounded_distance==rounded_value]
				s = set(index_list)
				for index in index_list:
					gist_relation[index] = s
			relation_list.append(gist_relation)

# train_test split
np.random.seed(args.seed)
random.seed(args.seed)
if args.data == 'amazon':  # amazon
	# 0-3304 are unlabeled nodes
	index = list(range(3305, len(labels)))
	idx_train, idx_test, y_train, y_test = train_test_split(index, labels[3305:], stratify=labels[3305:],test_size=0.60,
															random_state=2, shuffle=True)
else:
	index = list(range(len(labels)))
	idx_train, idx_test, y_train, y_test = train_test_split(index, labels, stratify=labels, test_size=0.60,
															random_state=2, shuffle=True)

# split pos neg sets for under-sampling
train_pos, train_neg = pos_neg_split(idx_train, y_train)

# initialize model input
features = nn.Embedding(feat_data.shape[0], feat_data.shape[1])
feat_data = normalize(feat_data)
features.weight = nn.Parameter(torch.FloatTensor(feat_data), requires_grad=False)
if args.cuda:
	features.cuda()

# set input graph
if args.model == 'SAGE':
	adj_lists = homo
else:
	adj_lists = relation_list

print(f'Model: {args.model}, Inter-AGG: {args.inter}, emb_size: {args.emb_size}.')

# build one-layer models
if args.model == 'CARE':
	inter1 = InterAgg(features, feat_data.shape[1], args.emb_size, adj_lists, [IntraAgg(features, feat_data.shape[1], cuda=args.cuda) for i in range(len(adj_lists))],
					  inter=args.inter, step_size=args.step_size, cuda=args.cuda)
	# intra1 = IntraAgg(features, feat_data.shape[1], cuda=args.cuda)
	# intra2 = IntraAgg(features, feat_data.shape[1], cuda=args.cuda)
	# intra3 = IntraAgg(features, feat_data.shape[1], cuda=args.cuda)
	# inter1 = InterAgg(features, feat_data.shape[1], args.emb_size, adj_lists, [intra1, intra2, intra3], inter=args.inter,
	# 				  step_size=args.step_size, cuda=args.cuda)
elif args.model == 'SAGE':
	agg1 = MeanAggregator(features, cuda=args.cuda)
	enc1 = Encoder(features, feat_data.shape[1], args.emb_size, adj_lists, agg1, gcn=True, cuda=args.cuda)

if args.model == 'CARE':
	gnn_model = OneLayerCARE(2, inter1, args.lambda_1)
elif args.model == 'SAGE':
	# the vanilla GraphSAGE model as baseline
	enc1.num_samples = 5
	gnn_model = GraphSage(2, enc1)

if args.cuda:
	gnn_model.cuda()

optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, gnn_model.parameters()), lr=args.lr, weight_decay=args.lambda_2)
times = []
performance_log = []

tprint("Training Started...")
# train the model
for epoch in range(args.num_epochs):
	# randomly under-sampling negative nodes for each epoch
	sampled_idx_train = undersample(train_pos, train_neg, scale=1)
	rd.shuffle(sampled_idx_train)

	# send number of batches to model to let the RLModule know the training progress
	num_batches = int(len(sampled_idx_train) / args.batch_size) + 1
	if args.model == 'CARE':
		inter1.batch_num = num_batches

	loss = 0.0
	epoch_time = 0

	# mini-batch training
	for batch in range(num_batches):
		start_time = time.time()
		i_start = batch * args.batch_size
		i_end = min((batch + 1) * args.batch_size, len(sampled_idx_train))
		batch_nodes = sampled_idx_train[i_start:i_end]
		batch_label = labels[np.array(batch_nodes)]
		optimizer.zero_grad()
		if args.cuda:
			loss = gnn_model.loss(batch_nodes, Variable(torch.cuda.LongTensor(batch_label)))
		else:
			loss = gnn_model.loss(batch_nodes, Variable(torch.LongTensor(batch_label)))
		loss.backward()
		optimizer.step()
		end_time = time.time()
		epoch_time += end_time - start_time
		loss += loss.item()

	print(f'Epoch: {epoch}, loss: {loss.item() / num_batches}, time: {epoch_time}s')

	# testing the model for every $test_epoch$ epoch
	if epoch % args.test_epochs == 0:
		if args.model == 'SAGE':
			test_sage(idx_test, y_test, gnn_model, args.batch_size)
		else:
			gnn_auc, label_auc, gnn_recall, label_recall = test_care(idx_test, y_test, gnn_model, args.batch_size)
			performance_log.append([gnn_auc, label_auc, gnn_recall, label_recall])

if PLOT:
	performance_log = np.array(performance_log)
	fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(6, 8))
	axes[0].plot(np.arange(len(performance_log)), performance_log[:, 0], label='gnn_auc')
	axes[0].plot(np.arange(len(performance_log)), performance_log[:, 1], label='label_auc')
	axes[1].plot(np.arange(len(performance_log)), performance_log[:, 2], label='gnn_recall')
	axes[1].plot(np.arange(len(performance_log)), performance_log[:, 3], label='label_recall')
	axes[0].set_ylim(0, 1.0)
	axes[1].set_ylim(0, 1.0)
	axes[0].legend(loc="upper left")
	axes[1].legend(loc="upper left")
	plt.show()
print(f'run on {args.data}; Dataset size {feat_data.shape}; USING_GIST_AS {USING_GIST_AS}; Random seed {args.seed}')
print(*[f"{key}: {value} |" for key, value in args._get_kwargs()])
tprint("Finished")