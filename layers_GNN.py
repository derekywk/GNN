import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch.autograd import Variable
from utils import tprint
from datetime import datetime

from operator import itemgetter
import math

"""
	CARE-GNN Layers
	Paper: Enhancing Graph Neural Network-based Fraud Detectors against Camouflaged Fraudsters
	Source: https://github.com/YingtongDou/CARE-GNN
"""


class GNNInterAgg(nn.Module):

	def __init__(self, features, feature_dim,
				 embed_dim, adj_lists, intraggs,
				 inter='GNN', cuda=True):
		"""
		Initialize the inter-relation aggregator
		:param features: the input node features or embeddings for all nodes
		:param feature_dim: the input dimension
		:param embed_dim: the output dimension
		:param adj_lists: a list of adjacency lists for each single-relation graph
		:param intraggs: the intra-relation aggregators used by each single-relation graph
		:param inter: the aggregator type: 'Att', 'Weight', 'Mean', 'GNN'
		:param cuda: whether to use GPU
		"""
		super(GNNInterAgg, self).__init__()

		self.features = features
		self.dropout = 0.6
		self.adj_lists = adj_lists
		self.intra_aggs = intraggs
		self.embed_dim = embed_dim
		self.feat_dim = feature_dim
		self.inter = inter if inter != 'GNN' else 'Mean'
		self.cuda = cuda
		for intra_agg in self.intra_aggs:
			intra_agg.cuda = cuda

		# number of batches for current epoch, assigned during training
		self.batch_num = 0

		# the activation function used by attention mechanism
		self.leakyrelu = nn.LeakyReLU(0.2)

		# parameter used to transform node embeddings before inter-relation aggregation
		self.weight = nn.Parameter(torch.FloatTensor(self.feat_dim, self.embed_dim))
		init.xavier_uniform_(self.weight)

		# weight parameter for each relation used by CARE-Weight
		self.alpha = nn.Parameter(torch.FloatTensor(self.embed_dim, len(intraggs)))
		init.xavier_uniform_(self.alpha)

		# parameters used by attention layer
		self.a = nn.Parameter(torch.FloatTensor(2 * self.embed_dim, 1))
		init.xavier_uniform_(self.a)

		# initialize the parameter logs
		self.weights_log = []

	def forward(self, nodes):
		"""
		:param nodes: a list of batch node ids
		:return combined: the embeddings of a batch of input node features
		:return center_scores: the label-aware scores of batch nodes
		"""

		# extract 1-hop neighbor ids from adj lists of each single-relation graph
		to_neighs = []
		for adj_list in self.adj_lists:
			to_neighs.append([set(adj_list[int(node)]) for node in nodes])

		# intra-aggregation steps for each relation
		# Eq. (8) in the paper
		relation_feats = []
		for i in range(len(to_neighs)):
			feats = self.intra_aggs[i].forward(to_neighs[i])
			relation_feats.append(feats)

		# concat the intra-aggregated embeddings from each relation
		neigh_feats = torch.cat(tuple(relation_feats), dim=0)
		# neigh_feats = torch.cat((r1_feats, r2_feats, r3_feats), dim=0)

		# get features or embeddings for batch nodes
		if self.cuda and isinstance(nodes, list):
			index = torch.LongTensor(nodes).cuda()
		else:
			index = torch.LongTensor(nodes)
		self_feats = self.features(index)

		# number of nodes in a batch
		n = len(nodes)

		# inter-relation aggregation steps
		# Eq. (9) in the paper
		if self.inter == 'Att':
			# 1) CARE-Att Inter-relation Aggregator
			combined, attention = att_inter_agg(len(self.adj_lists), self.leakyrelu, self_feats, neigh_feats, self.embed_dim,
												self.weight, self.a, n, self.dropout, self.training, self.cuda)
		elif self.inter == 'Weight':
			# 2) CARE-Weight Inter-relation Aggregator
			combined = weight_inter_agg(len(self.adj_lists), self_feats, neigh_feats, self.embed_dim, self.weight, self.alpha, n, self.cuda)
			gem_weights = F.softmax(torch.sum(self.alpha, dim=0), dim=0).tolist()
		elif self.inter == 'Mean':
			# 3) CARE-Mean Inter-relation Aggregator
			combined = mean_inter_agg(len(self.adj_lists), self_feats, neigh_feats, self.embed_dim, self.weight, n, self.cuda)

		return combined


class GNNIntraAgg(nn.Module):

	def __init__(self, features, feat_dim, cuda=False):
		"""
		Initialize the intra-relation aggregator
		:param features: the input node features or embeddings for all nodes
		:param feat_dim: the input dimension
		:param cuda: whether to use GPU
		"""
		super(GNNIntraAgg, self).__init__()

		self.features = features
		self.cuda = cuda
		self.feat_dim = feat_dim

	def forward(self, to_neighs_list):
		"""
		Code partially from https://github.com/williamleif/graphsage-simple/
		:param to_neighs_list: neighbor node id list for each batch node in one relation
		:return to_feats: the aggregated embeddings of batch nodes neighbors in one relation
		"""

		# filer neighbors under given relation
		samp_neighs = to_neighs_list

		# find the unique nodes among batch nodes and the filtered neighbors
		unique_nodes_list = list(set.union(*samp_neighs))
		unique_nodes = {n: i for i, n in enumerate(unique_nodes_list)}

		# intra-relation aggregation only with sampled neighbors
		mask = Variable(torch.zeros(len(samp_neighs), len(unique_nodes)))
		column_indices = [unique_nodes[n] for samp_neigh in samp_neighs for n in samp_neigh]
		row_indices = [i for i in range(len(samp_neighs)) for _ in range(len(samp_neighs[i]))]
		mask[row_indices, column_indices] = 1
		if self.cuda:
			mask = mask.cuda()
		num_neigh = mask.sum(1, keepdim=True)
		mask = mask.div(num_neigh)
		if self.cuda:
			embed_matrix = self.features(torch.LongTensor(unique_nodes_list).cuda())
		else:
			embed_matrix = self.features(torch.LongTensor(unique_nodes_list))
		to_feats = mask.mm(embed_matrix)
		to_feats = F.relu(to_feats)
		return to_feats

def mean_inter_agg(num_relations, self_feats, neigh_feats, embed_dim, weight, n, cuda):
	"""
	Mean inter-relation aggregator
	:param num_relations: number of relations in the graph
	:param self_feats: batch nodes features or embeddings
	:param neigh_feats: intra-relation aggregated neighbor embeddings for each relation
	:param embed_dim: the dimension of output embedding
	:param weight: parameter used to transform node embeddings before inter-relation aggregation
	:param n: number of nodes in a batch
	:param cuda: whether use GPU
	:return: inter-relation aggregated node embeddings
	"""

	# transform batch node embedding and neighbor embedding in each relation with weight parameter
	center_h = torch.mm(self_feats, weight)
	neigh_h = torch.mm(neigh_feats, weight)

	# initialize the final neighbor embedding
	if cuda:
		aggregated = torch.zeros(size=(n, embed_dim)).cuda()
	else:
		aggregated = torch.zeros(size=(n, embed_dim))

	# sum neighbor embeddings together
	for r in range(num_relations):
		aggregated += neigh_h[r * n:(r + 1) * n, :]

	# sum aggregated neighbor embedding and batch node embedding
	# take the average of embedding and feed them to activation function
	combined = F.relu((center_h + aggregated) / 4.0)

	return combined


def weight_inter_agg(num_relations, self_feats, neigh_feats, embed_dim, weight, alpha, n, cuda):
	"""
	Weight inter-relation aggregator
	Reference: https://arxiv.org/abs/2002.12307
	:param num_relations: number of relations in the graph
	:param self_feats: batch nodes features or embeddings
	:param neigh_feats: intra-relation aggregated neighbor embeddings for each relation
	:param embed_dim: the dimension of output embedding
	:param weight: parameter used to transform node embeddings before inter-relation aggregation
	:param alpha: weight parameter for each relation used by CARE-Weight
	:param n: number of nodes in a batch
	:param cuda: whether use GPU
	:return: inter-relation aggregated node embeddings
	"""

	# transform batch node embedding and neighbor embedding in each relation with weight parameter
	center_h = torch.mm(self_feats, weight)
	neigh_h = torch.mm(neigh_feats, weight)

	# compute relation weights using softmax
	w = F.softmax(alpha, dim=1)

	# initialize the final neighbor embedding
	if cuda:
		aggregated = torch.zeros(size=(n, embed_dim)).cuda()
	else:
		aggregated = torch.zeros(size=(n, embed_dim))

	# add weighted neighbor embeddings in each relation together
	for r in range(num_relations):
		aggregated += neigh_h[r * n:(r + 1) * n, :] * w[:, r]

	# sum aggregated neighbor embedding and batch node embedding
	# feed them to activation function
	combined = F.relu(center_h + aggregated)

	return combined


def att_inter_agg(num_relations, att_layer, self_feats, neigh_feats, embed_dim, weight, a, n, dropout, training, cuda):
	"""
	Attention-based inter-relation aggregator
	Reference: https://github.com/Diego999/pyGAT
	:param num_relations: num_relations: number of relations in the graph
	:param att_layer: the activation function used by the attention layer
	:param self_feats: batch nodes features or embeddings
	:param neigh_feats: intra-relation aggregated neighbor embeddings for each relation
	:param embed_dim: the dimension of output embedding
	:param weight: parameter used to transform node embeddings before inter-relation aggregation
	:param a: parameters used by attention layer
	:param n: number of nodes in a batch
	:param dropout: dropout for attention layer
	:param training: a flag indicating whether in the training or testing mode
	:param cuda: whether use GPU
	:return combined: inter-relation aggregated node embeddings
	:return att: the attention weights for each relation
	"""

	# transform batch node embedding and neighbor embedding in each relation with weight parameter
	center_h = torch.mm(self_feats, weight)
	neigh_h = torch.mm(neigh_feats, weight)

	import pdb
	pdb.set_trace()
	# compute attention weights
	combined = torch.cat((center_h.repeat(3, 1), neigh_h), dim=1)
	e = att_layer(combined.mm(a))
	attention = torch.cat((e[0:n, :], e[n:2 * n, :], e[2 * n:3 * n, :]), dim=1)
	ori_attention = F.softmax(attention, dim=1)
	attention = F.dropout(ori_attention, dropout, training=training)

	# initialize the final neighbor embedding
	if cuda:
		aggregated = torch.zeros(size=(n, embed_dim)).cuda()
	else:
		aggregated = torch.zeros(size=(n, embed_dim))

	# add neighbor embeddings in each relation together with attention weights
	for r in range(num_relations):
		aggregated += torch.mul(attention[:, r].unsqueeze(1).repeat(1, embed_dim), neigh_h[r * n:(r + 1) * n, :])

	# sum aggregated neighbor embedding and batch node embedding
	# feed them to activation function
	combined = F.relu((center_h + aggregated))

	# extract the attention weights
	att = F.softmax(torch.sum(ori_attention, dim=0), dim=0)

	return combined, att


def threshold_inter_agg(num_relations, self_feats, neigh_feats, embed_dim, weight, threshold, n, cuda):
	"""
	CARE-GNN inter-relation aggregator
	Eq. (9) in the paper
	:param num_relations: number of relations in the graph
	:param self_feats: batch nodes features or embeddings
	:param neigh_feats: intra-relation aggregated neighbor embeddings for each relation
	:param embed_dim: the dimension of output embedding
	:param weight: parameter used to transform node embeddings before inter-relation aggregation
	:param threshold: the neighbor filtering thresholds used as aggregating weights
	:param n: number of nodes in a batch
	:param cuda: whether use GPU
	:return: inter-relation aggregated node embeddings
	"""

	# transform batch node embedding and neighbor embedding in each relation with weight parameter
	center_h = torch.mm(self_feats, weight)
	neigh_h = torch.mm(neigh_feats, weight)

	# initialize the final neighbor embedding
	if cuda:
		aggregated = torch.zeros(size=(n, embed_dim)).cuda()
	else:
		aggregated = torch.zeros(size=(n, embed_dim))

	# add weighted neighbor embeddings in each relation together
	for r in range(num_relations):
		aggregated += neigh_h[r * n:(r + 1) * n, :] * threshold[r]

	# sum aggregated neighbor embedding and batch node embedding
	# feed them to activation function
	combined = F.relu(center_h + aggregated)

	return combined
