import torch
import torch.nn as nn
from torch.nn import init


"""
	CARE-GNN Models
	Paper: Enhancing Graph Neural Network-based Fraud Detectors against Camouflaged Fraudsters
	Source: https://github.com/YingtongDou/CARE-GNN
"""


class GNN(nn.Module):
	"""
	The CARE-GNN model in one layer
	"""

	def __init__(self, num_classes, inter1):
		"""
		Initialize the CARE-GNN model
		:param num_classes: number of classes (2 in our paper)
		:param inter1: the inter-relation aggregator that output the final embedding
		"""
		super(GNN, self).__init__()
		self.inter1 = inter1
		self.xent = nn.CrossEntropyLoss()

		# the parameter to transform the final embedding
		self.weight = nn.Parameter(torch.FloatTensor(inter1.embed_dim, num_classes))
		init.xavier_uniform_(self.weight)

	def forward(self, nodes):
		embeds1 = self.inter1(nodes)
		scores = torch.mm(embeds1, self.weight)
		return scores

	def to_prob(self, nodes):
		gnn_scores = self.forward(nodes)
		gnn_prob = nn.functional.softmax(gnn_scores, dim=1)
		return gnn_prob

	def loss(self, nodes, labels):
		gnn_scores = self.forward(nodes)
		gnn_loss = self.xent(gnn_scores, labels.squeeze())
		return gnn_loss
