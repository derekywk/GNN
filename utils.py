import pickle
import random as rd
import numpy as np
import scipy.sparse as sp
from scipy.io import loadmat
import copy as cp
from sklearn.metrics import f1_score, accuracy_score, recall_score, roc_auc_score, average_precision_score
from collections import defaultdict
from datetime import datetime


"""
	Utility functions to handle data and evaluate model.
"""


def load_data(data):
	"""
	Load graph, feature, and label given dataset name
	:returns: home and single-relation graphs, feature, label
	"""

	prefix = 'data/'
	if data == 'yelp':
		data_file = loadmat(prefix + 'YelpChi.mat')
		labels = data_file['label'].flatten()
		feat_data = data_file['features'].todense().A
		# load the preprocessed adj_lists
		with open(prefix + 'yelp_homo_adjlists.pickle', 'rb') as file:
			homo = pickle.load(file)
		file.close()
		with open(prefix + 'yelp_rur_adjlists.pickle', 'rb') as file:
			relation1 = pickle.load(file)
		file.close()
		with open(prefix + 'yelp_rtr_adjlists.pickle', 'rb') as file:
			relation2 = pickle.load(file)
		file.close()
		with open(prefix + 'yelp_rsr_adjlists.pickle', 'rb') as file:
			relation3 = pickle.load(file)
		file.close()
	elif data == 'amazon':
		data_file = loadmat(prefix + 'Amazon.mat')
		labels = data_file['label'].flatten()
		feat_data = data_file['features'].todense().A
		# load the preprocessed adj_lists
		with open(prefix + 'amz_homo_adjlists.pickle', 'rb') as file:
			homo = pickle.load(file)
		file.close()
		with open(prefix + 'amz_upu_adjlists.pickle', 'rb') as file:
			relation1 = pickle.load(file)
		file.close()
		with open(prefix + 'amz_usu_adjlists.pickle', 'rb') as file:
			relation2 = pickle.load(file)
		file.close()
		with open(prefix + 'amz_uvu_adjlists.pickle', 'rb') as file:
			relation3 = pickle.load(file)

	return [homo, relation1, relation2, relation3], feat_data, labels


def normalize(mx):
	"""
		Row-normalize sparse matrix
		Code from https://github.com/williamleif/graphsage-simple/
	"""
	rowsum = np.array(mx.sum(1)) + 0.01
	r_inv = np.power(rowsum, -1).flatten()
	r_inv[np.isinf(r_inv)] = 0.
	r_mat_inv = sp.diags(r_inv)
	mx = r_mat_inv.dot(mx)
	return mx


def sparse_to_adjlist(sp_matrix, filename):
	"""
	Transfer sparse matrix to adjacency list
	:param sp_matrix: the sparse matrix
	:param filename: the filename of adjlist
	"""
	# add self loop
	homo_adj = sp_matrix + sp.eye(sp_matrix.shape[0])
	# create adj_list
	adj_lists = defaultdict(set)
	edges = homo_adj.nonzero()
	for index, node in enumerate(edges[0]):
		adj_lists[node].add(edges[1][index])
		adj_lists[edges[1][index]].add(node)
	with open(filename, 'wb') as file:
		pickle.dump(adj_lists, file)
	file.close()


def pos_neg_split(nodes, labels):
	"""
	Find positive and negative nodes given a list of nodes and their labels
	:param nodes: a list of nodes
	:param labels: a list of node labels
	:returns: the spited positive and negative nodes
	"""
	pos_nodes = []
	neg_nodes = cp.deepcopy(nodes)
	aux_nodes = cp.deepcopy(nodes)
	for idx, label in enumerate(labels):
		if label == 1:
			pos_nodes.append(aux_nodes[idx])
			neg_nodes.remove(aux_nodes[idx])

	return pos_nodes, neg_nodes


def undersample(pos_nodes, neg_nodes, scale=1):
	"""
	Under-sample the negative nodes
	:param pos_nodes: a list of positive nodes
	:param neg_nodes: a list negative nodes
	:param scale: the under-sampling scale
	:return: a list of under-sampled batch nodes
	"""

	aux_nodes = cp.deepcopy(neg_nodes)
	aux_nodes = rd.sample(aux_nodes, k=int(len(pos_nodes)*scale))
	batch_nodes = pos_nodes + aux_nodes

	return batch_nodes


def test_sage(test_cases, labels, model, batch_size):
	"""
	Test the performance of GraphSAGE
	:param test_cases: a list of testing node
	:param labels: a list of testing node labels
	:param model: the GNN model
	:param batch_size: number nodes in a batch
	"""

	test_batch_num = int(len(test_cases) / batch_size) + 1
	f1_gnn = 0.0
	acc_gnn = 0.0
	recall_gnn = 0.0
	gnn_list = []
	for iteration in range(test_batch_num):
		i_start = iteration * batch_size
		i_end = min((iteration + 1) * batch_size, len(test_cases))
		batch_nodes = test_cases[i_start:i_end]
		batch_label = labels[i_start:i_end]
		gnn_prob = model.to_prob(batch_nodes)
		f1_gnn += f1_score(batch_label, gnn_prob.data.cpu().numpy().argmax(axis=1), average="macro")
		acc_gnn += accuracy_score(batch_label, gnn_prob.data.cpu().numpy().argmax(axis=1))
		recall_gnn += recall_score(batch_label, gnn_prob.data.cpu().numpy().argmax(axis=1), average="macro")
		gnn_list.extend(gnn_prob.data.cpu().numpy()[:, 1].tolist())

	auc_gnn = roc_auc_score(labels, np.array(gnn_list))
	ap_gnn = average_precision_score(labels, np.array(gnn_list))
	print(f"GNN F1: {f1_gnn / test_batch_num:.4f}")
	print(f"GNN Accuracy: {acc_gnn / test_batch_num:.4f}")
	print(f"GNN Recall: {recall_gnn / test_batch_num:.4f}")
	print(f"GNN auc: {auc_gnn:.4f}")
	print(f"GNN ap: {ap_gnn:.4f}")


def test_care(test_cases, labels, model, batch_size):
	"""
	Test the performance of CARE-GNN and its variants
	:param test_cases: a list of testing node
	:param labels: a list of testing node labels
	:param model: the GNN model
	:param batch_size: number nodes in a batch
	:returns: the AUC and Recall of GNN and Simi modules
	"""

	test_batch_num = int(len(test_cases) / batch_size) + 1
	f1_gnn = 0.0
	acc_gnn = 0.0
	recall_gnn = 0.0
	f1_label1 = 0.0
	acc_label1 = 0.00
	recall_label1 = 0.0
	gnn_list = []
	label_list1 = []

	for iteration in range(test_batch_num):
		i_start = iteration * batch_size
		i_end = min((iteration + 1) * batch_size, len(test_cases))
		batch_nodes = test_cases[i_start:i_end]
		batch_label = labels[i_start:i_end]
		gnn_prob, label_prob1 = model.to_prob(batch_nodes, batch_label, train_flag=False)

		f1_gnn += f1_score(batch_label, gnn_prob.data.cpu().numpy().argmax(axis=1), average="macro")
		acc_gnn += accuracy_score(batch_label, gnn_prob.data.cpu().numpy().argmax(axis=1))
		recall_gnn += recall_score(batch_label, gnn_prob.data.cpu().numpy().argmax(axis=1), average="macro")

		f1_label1 += f1_score(batch_label, label_prob1.data.cpu().numpy().argmax(axis=1), average="macro")
		acc_label1 += accuracy_score(batch_label, label_prob1.data.cpu().numpy().argmax(axis=1))
		recall_label1 += recall_score(batch_label, label_prob1.data.cpu().numpy().argmax(axis=1), average="macro")

		gnn_list.extend(gnn_prob.data.cpu().numpy()[:, 1].tolist())
		label_list1.extend(label_prob1.data.cpu().numpy()[:, 1].tolist())

	auc_gnn = roc_auc_score(labels, np.array(gnn_list))
	ap_gnn = average_precision_score(labels, np.array(gnn_list))
	auc_label1 = roc_auc_score(labels, np.array(label_list1))
	ap_label1 = average_precision_score(labels, np.array(label_list1))
	print(f"GNN F1\t{f1_gnn / test_batch_num:.4f}")
	print(f"GNN Accuracy\t{acc_gnn / test_batch_num:.4f}")
	print(f"GNN Recall\t{recall_gnn / test_batch_num:.4f}")
	print(f"GNN auc\t{auc_gnn:.4f}")
	print(f"GNN ap\t{ap_gnn:.4f}")
	print(f"Label1 F1\t{f1_label1 / test_batch_num:.4f}")
	print(f"Label1 Accuracy\t{acc_label1 / test_batch_num:.4f}")
	print(f"Label1 Recall\t{recall_label1 / test_batch_num:.4f}")
	print(f"Label1 auc\t{auc_label1:.4f}")
	print(f"Label1 ap\t{ap_label1:.4f}")

	return auc_gnn, auc_label1, recall_gnn / test_batch_num, recall_label1 / test_batch_num

def test_gnn(test_cases, labels, model, batch_size):
	"""
	Test the performance of CARE-GNN and its variants
	:param test_cases: a list of testing node
	:param labels: a list of testing node labels
	:param model: the GNN model
	:param batch_size: number nodes in a batch
	:returns: the AUC and Recall of GNN
	"""

	test_batch_num = int(len(test_cases) / batch_size) + 1
	f1_gnn = 0.0
	acc_gnn = 0.0
	recall_gnn = 0.0
	gnn_list = []

	for iteration in range(test_batch_num):
		i_start = iteration * batch_size
		i_end = min((iteration + 1) * batch_size, len(test_cases))
		batch_nodes = test_cases[i_start:i_end]
		batch_label = labels[i_start:i_end]
		gnn_prob = model.to_prob(batch_nodes)

		f1_gnn += f1_score(batch_label, gnn_prob.data.cpu().numpy().argmax(axis=1), average="macro")
		acc_gnn += accuracy_score(batch_label, gnn_prob.data.cpu().numpy().argmax(axis=1))
		recall_gnn += recall_score(batch_label, gnn_prob.data.cpu().numpy().argmax(axis=1), average="macro")

		gnn_list.extend(gnn_prob.data.cpu().numpy()[:, 1].tolist())

	auc_gnn = roc_auc_score(labels, np.array(gnn_list))
	ap_gnn = average_precision_score(labels, np.array(gnn_list))
	print(f"GNN F1\t{f1_gnn / test_batch_num:.4f}")
	print(f"GNN Accuracy\t{acc_gnn / test_batch_num:.4f}")
	print(f"GNN Recall\t{recall_gnn / test_batch_num:.4f}")
	print(f"GNN auc\t{auc_gnn:.4f}")
	print(f"GNN ap\t{ap_gnn:.4f}")

	return auc_gnn, recall_gnn / test_batch_num

def test_rf(test_cases, labels, model):
	"""
	Test the performance of CARE-GNN and its variants
	:param test_cases: a list of testing node
	:param labels: a list of testing node labels
	:param model: the RF model
	:returns: the AUC and Recall of RF
	"""

	f1_rf = 0.0
	acc_rf = 0.0
	recall_rf = 0.0
	rf_list = []

	rf_prob = [[round(x, 2) for x in proba] for proba in model.predict_proba(test_cases)]

	f1_rf += f1_score(labels, np.argmax(rf_prob, axis=1), average="macro")
	acc_rf += accuracy_score(labels, np.argmax(rf_prob, axis=1))
	recall_rf += recall_score(labels, np.argmax(rf_prob, axis=1), average="macro")

	rf_list.extend(np.array(rf_prob)[:, 1].tolist())

	auc_rf = roc_auc_score(labels, np.array(rf_list))
	ap_rf = average_precision_score(labels, np.array(rf_list))
	print(f"RF F1\t{f1_rf:.4f}")
	print(f"RF Accuracy\t{acc_rf:.4f}")
	print(f"RF Recall\t{recall_rf:.4f}")
	print(f"RF auc\t{auc_rf:.4f}")
	print(f"RF ap\t{ap_rf:.4f}")

	return auc_rf, recall_rf

VERBOSE = {
	'TIME_TAKEN': True,
	'TRAIN': False
}

def tprint(*args, prev_t=None):
	now = datetime.now()
	if prev_t:
		if VERBOSE['TIME_TAKEN']:
			print(now, f'Time taken: {(datetime.now() - prev_t).microseconds} ms;', *args)
	else:
		print(now, *args)
	return now

def is_ascii(string):
	try:
		string.encode('ascii')
	except UnicodeEncodeError:
		return False
	return True

def NoG(feature, relation):
	return {
		'Feature': feature,
		'Relation': relation
	}