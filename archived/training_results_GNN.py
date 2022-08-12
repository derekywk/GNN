################################################
# Yelp
# run on yelp; Dataset size (45954, 32); USING_GIST_AS feature and relation; NUMBER_OF_GIST 50; Random seed 1; TOTAL_VOTES_GT_1 True; NORMALIZATION max_column
# batch_size: 256 | cuda: True | data: yelp | emb_size: 64 | inter: GNN | lambda_1: 2 | lambda_2: 0.001 |
# lr: 0.01 | model: GNN | no_cuda: False | num_epochs: 19 | seed: 1 | step_size: 0.02 | test_epochs: 9 | under_sample: 1 |
_ = {"""
# 1
GNN F1	0.7098
GNN Accuracy	0.8484
GNN Recall	0.7248
GNN auc	0.8408
GNN ap	0.5153
# 2
GNN F1	0.6946
GNN Accuracy	0.7992
GNN Recall	0.7762
GNN auc	0.8565
GNN ap	0.5413
# 3
GNN F1	0.6433
GNN Accuracy	0.7313
GNN Recall	0.7669
GNN auc	0.8461
GNN ap	0.5292
# 4
GNN F1	0.6994
GNN Accuracy	0.8100
GNN Recall	0.7688
GNN auc	0.8450
GNN ap	0.5314
# 5
GNN F1	0.7153
GNN Accuracy	0.8227
GNN Recall	0.7841
GNN auc	0.8627
GNN ap	0.5712
"""}
################################################
# Without Gist
_ = {"""
# 1
GNN F1	0.6643
GNN Accuracy	0.7740
GNN Recall	0.7456
GNN auc	0.8165
GNN ap	0.4809
# 2
GNN F1	0.6070
GNN Accuracy	0.6898
GNN Recall	0.7374
GNN auc	0.8147
GNN ap	0.4753
# 3
GNN F1	0.6593
GNN Accuracy	0.7663
GNN Recall	0.7463
GNN auc	0.8152
GNN ap	0.4767
# 4
GNN F1	0.6556
GNN Accuracy	0.7589
GNN Recall	0.7487
GNN auc	0.8178
GNN ap	0.4788
# 5
GNN F1	0.6590
GNN Accuracy	0.7667
GNN Recall	0.7457
GNN auc	0.8143
GNN ap	0.4760
"""}
################################################
################################################
# With Gist Relations (distance threshold > 1.0)
# run on watch; Dataset size (151486, 34); USING_GIST_AS relation; NUMBER_OF_GIST 50; Random seed 1; TOTAL_VOTES_GT_1 True; NORMALIZATION max_column
# batch_size: 256 | cuda: True | data: watch | emb_size: 64 | inter: GNN | lambda_1: 2 | lambda_2: 0.001 |
# lr: 0.01 | model: GNN | no_cuda: False | num_epochs: 19 | seed: 3 | step_size: 0.02 | test_epochs: 3 | under_sample: 1 |
_ = {"""
# 1
# 2
# 3
# 4
# 5
"""}
################################################
# With Gist Features and Relations (distance threshold > 1.0)
#
_ = {"""
# 1
# 2
# 3
# 4
# 5
"""}
################################################