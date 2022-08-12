################################################
# Yelp
#
_ = {"""
# 1
GNN F1	0.7159
GNN Accuracy	0.8290
GNN Recall	0.7710
GNN auc	0.8570
GNN ap	0.5546
# 2
GNN F1	0.6676
GNN Accuracy	0.7681
GNN Recall	0.7646
GNN auc	0.8396
GNN ap	0.5183
# 3
GNN F1	0.6252
GNN Accuracy	0.7043
GNN Recall	0.7650
GNN auc	0.8504
GNN ap	0.5447
# 4
GNN F1	0.6821
GNN Accuracy	0.7904
GNN Recall	0.7604
GNN auc	0.8365
GNN ap	0.5092
# 5
GNN F1	0.6670
GNN Accuracy	0.7682
GNN Recall	0.7625
GNN auc	0.8386
GNN ap	0.5142
"""}
################################################
# Without Gist
# run on watch; Dataset size (151486, 34); USING_GIST_AS None; NUMBER_OF_GIST 1; Random seed 1; TOTAL_VOTES_GT_1 True; NORMALIZATION max_column
# batch_size: 256 | cuda: True | data: watch | emb_size: 64 | inter: GNN | lambda_1: 2 | lambda_2: 0.001 |
# lr: 0.01 | model: GNN | no_cuda: False | num_epochs: 19 | seed: 1 | step_size: 0.02 | test_epochs: 3 | under_sample: 1 |
_ = {"""
# 1
GNN F1	0.6516
GNN Accuracy	0.7538
GNN Recall	0.7481
GNN auc	0.8168
GNN ap	0.4761
# 2
GNN F1	0.6074
GNN Accuracy	0.6904
GNN Recall	0.7375
GNN auc	0.8143
GNN ap	0.4731
# 3
GNN F1	0.6217
GNN Accuracy	0.7108
GNN Recall	0.7421
GNN auc	0.8148
GNN ap	0.4707
# 4
GNN F1	0.6077
GNN Accuracy	0.6912
GNN Recall	0.7369
GNN auc	0.8134
GNN ap	0.4765
# 5
GNN F1	0.6592
GNN Accuracy	0.7677
GNN Recall	0.7437
GNN auc	0.8137
GNN ap	0.4679
"""}
################################################
# With Gist Features
# run on watch; Dataset size (151486, 84); USING_GIST_AS feature; NUMBER_OF_GIST 50; Random seed 1; TOTAL_VOTES_GT_1 True; NORMALIZATION max_column
# batch_size: 256 | cuda: True | data: watch | emb_size: 64 | inter: GNN | lambda_1: 2 | lambda_2: 0.001 |
# lr: 0.01 | model: GNN | no_cuda: False | num_epochs: 19 | seed: 1 | step_size: 0.02 | test_epochs: 9 | under_sample: 1 |
_ = {"""
# 1
GNN F1	0.6632
GNN Accuracy	0.7586
GNN Recall	0.7711
GNN auc	0.8496
GNN ap	0.5339
# 2
GNN F1	0.6169
GNN Accuracy	0.6939
GNN Recall	0.7613
GNN auc	0.8491
GNN ap	0.5313
# 3
GNN F1	0.6714
GNN Accuracy	0.7721
GNN Recall	0.7677
GNN auc	0.8462
GNN ap	0.5275
# 4
GNN F1	0.6394
GNN Accuracy	0.7253
GNN Recall	0.7672
GNN auc	0.8485
GNN ap	0.5331
# 5
GNN F1	0.6100
GNN Accuracy	0.6846
GNN Recall	0.7578
GNN auc	0.8493
GNN ap	0.5321
"""}
################################################
# With Gist Relations (distance threshold > 1.0)
# run on watch; Dataset size (151486, 34); USING_GIST_AS relation; NUMBER_OF_GIST 50; Random seed 1; TOTAL_VOTES_GT_1 True; NORMALIZATION max_column
# batch_size: 256 | cuda: True | data: watch | emb_size: 64 | inter: GNN | lambda_1: 2 | lambda_2: 0.001 |
# lr: 0.01 | model: GNN | no_cuda: False | num_epochs: 19 | seed: 3 | step_size: 0.02 | test_epochs: 3 | under_sample: 1 |
_ = {"""
# 1
GNN F1	0.6902
GNN Accuracy	0.8038
GNN Recall	0.7570
GNN auc	0.8355
GNN ap	0.5089
# 2
GNN F1	0.6534
GNN Accuracy	0.7501
GNN Recall	0.7598
GNN auc	0.8338
GNN ap	0.5074
# 3
GNN F1	0.6686
GNN Accuracy	0.7734
GNN Recall	0.7577
GNN auc	0.8297
GNN ap	0.4944
# 4
GNN F1	0.6669
GNN Accuracy	0.7696
GNN Recall	0.7606
GNN auc	0.8338
GNN ap	0.5036
# 5
GNN F1	0.6956
GNN Accuracy	0.8135
GNN Recall	0.7527
GNN auc	0.8338
GNN ap	0.4956
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