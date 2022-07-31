################################################
# -1 ; without normalization
# run on watch; Dataset size (151486, 34); USING_GIST_AS None; Random seed 1; TOTAL_VOTES_GT_1 True; NORMALIZATION None
# batch_size: 256 | cuda: False | data: watch | emb_size: 64 | inter: GNN | lambda_1: 2 | lambda_2: 0.001 |
# lr: 0.01 | model: CARE | no_cuda: False | num_epochs: 20 | seed: 1 | step_size: 0.02 | test_epochs: 3 | under_sample: 1 |
_ = {"""
# 1
GNN F1	0.6823
GNN Accuracy	0.7951
GNN Recall	0.7543
GNN auc	0.8288
GNN ap	0.5112
Label1 F1	0.6008
Label1 Accuracy	0.6836
Label1 Recall	0.7298
Label1 auc	0.8059
Label1 ap	0.4477
# 2
GNN F1	0.2377
GNN Accuracy	0.2400
GNN Recall	0.5455
GNN auc	0.7148
GNN ap	0.2607
Label1 F1	0.6358
Label1 Accuracy	0.7328
Label1 Recall	0.7422
Label1 auc	0.8130
Label1 ap	0.4249
# 3
GNN F1	0.5449
GNN Accuracy	0.6030
GNN Recall	0.7089
GNN auc	0.8002
GNN ap	0.4387
Label1 F1	0.5868
Label1 Accuracy	0.6693
Label1 Recall	0.7134
Label1 auc	0.7885
Label1 ap	0.4259
# 4
GNN F1	0.6502
GNN Accuracy	0.7469
GNN Recall	0.7571
GNN auc	0.8308
GNN ap	0.4840
Label1 F1	0.6837
Label1 Accuracy	0.8041
Label1 Recall	0.7410
Label1 auc	0.8131
Label1 ap	0.4136
# 5
GNN F1	0.6894
GNN Accuracy	0.8038
GNN Recall	0.7551
GNN auc	0.8348
GNN ap	0.5223
Label1 F1	0.6557
Label1 Accuracy	0.7566
Label1 Recall	0.7542
Label1 auc	0.8198
Label1 ap	0.4490
"""}
################################################
# -1 ; max_column
_ = {"""
# 1
GNN F1	0.6533
GNN Accuracy	0.7537
GNN Recall	0.7531
GNN auc	0.8261
GNN ap	0.4939
Label1 F1	0.6396
Label1 Accuracy	0.7477
Label1 Recall	0.7267
Label1 auc	0.7901
Label1 ap	0.4243
# 2
GNN F1	0.6157
GNN Accuracy	0.7005
GNN Recall	0.7438
GNN auc	0.8240
GNN ap	0.4897
Label1 F1	0.6246
Label1 Accuracy	0.7245
Label1 Recall	0.7258
Label1 auc	0.7890
Label1 ap	0.4229
# 3
GNN F1	0.6621
GNN Accuracy	0.7685
GNN Recall	0.7496
GNN auc	0.8216
GNN ap	0.4835
Label1 F1	0.6375
Label1 Accuracy	0.7458
Label1 Recall	0.7244
Label1 auc	0.7889
Label1 ap	0.4227
# 4
GNN F1	0.6132
GNN Accuracy	0.6979
GNN Recall	0.7411
GNN auc	0.8193
GNN ap	0.4847
Label1 F1	0.6400
Label1 Accuracy	0.7485
Label1 Recall	0.7267
Label1 auc	0.7897
Label1 ap	0.4263
# 5
GNN F1	0.6746
GNN Accuracy	0.7865
GNN Recall	0.7497
GNN auc	0.8228
GNN ap	0.4878
Label1 F1	0.6555
Label1 Accuracy	0.7766
Label1 Recall	0.7199
Label1 auc	0.7897
Label1 ap	0.4269
"""}
################################################
# -1 ; sum_column
# run on watch; Dataset size (151486, 34); USING_GIST_AS None; Random seed 1; TOTAL_VOTES_GT_1 True; NORMALIZATION sum_column
# batch_size: 256 | cuda: False | data: watch | emb_size: 64 | inter: GNN | lambda_1: 2 | lambda_2: 0.001 |
# lr: 0.01 | model: CARE | no_cuda: False | num_epochs: 20 | seed: 1 | step_size: 0.02 | test_epochs: 3 | under_sample: 1 |
_ = {"""
# 1
GNN F1	0.4609
GNN Accuracy	0.8551
GNN Recall	0.5000
GNN auc	0.5000
GNN ap	0.1451
Label1 F1	0.4609
Label1 Accuracy	0.8551
Label1 Recall	0.5000
Label1 auc	0.6956
Label1 ap	0.2512
# 2
GNN F1	0.4609
GNN Accuracy	0.8551
GNN Recall	0.5000
GNN auc	0.5000
GNN ap	0.1451
Label1 F1	0.4609
Label1 Accuracy	0.8551
Label1 Recall	0.5000
Label1 auc	0.6884
Label1 ap	0.2449
# 3
GNN F1	0.4609
GNN Accuracy	0.8551
GNN Recall	0.5000
GNN auc	0.5000
GNN ap	0.1451
Label1 F1	0.1263
Label1 Accuracy	0.1449
Label1 Recall	0.5000
Label1 auc	0.6700
Label1 ap	0.2369
# 4
GNN F1	0.4609
GNN Accuracy	0.8551
GNN Recall	0.5000
GNN auc	0.5000
GNN ap	0.1451
Label1 F1	0.4609
Label1 Accuracy	0.8551
Label1 Recall	0.5000
Label1 auc	0.6552
Label1 ap	0.2795
# 5
GNN F1	0.4609
GNN Accuracy	0.8551
GNN Recall	0.5000
GNN auc	0.5000
GNN ap	0.1451
Label1 F1	0.4609
Label1 Accuracy	0.8551
Label1 Recall	0.5000
Label1 auc	0.6677
Label1 ap	0.2391
"""}
################################################
# -1 with relations ; emb-size 64
################################################
# -1 with relations (2-sided) ; emb-size 64
################################################
################################################