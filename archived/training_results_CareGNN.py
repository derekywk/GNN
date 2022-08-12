################################################
# Yelp
# run on yelp; Dataset size (45954, 32); USING_GIST_AS None; Random seed 1; TOTAL_VOTES_GT_1 True; NORMALIZATION max_column
# batch_size: 256 | cuda: False | data: yelp | emb_size: 64 | inter: GNN | lambda_1: 2 | lambda_2: 0.001 |
# lr: 0.01 | model: CARE | no_cuda: False | num_epochs: 19 | seed: 1 | step_size: 0.02 | test_epochs: 3 | under_sample: 1 |
_ = {"""
# 1
GNN F1	0.6678
GNN Accuracy	0.7857
GNN Recall	0.7339
GNN auc	0.8187
GNN ap	0.4815
Label1 F1	0.5925
Label1 Accuracy	0.6847
Label1 Recall	0.7023
Label1 auc	0.7693
Label1 ap	0.3732
# 2
GNN F1	0.6697
GNN Accuracy	0.7790
GNN Recall	0.7495
GNN auc	0.8340
GNN ap	0.4975
Label1 F1	0.6116
Label1 Accuracy	0.7154
Label1 Recall	0.7051
Label1 auc	0.7698
Label1 ap	0.3754
# 3
GNN F1	0.5951
GNN Accuracy	0.6692
GNN Recall	0.7396
GNN auc	0.8270
GNN ap	0.4857
Label1 F1	0.5983
Label1 Accuracy	0.6936
Label1 Recall	0.7039
Label1 auc	0.7700
Label1 ap	0.3757
# 4
GNN F1	0.6600
GNN Accuracy	0.7610
GNN Recall	0.7568
GNN auc	0.8351
GNN ap	0.5056
Label1 F1	0.6002
Label1 Accuracy	0.6971
Label1 Recall	0.7032
Label1 auc	0.7701
Label1 ap	0.3745
# 5
GNN F1	0.5873
GNN Accuracy	0.6611
GNN Recall	0.7303
GNN auc	0.8176
GNN ap	0.4821
Label1 F1	0.5886
Label1 Accuracy	0.6772
Label1 Recall	0.7037
Label1 auc	0.7699
Label1 ap	0.3765
"""}
################################################
# Without Gist
# run on watch; Dataset size (151486, 34); USING_GIST_AS None; Random seed 1; TOTAL_VOTES_GT_1 True; NORMALIZATION max_column
# batch_size: 256 | cuda: False | data: watch | emb_size: 64 | inter: GNN | lambda_1: 2 | lambda_2: 0.001 |
# lr: 0.01 | model: CARE | no_cuda: False | num_epochs: 19 | seed: 1 | step_size: 0.02 | test_epochs: 3 | under_sample: 1 |
_ = {"""
# 1
GNN F1	0.6612
GNN Accuracy	0.7651
GNN Recall	0.7532
GNN auc	0.8261
GNN ap	0.4948
Label1 F1	0.6399
Label1 Accuracy	0.7479
Label1 Recall	0.7272
Label1 auc	0.7901
Label1 ap	0.4245
# 2
GNN F1	0.6192
GNN Accuracy	0.7064
GNN Recall	0.7428
GNN auc	0.8190
GNN ap	0.4804
Label1 F1	0.6249
Label1 Accuracy	0.7248
Label1 Recall	0.7259
Label1 auc	0.7890
Label1 ap	0.4227
# 3
GNN F1	0.6365
GNN Accuracy	0.7304
GNN Recall	0.7486
GNN auc	0.8236
GNN ap	0.4822
Label1 F1	0.6376
Label1 Accuracy	0.7460
Label1 Recall	0.7245
Label1 auc	0.7889
Label1 ap	0.4227
# 4
GNN F1	0.6130
GNN Accuracy	0.6973
GNN Recall	0.7418
GNN auc	0.8219
GNN ap	0.4888
Label1 F1	0.6401
Label1 Accuracy	0.7486
Label1 Recall	0.7267
Label1 auc	0.7897
Label1 ap	0.4264
# 5
GNN F1	0.6689
GNN Accuracy	0.7779
GNN Recall	0.7506
GNN auc	0.8243
GNN ap	0.4905
Label1 F1	0.6556
Label1 Accuracy	0.7768
Label1 Recall	0.7198
Label1 auc	0.7898
Label1 ap	0.4267
"""}
################################################
# With Gist Features
# run on watch; Dataset size (151486, 84); USING_GIST_AS feature; Random seed 5; TOTAL_VOTES_GT_1 True; NORMALIZATION max_column
# batch_size: 256 | cuda: False | data: watch | emb_size: 64 | inter: GNN | lambda_1: 2 | lambda_2: 0.001 |
# lr: 0.01 | model: CARE | no_cuda: False | num_epochs: 19 | seed: 5 | step_size: 0.02 | test_epochs: 3 | under_sample: 1 |
_ = {"""
# 1
GNN F1	0.6734
GNN Accuracy	0.7711
GNN Recall	0.7754
GNN auc	0.8527
GNN ap	0.5382
Label1 F1	0.6706
Label1 Accuracy	0.7685
Label1 Recall	0.7724
Label1 auc	0.8484
Label1 ap	0.5286
# 2
GNN F1	0.6363
GNN Accuracy	0.7202
GNN Recall	0.7682
GNN auc	0.8516
GNN ap	0.5408
Label1 F1	0.6546
Label1 Accuracy	0.7465
Label1 Recall	0.7705
Label1 auc	0.8479
Label1 ap	0.5274
# 3
GNN F1	0.6803
GNN Accuracy	0.7819
GNN Recall	0.7731
GNN auc	0.8507
GNN ap	0.5382
Label1 F1	0.6339
Label1 Accuracy	0.7170
Label1 Recall	0.7670
Label1 auc	0.8484
Label1 ap	0.5287
# 4
GNN F1	0.6420
GNN Accuracy	0.7283
GNN Recall	0.7689
GNN auc	0.8506
GNN ap	0.5390
Label1 F1	0.6638
Label1 Accuracy	0.7592
Label1 Recall	0.7715
Label1 auc	0.8479
Label1 ap	0.5298
# 5
GNN F1	0.6627
GNN Accuracy	0.7576
GNN Recall	0.7714
GNN auc	0.8505
GNN ap	0.5370
Label1 F1	0.6704
Label1 Accuracy	0.7684
Label1 Recall	0.7719
Label1 auc	0.8479
Label1 ap	0.5292
"""}
################################################
# With Gist Relations (distance threshold > 1.0)
# run on watch; Dataset size (151486, 34); USING_GIST_AS relation; Random seed 1; TOTAL_VOTES_GT_1 True; NORMALIZATION max_column
# batch_size: 256 | cuda: True | data: watch | emb_size: 64 | inter: GNN | lambda_1: 2 | lambda_2: 0.001 |
# lr: 0.01 | model: CARE | no_cuda: False | num_epochs: 19 | seed: 1 | step_size: 0.02 | test_epochs: 3 | under_sample: 1 |
_ = {"""
# 1
GNN F1	0.6712
GNN Accuracy	0.7762
GNN Recall	0.7595
GNN auc	0.8339
GNN ap	0.5073
Label1 F1	0.6395
Label1 Accuracy	0.7476
Label1 Recall	0.7269
Label1 auc	0.7901
Label1 ap	0.4244
# 2
GNN F1	0.6353
GNN Accuracy	0.7286
GNN Recall	0.7486
GNN auc	0.8246
GNN ap	0.4851
Label1 F1	0.6270
Label1 Accuracy	0.7280
Label1 Recall	0.7262
Label1 auc	0.7889
Label1 ap	0.4225
# 3
GNN F1	0.6502
GNN Accuracy	0.7480
GNN Recall	0.7549
GNN auc	0.8286
GNN ap	0.4918
Label1 F1	0.6376
Label1 Accuracy	0.7460
Label1 Recall	0.7245
Label1 auc	0.7889
Label1 ap	0.4229
# 4
GNN F1	0.6318
GNN Accuracy	0.7215
GNN Recall	0.7518
GNN auc	0.8299
GNN ap	0.4985
Label1 F1	0.6401
Label1 Accuracy	0.7485
Label1 Recall	0.7267
Label1 auc	0.7898
Label1 ap	0.4264
# 5
GNN F1	0.6908
GNN Accuracy	0.8071
GNN Recall	0.7526
GNN auc	0.8322
GNN ap	0.5061
Label1 F1	0.6556
Label1 Accuracy	0.7767
Label1 Recall	0.7199
Label1 auc	0.7898
Label1 ap	0.4267
"""}
################################################
# With Gist Features and Relations (distance threshold > 1.0)
# run on watch; Dataset size (151486, 84); USING_GIST_AS feature and relation; Random seed 1; TOTAL_VOTES_GT_1 True; NORMALIZATION max_column
# batch_size: 256 | cuda: True | data: watch | emb_size: 64 | inter: GNN | lambda_1: 2 | lambda_2: 0.001 |
# lr: 0.01 | model: CARE | no_cuda: False | num_epochs: 19 | seed: 1 | step_size: 0.02 | test_epochs: 3 | under_sample: 1
_ = {"""
# 1
GNN F1	0.6760
GNN Accuracy	0.7737
GNN Recall	0.7772
GNN auc	0.8540
GNN ap	0.5432
Label1 F1	0.6700
Label1 Accuracy	0.7677
Label1 Recall	0.7722
Label1 auc	0.8482
Label1 ap	0.5285
# 2
GNN F1	0.6111
GNN Accuracy	0.6848
GNN Recall	0.7612
GNN auc	0.8533
GNN ap	0.5422
Label1 F1	0.6547
Label1 Accuracy	0.7465
Label1 Recall	0.7705
Label1 auc	0.8480
Label1 ap	0.5274
# 3
GNN F1	0.6524
GNN Accuracy	0.7415
GNN Recall	0.7738
GNN auc	0.8522
GNN ap	0.5368
Label1 F1	0.6342
Label1 Accuracy	0.7175
Label1 Recall	0.7669
Label1 auc	0.8484
Label1 ap	0.5286
# 4
GNN F1	0.6457
GNN Accuracy	0.7320
GNN Recall	0.7732
GNN auc	0.8532
GNN ap	0.5409
Label1 F1	0.6641
Label1 Accuracy	0.7594
Label1 Recall	0.7719
Label1 auc	0.8485
Label1 ap	0.5315
# 5
GNN F1	0.7037
GNN Accuracy	0.8152
GNN Recall	0.7677
GNN auc	0.8529
GNN ap	0.5430
Label1 F1	0.6703
Label1 Accuracy	0.7683
Label1 Recall	0.7721
Label1 auc	0.8480
Label1 ap	0.5296
"""}
################################################
# With Gist Relations (distance threshold < 0.4)
# run on watch; Dataset size (151486, 34); USING_GIST_AS relation; Random seed 1; TOTAL_VOTES_GT_1 True; NORMALIZATION max_column
# batch_size: 256 | cuda: True | data: watch | emb_size: 64 | inter: GNN | lambda_1: 2 | lambda_2: 0.001 |
# lr: 0.01 | model: CARE | no_cuda: False | num_epochs: 19 | seed: 1 | step_size: 0.02 | test_epochs: 3 | under_sample: 1 |
_ = {"""
# 1
GNN F1	0.6715
GNN Accuracy	0.7767
GNN Recall	0.7594
GNN auc	0.8359
GNN ap	0.5106
Label1 F1	0.6398
Label1 Accuracy	0.7479
Label1 Recall	0.7269
Label1 auc	0.7901
Label1 ap	0.4244
# 2
# 3
# 4
# 5
"""}
################################################
# With Gist Relations (distance threshold > 1.2 ; < 1.2 & > 1.0)
# run on watch; Dataset size (151486, 34); USING_GIST_AS relation; Random seed 1; TOTAL_VOTES_GT_1 True; NORMALIZATION max_column
# batch_size: 256 | cuda: True | data: watch | emb_size: 64 | inter: GNN | lambda_1: 2 | lambda_2: 0.001 |
# lr: 0.01 | model: CARE | no_cuda: False | num_epochs: 19 | seed: 1 | step_size: 0.02 | test_epochs: 3 | under_sample: 1 |
_ = {"""
# 1
GNN F1	0.6792
GNN Accuracy	0.7909
GNN Recall	0.7534
GNN auc	0.8313
GNN ap	0.5019
Label1 F1	0.6397
Label1 Accuracy	0.7477
Label1 Recall	0.7270
Label1 auc	0.7901
Label1 ap	0.4243
# 2
# 3
# 4
# 5
"""}
################################################