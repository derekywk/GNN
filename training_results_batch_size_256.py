################################################
# -1 ; emb-size 64
# run on watch; Dataset size (151486, 84); USING_GIST_AS None; Random seed 2; TOTAL_VOTES_GT_1 True
# batch_size: 256 | cuda: False | data: watch | emb_size: 64 | inter: GNN | lambda_1: 2 | lambda_2: 0.001 |
# lr: 0.01 | model: CARE | no_cuda: False | num_epochs: 16 | seed: 3 | step_size: 0.02 | test_epochs: 3 | under_sample: 1 |
_ = {"""
# 1st
GNN F1	0.6740
GNN Accuracy	0.7937
GNN Recall	0.7358
GNN auc	0.8116
GNN ap	0.4921
Label1 F1	0.5825
Label1 Accuracy	0.6810
Label1 Recall	0.6815
Label1 auc	0.7565
Label1 ap	0.4119
# 2nd
GNN F1	0.5624
GNN Accuracy	0.6282
GNN Recall	0.7160
GNN auc	0.8120
GNN ap	0.4977
Label1 F1	0.6079
Label1 Accuracy	0.7238
Label1 Recall	0.6847
Label1 auc	0.7562
Label1 ap	0.4097
# 3rd
GNN F1	0.6670
GNN Accuracy	0.7811
GNN Recall	0.7407
GNN auc	0.8116
GNN ap	0.4891
Label1 F1	0.5911
Label1 Accuracy	0.6947
Label1 Recall	0.6838
Label1 auc	0.7566
Label1 ap	0.4094
# 4th
GNN F1	0.6304
GNN Accuracy	0.7258
GNN Recall	0.7401
GNN auc	0.8135
GNN ap	0.4928
Label1 F1	0.5868
Label1 Accuracy	0.6877
Label1 Recall	0.6827
Label1 auc	0.7568
Label1 ap	0.4102
# 5th
GNN F1	0.6382
GNN Accuracy	0.7395
GNN Recall	0.7378
GNN auc	0.8095
GNN ap	0.4964
Label1 F1	0.5997
Label1 Accuracy	0.7100
Label1 Recall	0.6838
Label1 auc	0.7563
Label1 ap	0.4091
"""}
################################################
# -1 with 48 gist features ; emb-size 64
# run on watch; Dataset size (151486, 84); USING_GIST_AS feature; Random seed 1; TOTAL_VOTES_GT_1 True
# batch_size: 256 | cuda: False | data: watch | emb_size: 64 | inter: GNN | lambda_1: 2 | lambda_2: 0.001 |
# lr: 0.01 | model: CARE | no_cuda: False | num_epochs: 16 | seed: 3 | step_size: 0.02 | test_epochs: 3 | under_sample: 1 |
_ = {"""
# 1st
GNN F1	0.6729
GNN Accuracy	0.7871
GNN Recall	0.7448
GNN auc	0.8182
GNN ap	0.4969
Label1 F1	0.5750
Label1 Accuracy	0.6623
Label1 Recall	0.6905
Label1 auc	0.7633
Label1 ap	0.4229
# 2nd
GNN F1	0.5685
GNN Accuracy	0.6354
GNN Recall	0.7216
GNN auc	0.8175
GNN ap	0.5001
Label1 F1	0.5972
Label1 Accuracy	0.7004
Label1 Recall	0.6912
Label1 auc	0.7633
Label1 ap	0.4222
# 3rd
GNN F1	0.6731
GNN Accuracy	0.7871
GNN Recall	0.7455
GNN auc	0.8171
GNN ap	0.4899
Label1 F1	0.5814
Label1 Accuracy	0.6735
Label1 Recall	0.6904
Label1 auc	0.7635
Label1 ap	0.4218
# 4th
GNN F1	0.6191
GNN Accuracy	0.7067
GNN Recall	0.7425
GNN auc	0.8204
GNN ap	0.4981
Label1 F1	0.5768
Label1 Accuracy	0.6656
Label1 Recall	0.6903
Label1 auc	0.7636
Label1 ap	0.4221
# 5th
GNN F1	0.6341
GNN Accuracy	0.7302
GNN Recall	0.7425
GNN auc	0.8163
GNN ap	0.4987
Label1 F1	0.5911
Label1 Accuracy	0.6895
Label1 Recall	0.6916
Label1 auc	0.7633
Label1 ap	0.4219
"""}
################################################
# -1 with 48 gist features with max-column normalization ; emb-size 64
# run on watch; Dataset size (151486, 84); USING_GIST_AS feature; Random seed 5; TOTAL_VOTES_GT_1 True; NORMALIZATION max_column
# batch_size: 256 | cuda: True | data: watch | emb_size: 64 | inter: GNN | lambda_1: 2 | lambda_2: 0.001 |
# lr: 0.01 | model: CARE | no_cuda: False | num_epochs: 19 | seed: 5 | step_size: 0.02 | test_epochs: 3 | under_sample: 1 |
_ = {"""
# 1st
GNN F1	0.6591
GNN Accuracy	0.7520
GNN Recall	0.7725
GNN auc	0.8520
GNN ap	0.5377
Label1 F1	0.6700
Label1 Accuracy	0.7676
Label1 Recall	0.7724
Label1 auc	0.8483
Label1 ap	0.5285
# 2nd
GNN F1	0.6137
GNN Accuracy	0.6891
GNN Recall	0.7607
GNN auc	0.8519
GNN ap	0.5388
Label1 F1	0.6550
Label1 Accuracy	0.7468
Label1 Recall	0.7709
Label1 auc	0.8483
Label1 ap	0.5284
# 3rd
GNN F1	0.6939
GNN Accuracy	0.8027
GNN Recall	0.7679
GNN auc	0.8498
GNN ap	0.5380
Label1 F1	0.6335
Label1 Accuracy	0.7166
Label1 Recall	0.7668
Label1 auc	0.8483
Label1 ap	0.5284
# 4th
GNN F1	0.6435
GNN Accuracy	0.7296
GNN Recall	0.7710
GNN auc	0.8520
GNN ap	0.5400
Label1 F1	0.6640
Label1 Accuracy	0.7595
Label1 Recall	0.7715
Label1 auc	0.8481
Label1 ap	0.5302
# 5th
GNN F1	0.6934
GNN Accuracy	0.8009
GNN Recall	0.7704
GNN auc	0.8508
GNN ap	0.5375
Label1 F1	0.6703
Label1 Accuracy	0.7682
Label1 Recall	0.7720
Label1 auc	0.8485
Label1 ap	0.5309
"""}
################################################
# -1 with 48 gist features ; emb-size 128
# run on watch; Dataset size (151486, 84); USING_GIST_AS feature; Random seed 1; TOTAL_VOTES_GT_1 True
# batch_size: 256 | cuda: False | data: watch | emb_size: 128 | inter: GNN | lambda_1: 2 | lambda_2: 0.001
# lr: 0.01 | model: CARE | no_cuda: False | num_epochs: 16 | seed: 1 | step_size: 0.02 | test_epochs: 3 | under_sample: 1 |
_ = {"""
# 1st
GNN F1	0.6714
GNN Accuracy	0.7850
GNN Recall	0.7448
GNN auc	0.8181
GNN ap	0.4975
Label1 F1	0.5751
Label1 Accuracy	0.6624
Label1 Recall	0.6906
Label1 auc	0.7632
Label1 ap	0.4229
# 2nd
GNN F1	0.5707
GNN Accuracy	0.6388
GNN Recall	0.7220
GNN auc	0.8169
GNN ap	0.5007
Label1 F1	0.5974
Label1 Accuracy	0.7006
Label1 Recall	0.6914
Label1 auc	0.7633
Label1 ap	0.4221
# 3rd
GNN F1	0.6759
GNN Accuracy	0.7916
GNN Recall	0.7446
GNN auc	0.8159
GNN ap	0.4884
Label1 F1	0.5815
Label1 Accuracy	0.6737
Label1 Recall	0.6904
Label1 auc	0.7634
Label1 ap	0.4217
# 4th
GNN F1	0.5731
GNN Accuracy	0.6414
GNN Recall	0.7248
GNN auc	0.8187
GNN ap	0.4965
Label1 F1	0.5770
Label1 Accuracy	0.6658
Label1 Recall	0.6903
Label1 auc	0.7635
Label1 ap	0.4219
# 5th
GNN F1	0.6600
GNN Accuracy	0.7687
GNN Recall	0.7444
GNN auc	0.8160
GNN ap	0.4990
Label1 F1	0.5911
Label1 Accuracy	0.6895
Label1 Recall	0.6917
Label1 auc	0.7633
Label1 ap	0.4219
"""}
################################################
# -1 with relations (1 sided; distance threshold > 1.0) ; emb-size 64
# run on watch; Dataset size (151486, 34); USING_GIST_AS relation; Random seed 1; TOTAL_VOTES_GT_1 True
# batch_size: 256 | cuda: False | data: watch | emb_size: 64 | inter: GNN | lambda_1: 2 | lambda_2: 0.001 |
# lr: 0.01 | model: CARE | no_cuda: False | num_epochs: 16 | seed: 1 | step_size: 0.02 | test_epochs: 3 | under_sample: 1 |
if USING_GIST_AS == 'relation':
    for col in [col for col in df.columns if 'f_gist_' in col]:
        tprint(f'Computing Relation for gist "{col[7:]}"...')
        gist_relation = defaultdict(set)
        related = df.loc[target_df_indices, col].reset_index(drop=True) > 1.0
        for index in range(len(related)):
            gist_relation[index] = set((index,))
        index_list = related.index[related]
        s = set(index_list)
        for index in index_list:
            gist_relation[index] = s
_ = {"""
# 1
GNN F1	0.6895
GNN Accuracy	0.8013
GNN Recall	0.7602
GNN auc	0.8376
GNN ap	0.5369
Label1 F1	0.5826
Label1 Accuracy	0.6810
Label1 Recall	0.6819
Label1 auc	0.7565
Label1 ap	0.4117
# 2
GNN F1	0.6822
GNN Accuracy	0.7903
GNN Recall	0.7622
GNN auc	0.8350
GNN ap	0.5345
Label1 F1	0.6082
Label1 Accuracy	0.7238
Label1 Recall	0.6851
Label1 auc	0.7565
Label1 ap	0.4103
# 3
GNN F1	0.7019
GNN Accuracy	0.8171
GNN Recall	0.7606
GNN auc	0.8401
GNN ap	0.5432
Label1 F1	0.5912
Label1 Accuracy	0.6948
Label1 Recall	0.6841
Label1 auc	0.7568
Label1 ap	0.4100
# 4
GNN F1	0.6514
GNN Accuracy	0.7468
GNN Recall	0.7607
GNN auc	0.8367
GNN ap	0.5396
Label1 F1	0.5869
Label1 Accuracy	0.6880
Label1 Recall	0.6828
Label1 auc	0.7566
Label1 ap	0.4098
# 5
GNN F1	0.6963
GNN Accuracy	0.8113
GNN Recall	0.7584
GNN auc	0.8371
GNN ap	0.5322
Label1 F1	0.6075
Label1 Accuracy	0.7229
Label1 Recall	0.6849
Label1 auc	0.7567
Label1 ap	0.4109
"""}
################################################
# -1 with relations (2-sided; distance threshold > 1.0 & < 0.4) ; emb-size 64
# run on watch; Dataset size (151486, 34); USING_GIST_AS relation; Random seed 1; TOTAL_VOTES_GT_1 True
# batch_size: 256 | cuda: False | data: watch | emb_size: 64 | inter: GNN | lambda_1: 2 | lambda_2: 0.001 |
# lr: 0.01 | model: CARE | no_cuda: False | num_epochs: 16 | seed: 1 | step_size: 0.02 | test_epochs: 3 | under_sample: 1 |
if USING_GIST_AS == 'relation':
    for col in [col for col in df.columns if 'f_gist_' in col]:
        tprint(f'Computing Relation for gist "{col[7:]}"...')
        gist_relation = defaultdict(set)
        related = df.loc[target_df_indices, col].reset_index(drop=True) > 1.0
        for index in range(len(related)):
            gist_relation[index] = set((index,))
        index_list = related.index[related]
        s = set(index_list)
        for index in index_list:
            gist_relation[index] = s
        relation_list.append(gist_relation)

        tprint(f'Computing Relation for gist "{col[7:]}"...')
        gist_relation = defaultdict(set)
        related = df.loc[target_df_indices, col].reset_index(drop=True) < 0.4
        for index in range(len(related)):
            gist_relation[index] = set((index,))
        index_list = related.index[related]
        s = set(index_list)
        for index in index_list:
            gist_relation[index] = s
        relation_list.append(gist_relation)
_ = {"""
# 1
GNN F1	0.6825
GNN Accuracy	0.7887
GNN Recall	0.7656
GNN auc	0.8404
GNN ap	0.5458
Label1 F1	0.5825
Label1 Accuracy	0.6809
Label1 Recall	0.6817
Label1 auc	0.7566
Label1 ap	0.4119
# 2
GNN F1	0.6891
GNN Accuracy	0.8003
GNN Recall	0.7611
GNN auc	0.8363
GNN ap	0.5386
Label1 F1	0.6081
Label1 Accuracy	0.7239
Label1 Recall	0.6850
Label1 auc	0.7565
Label1 ap	0.4104
# 3
GNN F1	0.6986
GNN Accuracy	0.8156
GNN Recall	0.7560
GNN auc	0.8366
GNN ap	0.5239
Label1 F1	0.5859
Label1 Accuracy	0.6860
Label1 Recall	0.6828
Label1 auc	0.7573
Label1 ap	0.4118
# 4
GNN F1	0.5888
GNN Accuracy	0.6586
GNN Recall	0.7422
GNN auc	0.8396
GNN ap	0.5472
Label1 F1	0.6076
Label1 Accuracy	0.7232
Label1 Recall	0.6846
Label1 auc	0.7565
Label1 ap	0.4113
# 5
GNN F1	0.7071
GNN Accuracy	0.8306
GNN Recall	0.7487
GNN auc	0.8357
GNN ap	0.5282
Label1 F1	0.6074
Label1 Accuracy	0.7227
Label1 Recall	0.6849
Label1 auc	0.7569
Label1 ap	0.4113
"""}
################################################
# -1 with relations (1-sided; distance threshold < 0.4) ; emb-size 64
# run on watch; Dataset size (151486, 34); USING_GIST_AS relation; Random seed 1; TOTAL_VOTES_GT_1 True; NORMALIZATION row
# batch_size: 256 | cuda: False | data: watch | emb_size: 64 | inter: GNN | lambda_1: 2 | lambda_2: 0.001 |
# lr: 0.01 | model: CARE | no_cuda: False | num_epochs: 20 | seed: 1 | step_size: 0.02 | test_epochs: 3 | under_sample: 1 |
if USING_GIST_AS == 'relation':
    for col in [col for col in df.columns if 'f_gist_' in col]:
        tprint(f'Computing Relation for gist "{col[7:]}"...')
        gist_relation = defaultdict(set)
        rounded_distance = ((df.loc[target_df_indices, col].reset_index(drop=True)/2).round(1))*2
        for rounded_value in rounded_distance.unique():
            index_list = rounded_distance.index[rounded_distance==rounded_value]
            s = set(index_list)
            for index in index_list:
                gist_relation[index] = s
_ = {"""
# 1
GNN F1	0.6569
GNN Accuracy	0.7543
GNN Recall	0.7621
GNN auc	0.8372
GNN ap	0.5398
Label1 F1	0.5976
Label1 Accuracy	0.7058
Label1 Recall	0.6845
Label1 auc	0.7572
Label1 ap	0.4132
# 2
GNN F1	0.6474
GNN Accuracy	0.7409
GNN Recall	0.7605
GNN auc	0.8382
GNN ap	0.5485
Label1 F1	0.5966
Label1 Accuracy	0.7042
Label1 Recall	0.6841
Label1 auc	0.7569
Label1 ap	0.4121
"""}