# Detector granularity — single-layer vs per-layer vs subset vs joint SVDD

**Model:** M2 · **Tap set:** default · **Seed:** 0

> **One seed: not reportable as a thesis number (C4).** Re-run with ≥ 5 seeds and aggregate.

| Granularity | SPEC | What it is |
|---|---|---|
| single | §6.2 D1 | one SVDD on one tap, judged on its own |
| per_layer | §6.2 D3 | one SVDD per tap, each score through its clean_cal CDF, then combined |
| subset | §9.6(1)(2) | one fused SVDD over a group of taps (position, leave-one-out, greedy) |
| joint | §6.2 D2 | one fused SVDD over all taps |

Every design is fitted on `clean_fit`, ν/γ chosen on `clean_cal`, thresholds at fixed α from
`clean_cal` (C2, C3). No design is chosen with fault data; exhaustive subsets, when run, are kept
out of these tables. Latency is normalise + SVDD + combine per image, batch 1, one process.

## INT8

### Track S (119,755 fault_test probes)

| granularity | design | k | TPR@0.001 | TPR@0.01 | TPR@0.05 | auroc | aupr | support_vectors | latency_ms_b1 |
|---|---|---|---|---|---|---|---|---|---|
| joint | joint:all | 6 | 0.0012 | 0.0101 | 0.0497 | 0.4901 | 0.9828 | 13 | 0.0936 |
| per_layer | per_layer:max | 6 | 0.0000 | 0.0116 | 0.0549 | 0.5069 | 0.9838 | 2454 | 0.6583 |
| per_layer | per_layer:fpr_weighted | 6 | 0.0011 | 0.0079 | 0.0406 | 0.5024 | 0.9833 | 2454 | 0.6285 |
| per_layer | per_layer:mean | 6 | 0.0011 | 0.0079 | 0.0406 | 0.5024 | 0.9833 | 2454 | 0.6409 |
| single | single:logits | 1 | 0.0022 | 0.0115 | 0.0417 | 0.5069 | 0.9838 | 609 | 0.1161 |
| single | single:stage3 | 1 | 0.0011 | 0.0123 | 0.0492 | 0.5026 | 0.9833 | 602 | 0.1175 |
| single | single:stage1 | 1 | 0.0012 | 0.0133 | 0.0533 | 0.5009 | 0.9836 | 301 | 0.1002 |
| single | single:stem | 1 | 0.0038 | 0.0142 | 0.0512 | 0.4992 | 0.9840 | 38 | 0.0832 |
| single | single:stage2 | 1 | 0.0010 | 0.0111 | 0.0527 | 0.4979 | 0.9834 | 303 | 0.0990 |
| single | single:pooled | 1 | 0.0015 | 0.0099 | 0.0477 | 0.4896 | 0.9831 | 601 | 0.1185 |
| subset | subset:early | 2 | 0.0017 | 0.0147 | 0.0479 | 0.5047 | 0.9840 | 613 | 0.1311 |
| subset | subset:without_pooled | 5 | 0.0015 | 0.0116 | 0.0505 | 0.5025 | 0.9835 | 308 | 0.1225 |
| subset | subset:backbone | 4 | 0.0017 | 0.0143 | 0.0453 | 0.5024 | 0.9837 | 619 | 0.2786 |
| subset | subset:late | 2 | 0.0019 | 0.0100 | 0.0495 | 0.4964 | 0.9832 | 301 | 0.1105 |
| subset | subset:middle | 2 | 0.0017 | 0.0116 | 0.0548 | 0.4963 | 0.9831 | 10 | 0.0903 |
| subset | subset:without_stage3 | 5 | 0.0015 | 0.0105 | 0.0463 | 0.4962 | 0.9833 | 50 | 0.0981 |
| subset | subset:without_stage1 | 5 | 0.0014 | 0.0121 | 0.0493 | 0.4958 | 0.9830 | 15 | 0.0924 |
| subset | subset:without_stage2 | 5 | 0.0016 | 0.0108 | 0.0449 | 0.4925 | 0.9830 | 34 | 0.0970 |
| subset | subset:without_logits | 5 | 0.0015 | 0.0099 | 0.0496 | 0.4924 | 0.9830 | 13 | 0.0926 |
| subset | subset:greedy_k3 | 3 | 0.0013 | 0.0116 | 0.0458 | 0.4899 | 0.9830 | 41 | 0.0939 |
| subset | subset:greedy_k4 | 4 | 0.0020 | 0.0113 | 0.0486 | 0.4894 | 0.9829 | 13 | 0.0970 |
| subset | subset:without_stem | 5 | 0.0012 | 0.0100 | 0.0484 | 0.4891 | 0.9827 | 14 | 0.0935 |
| subset | subset:greedy_k2 | 2 | 0.0028 | 0.0124 | 0.0490 | 0.4877 | 0.9831 | 33 | 0.0944 |

### Track H (1,205 fault_test probes)

| granularity | design | k | TPR@0.001 | TPR@0.01 | TPR@0.05 | auroc | aupr | support_vectors | latency_ms_b1 |
|---|---|---|---|---|---|---|---|---|---|
| joint | joint:all | 6 | 0.0050 | 0.0340 | 0.2100 | 0.6830 | 0.5611 | 13 | 0.0936 |
| per_layer | per_layer:mean | 6 | 0.0050 | 0.0183 | 0.1320 | 0.6743 | 0.5295 | 2454 | 0.6409 |
| per_layer | per_layer:fpr_weighted | 6 | 0.0050 | 0.0183 | 0.1320 | 0.6743 | 0.5295 | 2454 | 0.6285 |
| per_layer | per_layer:max | 6 | 0.0000 | 0.0332 | 0.1029 | 0.6515 | 0.4937 | 2454 | 0.6583 |
| single | single:logits | 1 | 0.0124 | 0.0448 | 0.1577 | 0.7410 | 0.5920 | 609 | 0.1161 |
| single | single:stage3 | 1 | 0.0008 | 0.0083 | 0.0772 | 0.6541 | 0.4787 | 602 | 0.1175 |
| single | single:pooled | 1 | 0.0000 | 0.0033 | 0.0573 | 0.5874 | 0.4305 | 601 | 0.1185 |
| single | single:stage1 | 1 | 0.0066 | 0.0290 | 0.0689 | 0.5572 | 0.4288 | 301 | 0.1002 |
| single | single:stem | 1 | 0.0116 | 0.0290 | 0.0788 | 0.5512 | 0.4458 | 38 | 0.0832 |
| single | single:stage2 | 1 | 0.0050 | 0.0257 | 0.0772 | 0.5220 | 0.4106 | 303 | 0.0990 |
| subset | subset:without_stage2 | 5 | 0.0050 | 0.0282 | 0.2075 | 0.7483 | 0.5989 | 34 | 0.0970 |
| subset | subset:without_stage1 | 5 | 0.0075 | 0.0606 | 0.2166 | 0.7119 | 0.5832 | 15 | 0.0924 |
| subset | subset:late | 2 | 0.0000 | 0.0124 | 0.1436 | 0.7079 | 0.5365 | 301 | 0.1105 |
| subset | subset:without_stem | 5 | 0.0041 | 0.0324 | 0.2033 | 0.6879 | 0.5574 | 14 | 0.0935 |
| subset | subset:without_pooled | 5 | 0.0050 | 0.0332 | 0.0996 | 0.6728 | 0.5227 | 308 | 0.1225 |
| subset | subset:without_stage3 | 5 | 0.0058 | 0.0274 | 0.1469 | 0.6684 | 0.5302 | 50 | 0.0981 |
| subset | subset:middle | 2 | 0.0108 | 0.0606 | 0.1909 | 0.6666 | 0.5421 | 10 | 0.0903 |
| subset | subset:without_logits | 5 | 0.0050 | 0.0290 | 0.1834 | 0.6554 | 0.5307 | 13 | 0.0926 |
| subset | subset:greedy_k2 | 2 | 0.0041 | 0.0349 | 0.1162 | 0.6372 | 0.5038 | 33 | 0.0944 |
| subset | subset:greedy_k3 | 3 | 0.0050 | 0.0332 | 0.1195 | 0.6314 | 0.4973 | 41 | 0.0939 |
| subset | subset:backbone | 4 | 0.0050 | 0.0415 | 0.0838 | 0.6103 | 0.4724 | 619 | 0.2786 |
| subset | subset:greedy_k4 | 4 | 0.0075 | 0.0266 | 0.1369 | 0.5952 | 0.4791 | 13 | 0.0970 |
| subset | subset:early | 2 | 0.0050 | 0.0365 | 0.0855 | 0.5773 | 0.4548 | 613 | 0.1311 |

### Paired differences against the joint SVDD (design − joint)

Bootstrap over injection instances (probes of one injection share a corrupted model); the clean side is held fixed. A CI that includes 0 is no detected difference.

| design | track | metric | difference | ci_low | ci_high | verdict |
|---|---|---|---|---|---|---|
| single:stem | S | auroc | 0.0092 | 0.0074 | 0.0109 | difference |
| single:stem | S | tpr@0.01 | 0.0041 | 0.0031 | 0.0052 | difference |
| single:stem | H | auroc | -0.1318 | -0.1523 | -0.1117 | difference |
| single:stem | H | tpr@0.01 | -0.0050 | -0.0201 | 0.0122 | no detected difference |
| single:stage1 | S | auroc | 0.0108 | 0.0088 | 0.0128 | difference |
| single:stage1 | S | tpr@0.01 | 0.0032 | 0.0023 | 0.0041 | difference |
| single:stage1 | H | auroc | -0.1258 | -0.1477 | -0.1044 | difference |
| single:stage1 | H | tpr@0.01 | -0.0050 | -0.0181 | 0.0097 | no detected difference |
| single:stage2 | S | auroc | 0.0078 | 0.0061 | 0.0096 | difference |
| single:stage2 | S | tpr@0.01 | 0.0010 | 0.0003 | 0.0017 | difference |
| single:stage2 | H | auroc | -0.1610 | -0.1815 | -0.1410 | difference |
| single:stage2 | H | tpr@0.01 | -0.0083 | -0.0203 | 0.0040 | no detected difference |
| single:stage3 | S | auroc | 0.0125 | 0.0107 | 0.0145 | difference |
| single:stage3 | S | tpr@0.01 | 0.0022 | 0.0015 | 0.0030 | difference |
| single:stage3 | H | auroc | -0.0289 | -0.0452 | -0.0131 | difference |
| single:stage3 | H | tpr@0.01 | -0.0257 | -0.0346 | -0.0167 | difference |
| single:pooled | S | auroc | -0.0004 | -0.0022 | 0.0015 | no detected difference |
| single:pooled | S | tpr@0.01 | -0.0003 | -0.0010 | 0.0004 | no detected difference |
| single:pooled | H | auroc | -0.0957 | -0.1112 | -0.0801 | difference |
| single:pooled | H | tpr@0.01 | -0.0307 | -0.0397 | -0.0211 | difference |
| single:logits | S | auroc | 0.0169 | 0.0148 | 0.0190 | difference |
| single:logits | S | tpr@0.01 | 0.0013 | 0.0004 | 0.0023 | difference |
| single:logits | H | auroc | 0.0580 | 0.0409 | 0.0744 | difference |
| single:logits | H | tpr@0.01 | 0.0108 | -0.0033 | 0.0253 | no detected difference |
| per_layer:max | S | auroc | 0.0168 | 0.0151 | 0.0186 | difference |
| per_layer:max | S | tpr@0.01 | 0.0015 | 0.0005 | 0.0024 | difference |
| per_layer:max | H | auroc | -0.0315 | -0.0479 | -0.0150 | difference |
| per_layer:max | H | tpr@0.01 | -0.0008 | -0.0149 | 0.0165 | no detected difference |
| per_layer:mean | S | auroc | 0.0123 | 0.0109 | 0.0137 | difference |
| per_layer:mean | S | tpr@0.01 | -0.0022 | -0.0027 | -0.0016 | difference |
| per_layer:mean | H | auroc | -0.0088 | -0.0215 | 0.0045 | no detected difference |
| per_layer:mean | H | tpr@0.01 | -0.0158 | -0.0256 | -0.0059 | difference |
| per_layer:fpr_weighted | S | auroc | 0.0123 | 0.0109 | 0.0137 | difference |
| per_layer:fpr_weighted | S | tpr@0.01 | -0.0022 | -0.0027 | -0.0016 | difference |
| per_layer:fpr_weighted | H | auroc | -0.0088 | -0.0215 | 0.0045 | no detected difference |
| per_layer:fpr_weighted | H | tpr@0.01 | -0.0158 | -0.0256 | -0.0059 | difference |
| subset:early | S | auroc | 0.0146 | 0.0124 | 0.0169 | difference |
| subset:early | S | tpr@0.01 | 0.0046 | 0.0037 | 0.0056 | difference |
| subset:early | H | auroc | -0.1057 | -0.1297 | -0.0823 | difference |
| subset:early | H | tpr@0.01 | 0.0025 | -0.0142 | 0.0206 | no detected difference |
| subset:middle | S | auroc | 0.0062 | 0.0051 | 0.0074 | difference |
| subset:middle | S | tpr@0.01 | 0.0015 | 0.0009 | 0.0021 | difference |
| subset:middle | H | auroc | -0.0164 | -0.0281 | -0.0047 | difference |
| subset:middle | H | tpr@0.01 | 0.0266 | 0.0133 | 0.0400 | difference |
| subset:late | S | auroc | 0.0063 | 0.0045 | 0.0081 | difference |
| subset:late | S | tpr@0.01 | -0.0001 | -0.0008 | 0.0006 | no detected difference |
| subset:late | H | auroc | 0.0249 | 0.0105 | 0.0390 | difference |
| subset:late | H | tpr@0.01 | -0.0216 | -0.0305 | -0.0124 | difference |
| subset:backbone | S | auroc | 0.0123 | 0.0104 | 0.0142 | difference |
| subset:backbone | S | tpr@0.01 | 0.0042 | 0.0033 | 0.0051 | difference |
| subset:backbone | H | auroc | -0.0727 | -0.0932 | -0.0522 | difference |
| subset:backbone | H | tpr@0.01 | 0.0075 | -0.0092 | 0.0259 | no detected difference |
| subset:without_stem | S | auroc | -0.0010 | -0.0014 | -0.0005 | difference |
| subset:without_stem | S | tpr@0.01 | -0.0001 | -0.0005 | 0.0003 | no detected difference |
| subset:without_stem | H | auroc | 0.0048 | 0.0005 | 0.0090 | difference |
| subset:without_stem | H | tpr@0.01 | -0.0017 | -0.0069 | 0.0033 | no detected difference |
| subset:without_stage1 | S | auroc | 0.0058 | 0.0052 | 0.0064 | difference |
| subset:without_stage1 | S | tpr@0.01 | 0.0020 | 0.0016 | 0.0025 | difference |
| subset:without_stage1 | H | auroc | 0.0289 | 0.0233 | 0.0346 | difference |
| subset:without_stage1 | H | tpr@0.01 | 0.0266 | 0.0156 | 0.0368 | difference |
| subset:without_stage2 | S | auroc | 0.0025 | 0.0013 | 0.0036 | difference |
| subset:without_stage2 | S | tpr@0.01 | 0.0007 | 0.0001 | 0.0013 | difference |
| subset:without_stage2 | H | auroc | 0.0653 | 0.0558 | 0.0755 | difference |
| subset:without_stage2 | H | tpr@0.01 | -0.0058 | -0.0146 | 0.0033 | no detected difference |
| subset:without_stage3 | S | auroc | 0.0062 | 0.0050 | 0.0073 | difference |
| subset:without_stage3 | S | tpr@0.01 | 0.0004 | -0.0003 | 0.0010 | no detected difference |
| subset:without_stage3 | H | auroc | -0.0147 | -0.0256 | -0.0029 | difference |
| subset:without_stage3 | H | tpr@0.01 | -0.0066 | -0.0189 | 0.0075 | no detected difference |
| subset:without_pooled | S | auroc | 0.0124 | 0.0109 | 0.0140 | difference |
| subset:without_pooled | S | tpr@0.01 | 0.0014 | 0.0007 | 0.0022 | difference |
| subset:without_pooled | H | auroc | -0.0102 | -0.0250 | 0.0042 | no detected difference |
| subset:without_pooled | H | tpr@0.01 | -0.0008 | -0.0158 | 0.0154 | no detected difference |
| subset:without_logits | S | auroc | 0.0024 | 0.0019 | 0.0028 | difference |
| subset:without_logits | S | tpr@0.01 | -0.0002 | -0.0006 | 0.0002 | no detected difference |
| subset:without_logits | H | auroc | -0.0276 | -0.0324 | -0.0231 | difference |
| subset:without_logits | H | tpr@0.01 | -0.0050 | -0.0097 | 0.0000 | no detected difference |
| subset:greedy_k2 | S | auroc | -0.0024 | -0.0036 | -0.0011 | difference |
| subset:greedy_k2 | S | tpr@0.01 | 0.0023 | 0.0017 | 0.0029 | difference |
| subset:greedy_k2 | H | auroc | -0.0458 | -0.0572 | -0.0330 | difference |
| subset:greedy_k2 | H | tpr@0.01 | 0.0008 | -0.0076 | 0.0096 | no detected difference |
| subset:greedy_k3 | S | auroc | -0.0002 | -0.0012 | 0.0010 | no detected difference |
| subset:greedy_k3 | S | tpr@0.01 | 0.0015 | 0.0008 | 0.0021 | difference |
| subset:greedy_k3 | H | auroc | -0.0516 | -0.0624 | -0.0400 | difference |
| subset:greedy_k3 | H | tpr@0.01 | -0.0008 | -0.0110 | 0.0106 | no detected difference |
| subset:greedy_k4 | S | auroc | -0.0007 | -0.0014 | 0.0001 | no detected difference |
| subset:greedy_k4 | S | tpr@0.01 | 0.0012 | 0.0007 | 0.0018 | difference |
| subset:greedy_k4 | H | auroc | -0.0878 | -0.0969 | -0.0796 | difference |
| subset:greedy_k4 | H | tpr@0.01 | -0.0075 | -0.0151 | 0.0008 | no detected difference |

### H2 (SPEC §1)

| track | best_single | best_single_tpr | joint_tpr | diff_single_minus_joint | ci_low | ci_high | h2 |
|---|---|---|---|---|---|---|---|
| S | single:stem | 0.0142 | 0.0101 | 0.0041 | 0.0031 | 0.0052 | rejected (fused not better within CI) |
| H | single:logits | 0.0448 | 0.0340 | 0.0108 | -0.0033 | 0.0253 | rejected (fused not better within CI) |

The best single layer is picked on the evaluation data, which favours the single-layer side, so this is a conservative test of H2.

### Localisation by the per-layer detector (first alarm, max)

| track | alarmed | first alarm in injected region | chance |
|---|---|---|---|
| S | 6254 | 37.2 % | 33.3 % |
| H | 114 | 43.9 % | 33.3 % |

### Cost of observing K taps (M-6, FP32, CPU, batch 1)

| k | model_ms | features_added_ms |
|---|---|---|
| 1 | 0.967 | 0.263 |
| 2 | 0.967 | 0.526 |
| 3 | 0.967 | 0.845 |
| 6 | 0.967 | 1.671 |
| 13 | 0.967 | 2.694 |
