# Analytical vs non-dummy Frontier RF

Predictor: `llama2_7b_dense_example` @ `h800`, `enable_dummy_mode=False`.
Analytical: Llama-2-7B shape, `duration_scale=1.0`, `compute_util=hbm_util=0.6`.
Gate: `MAX_REL_ERR = 0.05`.

| case | analytical_s | rf_s | analytical/rf | rel_err |
|------|-------------:|-----:|--------------:|--------:|
| multi_prefill | 1.175834e-02 | 1.176600e-02 | 0.9993 | 0.0007 |
| multi_decode | 1.100097e-02 | 1.064961e-02 | 1.0330 | 0.0330 |
| multi_prefill_with_kv_cache | 1.221646e-02 | 1.177315e-02 | 1.0377 | 0.0377 |
| single_prefill | 1.125629e-02 | 1.121803e-02 | 1.0034 | 0.0034 |
| single_decode | 1.070664e-02 | 1.065048e-02 | 1.0053 | 0.0053 |
| single_prefill_cache | 1.131115e-02 | 1.121803e-02 | 1.0083 | 0.0083 |

