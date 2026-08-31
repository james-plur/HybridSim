# Analytical vs non-dummy Frontier RF

Predictor: `llama2_7b_dense_example` @ `h800`, `enable_dummy_mode=False`.
Analytical: Llama-2-7B shape, `duration_scale=1.0`, `compute_util=hbm_util=0.6`.
Note: 5% gate suspended after shape-primitive refactor (was `MAX_REL_ERR = 0.05`).

| case | analytical_s | rf_s | analytical/rf | rel_err |
|------|-------------:|-----:|--------------:|--------:|
| multi_prefill | 1.166234e-02 | 1.176600e-02 | 0.9912 | 0.0088 |
| multi_decode | 1.083426e-02 | 1.064961e-02 | 1.0173 | 0.0173 |
| multi_prefill_with_kv_cache | 1.201761e-02 | 1.177315e-02 | 1.0208 | 0.0208 |
| single_prefill | 1.122887e-02 | 1.121803e-02 | 1.0010 | 0.0010 |
| single_decode | 1.070621e-02 | 1.065048e-02 | 1.0052 | 0.0052 |
| single_prefill_cache | 1.128372e-02 | 1.121803e-02 | 1.0059 | 0.0059 |

