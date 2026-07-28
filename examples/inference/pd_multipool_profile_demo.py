#!/usr/bin/env python3
"""Multi Prefill + Multi Decode PD demo with KV transfer, prefix cache, and profile.

Highlights for chrome://tracing:
  - Cluster Dispatch to different Prefill / Decode replicas (2P+2D)
  - EngineReq spans on each replica engine track
  - KvPull on Decode after handoff
  - ReplicaSchedule / ClusterSchedule (near-zero duration)

Open ``profile/pd_multipool_profile_demo.json`` in chrome://tracing or Perfetto.
"""

from __future__ import annotations

from hybridsim.request_profile import default_profile_dir
from hybridsim_infer import (
    InferenceConfig,
    InferenceRequest,
    build_inference_simulation,
)


def main() -> None:
    block_size = 8
    shared_prefix = list(range(100, 100 + 16))
    suffix_a = list(range(200, 216))
    suffix_b = list(range(300, 316))
    suffix_c = list(range(400, 416))

    prompt_a = shared_prefix + suffix_a
    prompt_b = shared_prefix + suffix_b  # partial share with A
    prompt_c = list(prompt_a)  # full reuse of A after Prefill cache
    prompt_d = shared_prefix + suffix_c

    cfg = InferenceConfig(
        cluster_type="pd",
        num_prefill_replicas=2,
        num_decode_replicas=2,
        enable_kv_client=True,
        enable_prefix_caching=True,
        block_size=block_size,
        num_gpu_blocks=512,
        tokens_per_step=8,
        decode_tokens_per_step=1,
        max_num_scheduled_tokens=64,
        max_num_running_reqs=16,
        step_interval=1e-3,
        kv_transfer_s=1e-4,
        kv_bandwidth_gbps=50.0,
        kv_bytes_per_token=16.0,
        kv_lookup_rtt_s=1e-3,
        duration_mode="token_proportional",
        prefill_s_per_token=5e-5,
        decode_s_per_token=2e-4,
        enable_request_profile=True,
        request_profile_path=default_profile_dir() / "pd_multipool_profile_demo.json",
    )
    infra = build_inference_simulation(cfg)
    assert len(infra.replicas) == 4

    requests = [
        # Concurrent Prefills so least-load spreads across P0/P1.
        InferenceRequest(
            request_id=1,
            arrived_at=0.0,
            num_prefill_tokens=len(prompt_a),
            num_decode_tokens=6,
            prompt_token_ids=list(prompt_a),
        ),
        InferenceRequest(
            request_id=2,
            arrived_at=0.0,
            num_prefill_tokens=len(prompt_b),
            num_decode_tokens=4,
            prompt_token_ids=list(prompt_b),
        ),
        InferenceRequest(
            request_id=3,
            arrived_at=0.001,
            num_prefill_tokens=len(prompt_d),
            num_decode_tokens=4,
            prompt_token_ids=list(prompt_d),
        ),
        InferenceRequest(
            request_id=4,
            arrived_at=0.001,
            num_prefill_tokens=len(prompt_b),
            num_decode_tokens=3,
            prompt_token_ids=list(prompt_b),
        ),
        # Later arrivals can hit Prefill local prefix cache after earlier handoffs.
        InferenceRequest(
            request_id=5,
            arrived_at=0.35,
            num_prefill_tokens=len(prompt_c),
            num_decode_tokens=5,
            prompt_token_ids=list(prompt_c),
        ),
        InferenceRequest(
            request_id=6,
            arrived_at=0.36,
            num_prefill_tokens=len(prompt_a),
            num_decode_tokens=3,
            prompt_token_ids=list(prompt_a),
        ),
    ]

    print("=== PD multipool profile demo (2P+2D, KV, prefix) ===")
    print(
        f"replicas={len(infra.replicas)} profile={cfg.request_profile_path}"
    )
    infra.schedule_arrivals(requests)
    infra.run()
    infra.check_errors()

    finished = sorted(infra.finished_requests, key=lambda r: r.request_id)
    print(
        f"arrived={infra.cluster.arrived_count} finished={len(finished)} "
        f"now={infra.now:.4f}s"
    )
    for req in finished:
        params = req.kv_transfer_params or {}
        print(
            f"  req={req.request_id} completed={req.completed} "
            f"src_P={params.get('remote_replica_id')} "
            f"handed_off={bool(params.get('_handed_off'))}"
        )
        if not req.completed:
            raise SystemExit(f"req {req.request_id} did not complete")

    if len(finished) != len(requests):
        raise SystemExit(f"expected {len(requests)} finished, got {len(finished)}")
    if infra.profile_path is not None:
        print(f"request profile: {infra.profile_path}")
        print("Open in chrome://tracing or https://ui.perfetto.dev/")
    print("ok")


if __name__ == "__main__":
    main()
