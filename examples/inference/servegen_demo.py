#!/usr/bin/env python3
"""NO_NETWORK demo: ServeGen RequestGenerator → schedule_from_generator."""

from __future__ import annotations

from hybridsim_infer import (
    InferenceConfig,
    ServeGenRequestGenerator,
    build_inference_simulation,
)


def main() -> None:
    from hybridsim.request_profile import default_profile_dir

    cfg = InferenceConfig(
        num_replicas=1,
        step_interval=1e-3,
        duration_mode="batch_level",
        batch_predictor="token_proportional",
        prefill_s_per_token=1e-5,
        decode_s_per_token=1e-4,
        tokens_per_step=64,
        max_num_scheduled_tokens=256,
        max_num_running_reqs=32,
        num_gpu_blocks=4096,
        enable_request_profile=True,
        request_profile_path=default_profile_dir() / "servegen_demo.json",
    )
    infra = build_inference_simulation(cfg)

    gen = ServeGenRequestGenerator(
        model="m-small",
        duration=60,
        rate=2.0,
        seed=0,
        max_requests=8,
    )
    requests = infra.schedule_from_generator(gen)
    infra.run()
    infra.check_errors()

    finished = infra.finished_requests
    print(
        f"generated={len(requests)} arrived={infra.cluster.arrived_count} "
        f"finished={len(finished)} now={infra.now:.4f}"
    )
    for req in finished[:5]:
        print(
            f"  req={req.request_id} arrived={req.arrived_at:.4f} "
            f"prefill={req.num_prefill_tokens} decode={req.num_decode_tokens} "
            f"completed={req.completed}"
        )
    if len(finished) != len(requests):
        raise SystemExit(f"expected {len(requests)} finished, got {len(finished)}")
    if infra.profile_path is not None:
        print(f"request profile: {infra.profile_path}")


if __name__ == "__main__":
    main()
