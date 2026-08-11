"""Print available red/green color profiles and trajectory counts."""

from __future__ import annotations

from config import COLOR_PROFILES, TrainConfig
from grid_paths import build_catalog, make_env


def main() -> None:
    print("Color profiles:\n")
    for name, spec in COLOR_PROFILES.items():
        env = make_env(**TrainConfig(color_profile=name).profile_kwargs())
        cat = build_catalog(env)
        print(f"  {name}:")
        print(f"    {spec['description']}")
        print(f"    red={spec['red_center']} green={spec['green_center']} T={spec['temperature']}")
        print(f"    trajectories={cat.num_trajectories}  reward=[{cat.min_reward:.4f}, {cat.max_reward:.4f}]")
        print()


if __name__ == "__main__":
    main()
