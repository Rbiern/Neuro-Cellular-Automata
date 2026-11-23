"""Train the Studio pattern demo on simulated Gray–Scott trajectories."""

import argparse
from pathlib import Path

import torch
from torch import Tensor

from compute import configure_cpu, positive_int, thread_count
from nca import PATTERN_MODE, ModelSettings, PatternCA, save_model


def gray_scott_delta(local_state: Tensor) -> Tensor:
    """Gray–Scott kinetics (Pearson, 1993, doi:10.1126/science.261.5118.189)."""
    u, v, lap_u, lap_v = local_state.unbind(dim=1)
    reaction = u * v.square()
    return torch.stack(
        (0.16 * lap_u - reaction + 0.025 * (1 - u), 0.08 * lap_v + reaction - (0.025 + 0.055) * v),
        dim=1,
    )


@torch.no_grad()
def sample_trajectories(model: PatternCA, generator: torch.Generator) -> Tensor:
    field = model.initial_field(48, generator, batch=12)
    samples = []
    for step in range(800):
        features = model.perceive(field)
        if step % 16 == 0:
            flat = features.permute(0, 2, 3, 1).reshape(-1, 4)
            indices = torch.randint(len(flat), (2048,), generator=generator)
            samples.append(flat[indices])
        field = (field + gray_scott_delta(features)).clamp(0, 1)
    return torch.cat(samples).view(-1, 4, 1, 1)


def train_pattern(output: Path, iterations: int, seed: int, cpu_threads: int = 0) -> None:
    workers = configure_cpu(cpu_threads, automatic=1)
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    settings = ModelSettings(72, PATTERN_MODE, "Gray–Scott", perception="isotropic", fire_rate=1)
    model = PatternCA()
    print(f"Generating Gray–Scott trajectories on CPU ({workers} threads)…", flush=True)
    features = sample_trajectories(model, generator)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.008)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, iterations, eta_min=0.00015)
    for iteration in range(iterations):
        indices = torch.randint(len(features), (384,), generator=generator)
        # Vary the neighborhood independently of the cell center to distinguish
        # transport from reaction and include states a brush stroke could create.
        levels = torch.rand((128, 2, 1, 1), generator=generator)
        neighbors = torch.rand(levels.shape, generator=generator)
        perturbed = torch.cat((levels, neighbors - levels), dim=1)
        batch = torch.cat((features[indices], perturbed))
        target = gray_scott_delta(batch)
        prediction = model.predict_delta(batch)
        loss = (10 * (prediction - target)).square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()
        if iteration == 0 or (iteration + 1) % 500 == 0 or iteration + 1 == iterations:
            print(
                f"{iteration + 1:>5}/{iterations}  update RMSE {loss.sqrt().item() / 10:.6f}",
                flush=True,
            )
    save_model(output, model, settings)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=positive_int, default=6000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--cpu-threads", type=thread_count, default=0, help="CPU workers (0 chooses automatically)"
    )
    args = parser.parse_args()
    train_pattern(args.output, args.iterations, args.seed, args.cpu_threads)


if __name__ == "__main__":
    main()
