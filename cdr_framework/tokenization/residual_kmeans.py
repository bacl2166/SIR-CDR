from __future__ import annotations

import torch


def _assign_nearest(
    vectors: torch.Tensor,
    centroids: torch.Tensor,
    *,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    assignments = []
    errors = []
    for start in range(0, len(vectors), chunk_size):
        chunk = vectors[start : start + chunk_size]
        distances = torch.cdist(chunk, centroids)
        error, assignment = distances.min(dim=1)
        assignments.append(assignment)
        errors.append(error)
    return torch.cat(assignments), torch.cat(errors)


def _initial_centroids(
    vectors: torch.Tensor,
    codebook_size: int,
    generator: torch.Generator,
) -> torch.Tensor:
    order = torch.randperm(len(vectors), generator=generator, device="cpu")
    indices = order[: min(len(vectors), codebook_size)].to(vectors.device)
    if len(indices) < codebook_size:
        repeats = (codebook_size + len(indices) - 1) // len(indices)
        indices = indices.repeat(repeats)[:codebook_size]
    return vectors[indices].clone()


def _fit_level(
    vectors: torch.Tensor,
    codebook_size: int,
    iterations: int,
    generator: torch.Generator,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    centroids = _initial_centroids(vectors, codebook_size, generator)
    previous = None
    for _ in range(iterations):
        assignments, errors = _assign_nearest(
            vectors, centroids, chunk_size=chunk_size
        )
        if previous is not None and torch.equal(assignments, previous):
            break
        previous = assignments

        counts = torch.bincount(assignments, minlength=codebook_size)
        sums = torch.zeros_like(centroids)
        sums.index_add_(0, assignments, vectors)
        occupied = counts > 0
        centroids[occupied] = sums[occupied] / counts[occupied, None]

        empty = (~occupied).nonzero(as_tuple=False).flatten()
        if len(empty):
            farthest = torch.argsort(errors, descending=True)
            replacements = farthest.repeat(
                (len(empty) + len(farthest) - 1) // len(farthest)
            )[: len(empty)]
            centroids[empty] = vectors[replacements]

    assignments, _ = _assign_nearest(vectors, centroids, chunk_size=chunk_size)
    return assignments, centroids


def fit_residual_kmeans(
    vectors: torch.Tensor,
    *,
    codebook_size: int,
    token_length: int,
    iterations: int,
    seed: int,
    chunk_size: int = 4096,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if vectors.ndim != 2 or not len(vectors):
        raise ValueError("vectors must have shape [num_items, hidden_dim].")
    if codebook_size <= 0 or token_length <= 0 or iterations <= 0:
        raise ValueError("Residual K-means settings must be positive.")

    residual = vectors.detach().float().clone()
    reconstruction = torch.zeros_like(residual)
    token_columns = []
    codebooks = []
    generator = torch.Generator(device="cpu").manual_seed(seed)
    for _ in range(token_length):
        tokens, centroids = _fit_level(
            residual,
            codebook_size,
            iterations,
            generator,
            chunk_size,
        )
        chosen = centroids[tokens]
        token_columns.append(tokens)
        codebooks.append(centroids)
        reconstruction += chosen
        residual -= chosen
    return (
        torch.stack(token_columns, dim=1),
        reconstruction,
        torch.stack(codebooks, dim=0),
    )


def quantize_residuals(
    vectors: torch.Tensor,
    codebooks: torch.Tensor,
    *,
    chunk_size: int = 4096,
) -> tuple[torch.Tensor, torch.Tensor]:
    if codebooks.ndim != 3 or vectors.ndim != 2:
        raise ValueError("Expected vectors [N, D] and codebooks [L, K, D].")
    residual = vectors.detach().float().clone()
    reconstruction = torch.zeros_like(residual)
    token_columns = []
    for centroids in codebooks:
        tokens, _ = _assign_nearest(residual, centroids, chunk_size=chunk_size)
        chosen = centroids[tokens]
        token_columns.append(tokens)
        reconstruction += chosen
        residual -= chosen
    return torch.stack(token_columns, dim=1), reconstruction
