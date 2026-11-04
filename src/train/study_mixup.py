"""视角签名匹配的 study 级 Mixup。"""
import torch


def mixup_studies(images: torch.Tensor, offsets: torch.Tensor, labels: torch.Tensor,
                   views: torch.Tensor, *, alpha: float, probability: float) -> tuple[torch.Tensor, torch.Tensor, int]:
    if probability == 0:
        return images, labels, 0
    boundaries = offsets.detach().cpu().tolist()
    view_list = views.detach().cpu().tolist()
    groups: dict[tuple[int, int], list[int]] = {}
    indices: list[tuple[list[int], list[int]]] = []
    for start, end in zip(boundaries, boundaries[1:]):
        segment = view_list[start:end]
        front = [start + i for i, view in enumerate(segment) if view == 0]
        lateral = [start + i for i, view in enumerate(segment) if view == 1]
        indices.append((front, lateral))
        if len(front) + len(lateral) == len(segment):
            groups.setdefault((len(front), len(lateral)), []).append(len(indices) - 1)
    mixed_images, mixed_labels = images.clone(), labels.clone()
    count = 0
    for group in groups.values():
        if len(group) < 2:
            continue
        order = torch.randperm(len(group)).tolist()
        for pos in range(0, len(order) - 1, 2):
            if torch.rand(()).item() >= probability:
                continue
            a, b = group[order[pos]], group[order[pos + 1]]
            lam = torch.distributions.Beta(alpha, alpha).sample().item()
            ia = indices[a][0] + indices[a][1]
            ib = indices[b][0] + indices[b][1]
            mixed_images[ia] = lam * images[ia] + (1 - lam) * images[ib]
            mixed_images[ib] = lam * images[ib] + (1 - lam) * images[ia]
            mixed_labels[a] = lam * labels[a] + (1 - lam) * labels[b]
            mixed_labels[b] = lam * labels[b] + (1 - lam) * labels[a]
            count += 2
    return mixed_images, mixed_labels, count
