import torch
import torch.nn.functional as F
from utils.edge import rgb_to_gray_tensor, extract_edges, get_edge_gradient_direction, get_neighbor_pixels_by_gradient
from utils.polynomial import fit_polynomial_with_gradient
def prior_matching_loss(input_images, labels, class_representatives, degree, classes, λ=1.0, max_edges=5):
    B, C, H, W = input_images.shape
    device = input_images.device
    total_loss = 0.0
    count = 0

    for b in range(B):
        label_idx = labels[b].item()
        class_name = classes[label_idx]
        image = input_images[b]
        gray = rgb_to_gray_tensor(image).cpu()
        edge_coords = extract_edges(gray)
        angle_map = get_edge_gradient_direction(gray)

        if len(edge_coords) == 0:
            continue

        for c_idx, channel in enumerate(['red', 'green', 'blue']):
            coeff_list = []

            for (y, x) in edge_coords[:max_edges]:
                angle = angle_map[y, x]
                neighbors = get_neighbor_pixels_by_gradient(y, x, angle, step=1, H=H, W=W)
                if neighbors is None:
                    continue
                (y1, x1), (y2, x2) = neighbors

                x1n, y1n = x1 / (W - 1), y1 / (H - 1)
                x2n, y2n = x2 / (W - 1), y2 / (H - 1)
                v1 = image[c_idx, y1, x1].item()
                v2 = image[c_idx, y2, x2].item()

                grad_estimate = v2 - v1
                gx = (x2n - x1n)
                gy = (y2n - y1n)
                norm = gx**2 + gy**2
                if norm == 0:
                    continue
                gx *= grad_estimate / norm
                gy *= grad_estimate / norm

                edge_points = [((x1n, y1n, v1), (x2n, y2n, v2), (gx, gy))]
                coeff = fit_polynomial_with_gradient(edge_points, degree)
                if coeff is not None:
                    coeff_list.append(coeff.to(device))

            if len(coeff_list) == 0:
                continue

            avg_coeff = torch.stack(coeff_list).mean(dim=0)

            if channel not in class_representatives[class_name]:
                continue
            coeff_target = class_representatives[class_name][channel].to(device)
            loss = F.mse_loss(avg_coeff, coeff_target, reduction='sum')

            total_loss += loss
            count += 1

    if count == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)
    return λ * total_loss / count