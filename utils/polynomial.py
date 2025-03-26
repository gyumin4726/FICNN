import torch

def build_feature_vector(x, y):
    return torch.tensor([1, x, y, x**2, x*y, y**2], dtype=torch.float32)

def build_gradient_vector(x, y):
    # ∂f/∂x = a1 + 2a3x + a4y
    # ∂f/∂y = a2 + a4x + 2a5y
    df_dx = torch.tensor([0, 1, 0, 2 * x, y, 0], dtype=torch.float32)
    df_dy = torch.tensor([0, 0, 1, 0, x, 2 * y], dtype=torch.float32)
    return df_dx, df_dy

def fit_polynomial_with_gradient(edge_points):
    """
    edge_points: List of tuples like:
    [((x1, y1, v1), (x2, y2, v2), (gx, gy)), ...]
    """
    features = []
    targets = []

    for (x1, y1, v1), (x2, y2, v2), (gx, gy) in edge_points:
        xm, ym = (x1 + x2) / 2, (y1 + y2) / 2

        # 함수값 방정식 2개
        features.append(build_feature_vector(x1, y1))
        targets.append(v1)
        features.append(build_feature_vector(x2, y2))
        targets.append(v2)

        # 기울기 방정식 2개 (x, y)
        dfdx, dfdy = build_gradient_vector(xm, ym)
        features.append(dfdx)
        targets.append(gx)
        features.append(dfdy)
        targets.append(gy)

    A = torch.stack(features)  # (4N, 6)
    b = torch.tensor(targets, dtype=torch.float32).unsqueeze(1)  # (4N, 1)

    solution = torch.linalg.lstsq(A, b)
    coeffs = solution.solution.squeeze()
    return coeffs
