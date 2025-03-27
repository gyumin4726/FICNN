import torch
from itertools import product

def get_monomial_terms(degree):
    return [(i, j) for i in range(degree + 1) for j in range(degree + 1) if i + j <= degree]

def build_feature_vector(x, y, degree):
    terms = get_monomial_terms(degree)
    return torch.tensor([x**i * y**j for (i, j) in terms], dtype=torch.float32)

def build_gradient_vector(x, y, degree):
    terms = get_monomial_terms(degree)
    df_dx = []
    df_dy = []
    for (i, j) in terms:
        if i == 0:
            df_dx.append(0.0)
        else:
            df_dx.append(i * x**(i - 1) * y**j)
        if j == 0:
            df_dy.append(0.0)
        else:
            df_dy.append(j * x**i * y**(j - 1))
    return torch.tensor(df_dx, dtype=torch.float32), torch.tensor(df_dy, dtype=torch.float32)

def fit_polynomial_with_gradient(edge_points, degree):
    features = []
    targets = []

    for (x1, y1, v1), (x2, y2, v2), (gx, gy) in edge_points:
        xm, ym = (x1 + x2) / 2, (y1 + y2) / 2

        # 함수값 2개
        features.append(build_feature_vector(x1, y1, degree))
        targets.append(v1)
        features.append(build_feature_vector(x2, y2, degree))
        targets.append(v2)

        # 기울기 2개
        dfdx, dfdy = build_gradient_vector(xm, ym, degree)
        features.append(dfdx)
        targets.append(gx)
        features.append(dfdy)
        targets.append(gy)

    A = torch.stack(features)  # (4N, num_terms)
    b = torch.tensor(targets, dtype=torch.float32).unsqueeze(1)  # (4N, 1)

    solution = torch.linalg.lstsq(A, b)
    coeffs = solution.solution.squeeze()
    return coeffs
