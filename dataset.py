import torch
import torchvision
import torchvision.transforms as transforms
from collections import defaultdict
from tqdm import tqdm
from utils.edge import rgb_to_gray_tensor, extract_edges, get_edge_gradient_direction, get_neighbor_pixels_by_gradient
from utils.polynomial import fit_polynomial_with_gradient

def extract_gradient_polynomials(image, degree=2, max_patches=100):
    H, W = image.shape[1], image.shape[2]
    gray = rgb_to_gray_tensor(image)
    edge_coords = extract_edges(gray)
    angle_map = get_edge_gradient_direction(gray)

    patch_polys = {'red': [], 'green': [], 'blue': []}
    edge_tuples = {'red': [], 'green': [], 'blue': []}
    count = 0

    for (y, x) in edge_coords:
        if count >= max_patches:
            break
        angle = angle_map[y, x]
        neighbors = get_neighbor_pixels_by_gradient(y, x, angle, step=1, H=H, W=W)
        if neighbors is None:
            continue
        (y1, x1), (y2, x2) = neighbors

        x1n, y1n = x1 / (W - 1), y1 / (H - 1)
        x2n, y2n = x2 / (W - 1), y2 / (H - 1)

        for c_idx, channel in enumerate(['red', 'green', 'blue']):
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

            edge_tuples[channel].append(((x1n, y1n, v1), (x2n, y2n, v2), (gx, gy)))

        count += 1

    for channel in ['red', 'green', 'blue']:
        if len(edge_tuples[channel]) >= 2:
            coeff = fit_polynomial_with_gradient(edge_tuples[channel])
            patch_polys[channel].append(coeff)

    return patch_polys

def compute_classwise_mean_polynomials(patchwise_polynomials, classes):
    class_representatives = {}
    for class_name in classes:
        class_representatives[class_name] = {}
        for channel in ['red', 'green', 'blue']:
            all_vecs = patchwise_polynomials[class_name][channel]
            if len(all_vecs) == 0:
                print(f"[WARNING] {class_name} 클래스의 {channel} 채널 prior 없음")
                continue
            stacked = torch.stack(all_vecs)
            mean_vec = stacked.mean(dim=0)
            class_representatives[class_name][channel] = mean_vec
    print("[INFO] 클래스별 평균 prior 다항식 생성 완료 ✅")
    return class_representatives

def load_data_and_preprocess(degree=2, max_patches_per_image=50, max_images_per_class=1000):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)
    classes = trainset.classes

    patchwise_polynomials = {
        class_name: {'red': [], 'green': [], 'blue': []}
        for class_name in classes
    }

    print("[INFO] 엣지 기반 다항식 prior 추출 중...")

    # ✅ 클래스별 이미지 미리 모으기 (속도 개선 + tqdm 정확도 향상)
    classwise_images = defaultdict(list)
    for img, label in trainset:
        class_name = classes[label]
        classwise_images[class_name].append(img)

    for class_name in tqdm(classes, desc="전체 클래스 진행"):
        images = classwise_images[class_name][:max_images_per_class]
        for img in tqdm(images, desc=f"→ {class_name}", leave=False):
            patch_polys = extract_gradient_polynomials(img, degree=degree, max_patches=max_patches_per_image)
            for channel in ['red', 'green', 'blue']:
                patchwise_polynomials[class_name][channel].extend(patch_polys[channel])

        total = sum(len(patchwise_polynomials[class_name][ch]) for ch in ['red', 'green', 'blue'])
        print(f"[DEBUG] {class_name} 클래스: 총 다항식 {total}개")

    print("[INFO] 모든 클래스 prior 추출 완료 ✅")

    classwise_representatives = compute_classwise_mean_polynomials(patchwise_polynomials, classes)

    return trainloader, testloader, classes, patchwise_polynomials, classwise_representatives
