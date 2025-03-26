import torch
import torchvision
import torchvision.transforms as transforms
from collections import defaultdict
from tqdm import tqdm
from utils.edge import rgb_to_gray_tensor, extract_edges, get_edge_gradient_direction, get_neighbor_pixels_by_gradient
from utils.polynomial import fit_polynomial_with_gradient

def save_polynomials(patchwise_polynomials, filename):
    torch.save(patchwise_polynomials, filename)
    print(f"다항식이 {filename}에 저장되었습니다.")

def load_polynomials(filename):
    patchwise_polynomials = torch.load(filename)
    print(f"다항식이 {filename}에서 불러와졌습니다.")
    return patchwise_polynomials

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

def load_data_and_preprocess(degree=2, max_patches_per_image=50, max_images_per_class=1000, save_file='patchwise_polynomials.pth'):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)
    classes = trainset.classes

    # 기존 다항식 파일이 있으면 로드
    try:
        patchwise_polynomials = load_polynomials(save_file)
        print(f"저장된 다항식 파일 {save_file}을 성공적으로 불러왔습니다.")
        # 저장된 다항식을 불러온 경우에도 클래스 대표 다항식을 계산합니다.
        classwise_representatives = compute_classwise_mean_polynomials(patchwise_polynomials, classes)
        return trainloader, testloader, classes, patchwise_polynomials, classwise_representatives
    except FileNotFoundError:
        print(f"저장된 다항식 파일 {save_file}을 찾을 수 없습니다. 다항식을 추출하여 저장합니다.")

    # 다항식 추출
    patchwise_polynomials = {
        class_name: {'red': [], 'green': [], 'blue': []}
        for class_name in classes
    }

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

    # 추출한 다항식을 저장
    save_polynomials(patchwise_polynomials, save_file)

    return trainloader, testloader, classes, patchwise_polynomials, classwise_representatives
