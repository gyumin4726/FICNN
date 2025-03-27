import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from models.simple_cnn import SimpleCNN
import torchvision
from collections import defaultdict
from torch.utils.data import Subset
#✅ CUDA 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"✅ 사용 중인 디바이스: {device}")

# ✅ 데이터셋 및 전처리
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

from collections import defaultdict
from torch.utils.data import Subset
trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)

# ✅ 클래스별 5장만 뽑기
def get_few_shot_subset(dataset, k=5):
    class_counts = defaultdict(int)
    indices = []

    for idx, (_, label) in enumerate(dataset):
        if class_counts[label] < k:
            indices.append(idx)
            class_counts[label] += 1
        if len(indices) >= k * 100:  # CIFAR-100 기준
            break

    return Subset(dataset, indices)


# ✅ 5-shot trainset으로 교체
fewshot_trainset = get_few_shot_subset(trainset, k=5)
trainloader = DataLoader(fewshot_trainset, batch_size=64, shuffle=True)
testloader = DataLoader(testset, batch_size=64, shuffle=False)
# 모델, 손실 함수, 옵티마이저 정의
model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ✅ 학습 함수
def train(model, loader, optimizer, criterion):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total

# ✅ 테스트 함수
def evaluate(model, loader, criterion):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            test_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

    return test_loss / total, correct / total

# ✅ 메인 학습 루프
for epoch in range(1, 2):  # 1 epoch만 (비교 목적)
    train_loss, train_acc = train(model, trainloader, optimizer, criterion)
    test_loss, test_acc = evaluate(model, testloader, criterion)

    print(f"[Epoch {epoch}] Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
