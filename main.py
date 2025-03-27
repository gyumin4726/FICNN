import torch
import torch.nn as nn
import torch.optim as optim
from models.simple_cnn import SimpleCNN
from dataset import load_data_and_preprocess
from train import train
from test import test

# 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("✅ 사용 중인 디바이스:", device)

degree = 5
λ = 0.01
EPOCHS = 10
LEARNING_RATE = 0.001

# 데이터 로드 및 구조적 prior 계산
trainloader, testloader, classes, patchwise_polynomials, class_representatives = load_data_and_preprocess(degree=degree)

# 대표 다항식 출력
print("\n🎯 예시로 몇 개의 클래스 대표 다항식을 출력합니다:\n")
for class_name in list(class_representatives.keys())[:2]:
    print(f"📦 클래스: {class_name}")
    for channel in ['red', 'green', 'blue']:
        coeff = class_representatives[class_name].get(channel, None)
        if coeff is not None:
            coeff_list = [round(float(c), 4) for c in coeff]
            print(f"  🎨 채널: {channel} → 계수: {coeff_list}")
        else:
            print(f"  🎨 채널: {channel} → (다항식 없음)")


# 모델, 손실 함수, 옵티마이저 정의
model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# 학습 루프
for epoch in range(EPOCHS):
    train_loss, train_acc = train(model, trainloader, class_representatives, degree, λ, classes, criterion, optimizer, device)
    test_loss, test_acc = test(model, testloader, criterion, device)

    print(f"[Epoch {epoch+1}] "
          f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
          f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
