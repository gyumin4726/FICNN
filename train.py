import torch
from tqdm import tqdm
from utils.function_prior_loss import prior_matching_loss

def train(model, loader, class_representatives, degree, λ, classes, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    # tqdm progress bar 설정
    loader = tqdm(loader, desc="[Training]", leave=False, dynamic_ncols=True)

    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)

        ce_loss = criterion(outputs, labels)
        prior_loss = prior_matching_loss(inputs, labels, class_representatives, degree, classes, λ)
        loss = ce_loss + prior_loss

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # 실시간 진행률 표시 업데이트
        loader.set_postfix({
            'Loss': f"{loss.item():.4f}",
            'Acc': f"{100. * correct / total:.2f}%"
        })

    epoch_loss = running_loss / len(loader)
    epoch_acc = correct / total

    return epoch_loss, epoch_acc