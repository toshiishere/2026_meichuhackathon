import numpy as np
import torch
import torch.nn as nn
import argparse
from util import load_data_n_model
import csv
import os
from datetime import datetime
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import random
from sklearn.metrics import f1_score



def set_random_seed(seed=666):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True



def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device).long()

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * inputs.size(0)

            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    acc = np.mean(np.array(all_preds) == np.array(all_labels))
    macro_f1 = f1_score(all_labels, all_preds, average='macro')

    return acc, macro_f1, avg_loss



def train(model, train_loader, test_loader, num_epochs,
          learning_rate, criterion, device,
          csv_path, model_name):

    model = model.to(device)

    optimizer = optim.AdamW(model.parameters(),
                            lr=learning_rate,
                            weight_decay=1e-4)

    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

    os.makedirs(csv_path, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    train_csv = os.path.join(
        csv_path,
        f'training_results_{model_name}_{timestamp}.csv'
    )

    final_test_csv = os.path.join(
        csv_path,
        f'final_test_results_{model_name}_{timestamp}.csv'
    )

    # 写训练日志表头
    with open(train_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Epoch',
            'TrainAcc',
            'TrainLoss',
            'TestAcc',
            'TestF1',
            'TestLoss'
        ])

    best_train_acc = 0.0
    best_model_path = os.path.join(
        csv_path,
        f'best_model_{model_name}.pth'
    )

 
    for epoch in range(num_epochs):
        model.train()

        running_loss = 0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device).long()

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / len(train_loader.dataset)
        train_acc = correct / total

     
        test_acc, test_f1, test_loss = evaluate(
            model, test_loader, criterion, device
        )

        print(
        f"Epoch [{epoch+1}/{num_epochs}] "
        f"TrainAcc: {train_acc:.4f} "
        f"TrainLoss: {train_loss:.4f} "
        f"TestAcc: {test_acc:.4f} "
        f"F1: {test_f1:.4f} "
        f"TestLoss: {test_loss:.4f}"
        )

     
        with open(train_csv, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch + 1,
                train_acc,
                train_loss,
                test_acc,
                test_f1,
                test_loss
            ])

        
        if train_acc > best_train_acc:
            best_train_acc = train_acc
            torch.save(model.state_dict(), best_model_path)

        scheduler.step()

    print("\nTraining completed.")
    print(f"Best Train Acc: {best_train_acc:.4f}")


    print("\nEvaluating Best Model on Test Set...")

    model.load_state_dict(torch.load(best_model_path))
    model = model.to(device)

    final_test_acc, final_test_f1, final_test_loss = evaluate(
        model, test_loader, criterion, device
    )

    print("\n========== Final Test Results ==========")
    print(f"Test Accuracy : {final_test_acc:.4f}")
    print(f"Test Macro-F1 : {final_test_f1:.4f}")
    print(f"Test Loss     : {final_test_loss:.6f}")
    print("========================================\n")

  
    with open(final_test_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Model', 'TestAccuracy', 'TestMacroF1', 'TestLoss'])
        writer.writerow([
            model_name,
            final_test_acc,
            final_test_f1,
            final_test_loss
        ])

    print(f"Final test results saved to: {final_test_csv}")
    print(f"Training logs saved to: {train_csv}")

    return best_model_path



def main():
    root = './Data'
    csv_path = './training_logs/'

    torch.cuda.empty_cache()
    set_random_seed(seed=666)

    parser = argparse.ArgumentParser('ESP-Fi HAR Benchmark')

    parser.add_argument('--dataset',
                        choices=['ESP-Fi_HAR'],
                        default='ESP-Fi_HAR')

    parser.add_argument('--model',
                        choices=[
                            'CNN', 'ResNet18',
                            'GRU', 'LSTM', 'Transformer',
                            'MobileNetV3','EfficientNetLite'
                        ],
                        required=True)

    args = parser.parse_args()

    train_loader, test_loader, model, train_epoch = \
        load_data_n_model(args.dataset, args.model, root)

    criterion = nn.CrossEntropyLoss()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    train(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        num_epochs=train_epoch,
        learning_rate=1e-3,
        criterion=criterion,
        device=device,
        csv_path=csv_path,
        model_name=args.model
    )


if __name__ == "__main__":
    main()