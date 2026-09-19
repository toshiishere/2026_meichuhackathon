from dataset import ESP_Fi_HAR_Dataset
from ESP_Fi_model import *
import torch


def load_data_n_model(dataset_name, model_name, root):
    """
    Load ESP-Fi HAR dataset and corresponding model

    Args:
        dataset_name (str): 'ESP_Fi_HAR'
        model_name (str): model name
        root (str): dataset root directory

    Returns:
        train_loader, test_loader, model, train_epoch
    """

    if dataset_name != 'ESP-Fi_HAR':
        raise ValueError(
            f"Unsupported dataset: {dataset_name}. "
            f"Only 'ESP-Fi_HAR' is supported."
        )

    print('Using dataset: ESP-Fi HAR')

    num_classes = 7

    # =====================
    # Dataset & DataLoader
    # =====================
    train_loader = torch.utils.data.DataLoader(
        dataset=ESP_Fi_HAR_Dataset(
            root_dir=root,
            split='train_amp'
        ),
        batch_size=4, #32,64
        shuffle=True
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=ESP_Fi_HAR_Dataset(
            root_dir=root,
            split='test_amp'
        ),
        batch_size=4, #32,64
        shuffle=False
    )

    # =====================
    # Model Selection
    # =====================


    if model_name == 'CNN':
        model = CNN(num_classes)
        train_epoch = 50

    elif model_name == 'ResNet18':
        model = ESP_Fi_ResNet18(num_classes)
        train_epoch = 50

    elif model_name == 'Transformer':
        model = ESP_Fi_Transformer(num_classes)
        train_epoch = 100

    elif model_name == 'GRU':
        model = ESP_Fi_GRU(num_classes)
        train_epoch = 100

    elif model_name == 'LSTM':
        model = ESP_Fi_LSTM(num_classes)
        train_epoch = 100

    elif model_name == 'MobileNetV3':
        model = MobileNetV3(num_classes)
        train_epoch = 50

    elif model_name == 'EfficientNetLite':
        model = EfficientNetLite(num_classes)
        train_epoch = 50

    else:
        raise ValueError(f"Unsupported model: {model_name}")

    return train_loader, test_loader, model, train_epoch
