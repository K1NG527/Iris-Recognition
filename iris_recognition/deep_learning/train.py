"""
Training Pipeline for IrisDeepNet using ArcMargin Loss.
Optimized for efficient training on CPU or GPU.
"""

import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .model import IrisDeepNet
from .dataset import NormalizedIrisDataset


def train_iris_deepnet(
    train_subjects=None,
    epochs=12,
    batch_size=16,
    lr=1e-3,
    weight_decay=1e-4,
    device="cpu",
    save_path="results/deep_learning/iris_deep_net.pt",
    verbose=True
):
    """
    Trains IrisDeepNet on normalized iris strips with ArcFace loss.
    
    Parameters:
    - train_subjects: list of subject IDs to train on (e.g. 25-35 subjects)
    - epochs: number of training epochs
    - batch_size: batch size
    - lr: initial learning rate
    - device: "cpu" or "cuda"
    - save_path: file path to save trained model weights
    
    Returns:
    - model: trained IrisDeepNet model
    - history: dict with loss and accuracy logs
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    if verbose:
        print(f"\nInitializing Normalized Iris Dataset...")
        print(f"  Subjects in training pool : {len(train_subjects) if train_subjects else 'All'}")

    dataset = NormalizedIrisDataset(
        dataset_dir="Dataset",
        segmentation_csv="results/segmentation_results.csv",
        subject_filter=train_subjects,
        is_train=True,
        augment=True,
        cache_in_memory=True
    )

    if len(dataset) == 0:
        raise ValueError("Training dataset is empty! Verify subject filters and segmentation CSV.")

    num_classes = dataset.num_classes
    if verbose:
        print(f"  Loaded training samples   : {len(dataset)}")
        print(f"  Distinct eye identities   : {num_classes}")

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=False
    )

    model = IrisDeepNet(in_channels=1, embedding_dim=256, num_classes=num_classes)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    history = {"loss": [], "accuracy": [], "epoch_times": []}

    if verbose:
        print(f"\nCommencing IrisDeepNet Training ({epochs} epochs, device: {device})...")
        print(f"{'Epoch':^7} | {'Loss':^10} | {'Train Acc':^12} | {'LR':^10} | {'Time':^8}")
        print("-" * 56)

    model.train()
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch in dataloader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            _, logits = model(images, labels)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += images.size(0)

        scheduler.step()
        epoch_time = time.time() - t0
        epoch_loss = running_loss / total
        epoch_acc = (correct / total) * 100.0
        curr_lr = optimizer.param_groups[0]["lr"]

        history["loss"].append(epoch_loss)
        history["accuracy"].append(epoch_acc)
        history["epoch_times"].append(epoch_time)

        if verbose:
            print(f"{epoch:^7d} | {epoch_loss:^10.4f} | {epoch_acc:^10.2f}% | {curr_lr:^10.5f} | {epoch_time:^7.1f}s")

    total_time = time.time() - start_time
    if verbose:
        print("-" * 56)
        print(f"Training completed in {total_time:.1f}s (Final Acc: {history['accuracy'][-1]:.2f}%)")

    # Save checkpoint
    torch.save({
        "model_state_dict": model.state_dict(),
        "embedding_dim": 256,
        "num_classes": num_classes,
        "history": history,
        "identity_to_label": dataset.identity_to_label
    }, save_path)

    if verbose:
        print(f"Model saved to: {save_path}")

    return model, history
