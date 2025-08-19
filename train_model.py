import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from unet import UNet  # your UNet implementation
from readdata import SegmentationDataset

def parse_args():
    parser = argparse.ArgumentParser("Train UNet Model")
    parser.add_argument("--data_dir", type=str, required=True, help="Dataset directory")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--model_output", type=str, default="unet_model.pth", help="Path to save trained model")
    return parser.parse_args()


def main():
    args = parse_args()

    # Detect device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")

    # Load datasets
    train_dataset = SegmentationDataset(f"{args.data_dir}/train.csv")
    test_dataset = SegmentationDataset(f"{args.data_dir}/test.csv")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Number of classes
    all_masks = np.concatenate([mask.numpy().flatten() for _, mask in train_dataset])
    n_classes = len(np.unique(all_masks))

    # Define model, loss, optimizer
    model = UNet(n_classes=n_classes, in_channels=3).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    train_losses, test_losses = [], []

    # Training loop
    for epoch in range(args.epochs):
        model.train()
        total_train_loss = 0
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Evaluation on test data
        model.eval()
        total_test_loss = 0
        with torch.no_grad():
            for images, masks in test_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                total_test_loss += loss.item()

        avg_test_loss = total_test_loss / len(test_loader)
        test_losses.append(avg_test_loss)

        print(f"Epoch {epoch+1}/{args.epochs}, Train Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}")

    # Save model
    torch.save(model.state_dict(), args.model_output)
    print(f"Model saved to {args.model_output}")


if __name__ == "__main__":
    main()

