# To run python pipeline.py --data_dir . --epochs 3 --output_dir cam_results
 
 
import argparse
import subprocess

def parse_args():
    parser = argparse.ArgumentParser("Local Training Pipeline")
    parser.add_argument("--data_dir", type=str, required=True, help="Dataset directory")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--output_dir", type=str, default="cam_outputs", help="Directory to save CAM results")
    return parser.parse_args()

def main():
    args = parse_args()

    # Step 1: Train model
    subprocess.run([
        "python", "train_model.py",
        "--data_dir", args.data_dir,
        "--epochs", str(args.epochs),
        "--batch_size", str(args.batch_size),
        "--model_output", "unet_model.pth"
    ], check=True)

    # Step 2: Run CAM visualization (for all classes, saving PNGs)
    subprocess.run([
        "python", "unet_CAM.py",
        "--data_dir", args.data_dir,
        "--model_path", "unet_model.pth",
        "--output_dir", args.output_dir
    ], check=True)


if __name__ == "__main__":
    main()
