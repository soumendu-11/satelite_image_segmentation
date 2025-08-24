# To run python pipeline.py --data_dir . --epochs 3 --batch_size 4 --output_dir cam_results

# =========================
# Local Training Pipeline for 3 nodes
# =========================
import argparse
import subprocess

def parse_args():
    parser = argparse.ArgumentParser("Local Training Pipeline for 3 nodes")
    parser.add_argument("--data_dir", type=str, required=True, help="Dataset directory")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--output_dir", type=str, default="pipeline_outputs", help="Directory to save outputs")
    return parser.parse_args()


def main():
    args = parse_args()

    # =========================
    # Step 1: Run prep_data.py
    # =========================
    print("Running prep_data.py ...")
    subprocess.run([
        "python", "prep_data.py",
        "--data_dir", args.data_dir
    ], check=True)

    # =========================
    # Step 2: Run train_test_valid.py
    # =========================
    print("Running train_test_valid.py ...")
    subprocess.run([
        "python", "train_test_valid.py",
        "--data_dir", args.data_dir,
        "--output_dir", args.output_dir
    ], check=True)

    # =========================
    # Step 3: Run training.py
    # =========================
    print("Running training.py ...")
    subprocess.run([
        "python", "training.py",
        "--data_dir", args.data_dir,
        "--epochs", str(args.epochs),
        "--batch_size", str(args.batch_size),
        "--output_dir", args.output_dir
    ], check=True)

    print("Pipeline execution completed!")


if __name__ == "__main__":
    main()
