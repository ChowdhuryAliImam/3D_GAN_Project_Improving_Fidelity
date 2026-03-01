
import argparse
import os
import torch

import params
from trainer import trainer

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


def parse_args():
    parser = argparse.ArgumentParser(description="3D DCGAN  + PatchGAN")

    # modes
    parser.add_argument("--train", action="store_true",
                        help="Run training")
    parser.add_argument("--test", action="store_true",
                        help="Run testing / sampling")

    # optional name (for consistency with your old scripts)
    parser.add_argument("--model_name", type=str, default="dcgan3d",
                        help="Model name (for bookkeeping only)")

    # optionally override epochs from command line
    parser.add_argument("--epochs", type=int, default=None,
                        help="Number of epochs (overrides params.epochs if set)")

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    if args.epochs is not None:
        params.epochs = args.epochs

    print("torch.cuda.is_available():", torch.cuda.is_available())
    print("Using device:", params.device)
    print("Output directory:", params.output_dir)


    os.makedirs(params.output_dir, exist_ok=True)

    if args.train:
        print("==> Training mode")
        trainer(args)

    elif args.test:
        print("==> Test mode not yet implemented for new architecture.")
        print("You can later implement tester(args) and call it here.")

    else:
        print("Please specify --train or --test")
        print("Example:")
        print("  python main.py --train")
        # or with logging:
        print("  python main.py --train 2>&1 | tee train_log.txt")


if __name__ == "__main__":
    main()
