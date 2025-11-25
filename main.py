"""
Main entry point for Heart Sound Classification
Run this script to train and evaluate the model
"""
import argparse
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train import train_model
from config import EPOCHS, BATCH_SIZE


def main():
    parser = argparse.ArgumentParser(description='Heart Sound Classification')
    
    parser.add_argument(
        '--model', 
        type=str, 
        default='cnn_attention',
        choices=['cnn', 'cnn_attention', 'resnet'],
        help='Model architecture to use (default: cnn_attention)'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=EPOCHS,
        help=f'Number of training epochs (default: {EPOCHS})'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=BATCH_SIZE,
        help=f'Batch size (default: {BATCH_SIZE})'
    )
    
    parser.add_argument(
        '--no-augment',
        action='store_true',
        help='Disable data augmentation'
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("     HEART SOUND CLASSIFICATION - PASCAL Dataset")
    print("="*60)
    print(f"  Model: {args.model}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Augmentation: {not args.no_augment}")
    print("="*60 + "\n")
    
    # Train model
    model, history, metrics = train_model(
        model_type=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        augment=not args.no_augment
    )
    
    print("\n" + "="*60)
    print("                   TRAINING SUMMARY")
    print("="*60)
    print(f"  Accuracy:  {metrics['accuracy']*100:.2f}%")
    print(f"  Precision: {metrics['precision']*100:.2f}%")
    print(f"  Recall:    {metrics['recall']*100:.2f}%")
    print(f"  F1 Score:  {metrics['f1_score']*100:.2f}%")
    print("="*60)
    print("\nModel and results saved in 'models/' directory")
    

if __name__ == "__main__":
    main()
