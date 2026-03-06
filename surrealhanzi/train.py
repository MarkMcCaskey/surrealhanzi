"""Train the IDS transformer and generate novel sequences."""

import argparse
import json
import os
import sys
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from .ids_dataset import IDSVocab, load_ids_sequences, prepare_dataset
from .ids_parser import parse_ids, IDSParseError
from .transformer import IDSTransformer

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")


def train(
    d_model: int = 128,
    n_heads: int = 4,
    n_layers: int = 2,
    max_len: int = 16,
    batch_size: int = 128,
    epochs: int = 50,
    lr: float = 3e-4,
    device: str = "cpu",
) -> None:
    """Train the IDS transformer model."""
    print("Loading IDS sequences...")
    sequences = load_ids_sequences()
    print(f"  {len(sequences)} sequences loaded")

    vocab = IDSVocab()
    vocab.build(sequences)
    print(f"  Vocab size: {vocab.size}")

    train_data, val_data = prepare_dataset(sequences, vocab, max_len=max_len)
    print(f"  Train: {len(train_data)}, Val: {len(val_data)}")

    train_tensor = torch.tensor(train_data, dtype=torch.long)
    val_tensor = torch.tensor(val_data, dtype=torch.long)

    train_loader = DataLoader(
        TensorDataset(train_tensor), batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(val_tensor), batch_size=batch_size
    )

    model = IDSTransformer(
        vocab_size=vocab.size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        max_len=max_len,
        pad_id=vocab.pad_id,
    ).to(device)

    print(f"  Model params: {model.param_count():,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float("inf")
    os.makedirs(MODEL_DIR, exist_ok=True)

    for epoch in range(1, epochs + 1):
        # Training
        model.train()
        total_loss = 0.0
        n_batches = 0

        for (batch,) in train_loader:
            batch = batch.to(device)
            # Input: all tokens except last; Target: all tokens except first
            x = batch[:, :-1]
            y = batch[:, 1:]

            logits = model(x)
            loss = F.cross_entropy(
                logits.reshape(-1, vocab.size),
                y.reshape(-1),
                ignore_index=vocab.pad_id,
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_train_loss = total_loss / n_batches

        # Validation
        model.eval()
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for (batch,) in val_loader:
                batch = batch.to(device)
                x = batch[:, :-1]
                y = batch[:, 1:]
                logits = model(x)
                loss = F.cross_entropy(
                    logits.reshape(-1, vocab.size),
                    y.reshape(-1),
                    ignore_index=vocab.pad_id,
                )
                val_loss += loss.item()
                val_batches += 1

        avg_val_loss = val_loss / val_batches
        scheduler.step()

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}: train_loss={avg_train_loss:.4f} val_loss={avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_path = os.path.join(MODEL_DIR, "ids_transformer.pt")
            torch.save({
                "model_state": model.state_dict(),
                "vocab": {
                    "token_to_id": vocab.token_to_id,
                    "id_to_token": {int(k): v for k, v in vocab.id_to_token.items()},
                },
                "config": {
                    "d_model": d_model,
                    "n_heads": n_heads,
                    "n_layers": n_layers,
                    "max_len": max_len,
                    "vocab_size": vocab.size,
                    "pad_id": vocab.pad_id,
                },
            }, save_path)

    print(f"\n  Best val_loss: {best_val_loss:.4f}")
    print(f"  Model saved to {save_path}")

    # Generate samples
    print("\nGenerating samples...")
    _generate_samples(model, vocab, n=20, temperature=1.0, top_k=50)


def _generate_samples(
    model: IDSTransformer,
    vocab: IDSVocab,
    n: int = 10,
    temperature: float = 1.0,
    top_k: int = 50,
) -> list[str]:
    """Generate and print novel IDS sequences."""
    model.eval()
    valid = []
    attempts = 0
    max_attempts = n * 10

    while len(valid) < n and attempts < max_attempts:
        attempts += 1
        token_ids = model.generate(
            bos_id=vocab.bos_id,
            eos_id=vocab.eos_id,
            temperature=temperature,
            top_k=top_k,
        )
        ids_str = vocab.decode(token_ids)
        if not ids_str:
            continue

        # Validate: must parse as valid IDS
        try:
            tree = parse_ids(ids_str)
            valid.append(ids_str)
        except IDSParseError:
            continue

    for i, ids_str in enumerate(valid):
        print(f"  {i+1:2d}. {ids_str}")

    print(f"\n  Generated {len(valid)} valid sequences in {attempts} attempts "
          f"({len(valid)/max(attempts,1)*100:.0f}% valid)")
    return valid


def generate(
    n: int = 10,
    temperature: float = 1.0,
    top_k: int = 50,
) -> list[str]:
    """Load saved model and generate sequences."""
    save_path = os.path.join(MODEL_DIR, "ids_transformer.pt")
    if not os.path.exists(save_path):
        print(f"No model found at {save_path}. Run training first.", file=sys.stderr)
        sys.exit(1)

    checkpoint = torch.load(save_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]

    vocab = IDSVocab()
    vocab.token_to_id = checkpoint["vocab"]["token_to_id"]
    vocab.id_to_token = {int(k): v for k, v in checkpoint["vocab"]["id_to_token"].items()}

    model = IDSTransformer(
        vocab_size=config["vocab_size"],
        d_model=config["d_model"],
        n_heads=config["n_heads"],
        n_layers=config["n_layers"],
        max_len=config["max_len"],
        pad_id=config["pad_id"],
    )
    model.load_state_dict(checkpoint["model_state"])

    return _generate_samples(model, vocab, n=n, temperature=temperature, top_k=top_k)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train/generate IDS sequences")
    sub = parser.add_subparsers(dest="command")

    train_p = sub.add_parser("train", help="Train the model")
    train_p.add_argument("--epochs", type=int, default=50)
    train_p.add_argument("--d-model", type=int, default=128)
    train_p.add_argument("--n-layers", type=int, default=2)
    train_p.add_argument("--n-heads", type=int, default=4)
    train_p.add_argument("--batch-size", type=int, default=128)
    train_p.add_argument("--lr", type=float, default=3e-4)

    gen_p = sub.add_parser("generate", help="Generate novel IDS sequences")
    gen_p.add_argument("-n", type=int, default=20)
    gen_p.add_argument("--temperature", type=float, default=1.0)
    gen_p.add_argument("--top-k", type=int, default=50)

    args = parser.parse_args()

    if args.command == "train":
        train(
            d_model=args.d_model,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
        )
    elif args.command == "generate":
        generate(n=args.n, temperature=args.temperature, top_k=args.top_k)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
