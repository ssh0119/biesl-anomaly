import argparse
import time

import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from src import config
from src.data.dataset import Stage1Dataset, Stage2Dataset
from src.models.factory import build_model, num_classes_for_stage
from src.utils import compute_class_weights, to_device


def build_datasets(stage: int, modality: str, limit: int | None):
    cls = Stage1Dataset if stage == 1 else Stage2Dataset
    train_ds = cls(modality, "train", limit=limit)
    val_ds = cls(modality, "val", limit=None if limit is None else max(limit // 5, 100))
    return train_ds, val_ds


@torch.no_grad()
def evaluate(model, loader, device, autocast_dtype):
    model.eval()
    all_preds, all_labels = [], []
    for inputs, labels in loader:
        inputs = to_device(inputs, device)
        with torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=device.type == "cuda"):
            logits = model(inputs)
        all_preds.append(logits.argmax(dim=1).cpu())
        all_labels.append(labels)
    preds = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()
    return f1_score(labels, preds, average="macro", zero_division=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=int, required=True, choices=[1, 2])
    parser.add_argument("--modality", type=str, required=True, choices=["ecg", "ppg", "both"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None, help="subsample N rows per split (for smoke tests)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    autocast_dtype = torch.bfloat16

    train_ds, val_ds = build_datasets(args.stage, args.modality, args.limit)
    print(f"train={len(train_ds):,} val={len(val_ds):,} stage={args.stage} modality={args.modality}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
        pin_memory=True, drop_last=True, persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
        pin_memory=True, persistent_workers=args.num_workers > 0,
    )

    model = build_model(args.stage, args.modality).to(device)
    num_classes = num_classes_for_stage(args.stage)

    label_col = "stage1_label" if args.stage == 1 else "stage2_idx"
    class_weights = compute_class_weights(train_ds.df[label_col].to_numpy(), num_classes).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = config.CHECKPOINT_DIR / f"stage{args.stage}_{args.modality}.pt"
    best_f1 = -1.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        running_loss = 0.0
        for step, (inputs, labels) in enumerate(train_loader, 1):
            inputs = to_device(inputs, device)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=device.type == "cuda"):
                logits = model(inputs)
                loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if step % 50 == 0:
                print(f"epoch {epoch} step {step}/{len(train_loader)} loss={running_loss / step:.4f}")

        val_f1 = evaluate(model, val_loader, device, autocast_dtype)
        dt = time.time() - t0
        print(f"epoch {epoch} done in {dt:.1f}s | train_loss={running_loss / len(train_loader):.4f} | val_macro_f1={val_f1:.4f}")

        torch.save({"model": model.state_dict(), "args": vars(args)}, config.CHECKPOINT_DIR / f"stage{args.stage}_{args.modality}_last.pt")
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save({"model": model.state_dict(), "args": vars(args)}, ckpt_path)
            print(f"  saved new best checkpoint -> {ckpt_path}")


if __name__ == "__main__":
    main()
