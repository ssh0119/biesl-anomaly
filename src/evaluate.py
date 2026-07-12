import argparse

import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, classification_report, confusion_matrix, roc_auc_score
from torch.utils.data import DataLoader

from src import config
from src.data.dataset import Stage1Dataset, Stage2Dataset
from src.data.labels import STAGE2_CLASSES
from src.models.factory import build_model
from src.utils import to_device


@torch.no_grad()
def collect_predictions(model, loader, device):
    model.eval()
    all_probs, all_labels = [], []
    for inputs, labels in loader:
        inputs = to_device(inputs, device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model(inputs)
        probs = F.softmax(logits.float(), dim=1).cpu()
        all_probs.append(probs)
        all_labels.append(labels)
    return torch.cat(all_probs).numpy(), torch.cat(all_labels).numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=int, required=True, choices=[1, 2])
    parser.add_argument("--modality", type=str, required=True, choices=["ecg", "ppg", "both"])
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds_cls = Stage1Dataset if args.stage == 1 else Stage2Dataset
    test_ds = ds_cls(args.modality, "test", limit=args.limit)
    loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    ckpt_path = args.checkpoint or config.CHECKPOINT_DIR / f"stage{args.stage}_{args.modality}.pt"
    ckpt = torch.load(ckpt_path, map_location=device)
    model = build_model(args.stage, args.modality).to(device)
    model.load_state_dict(ckpt["model"])

    probs, labels = collect_predictions(model, loader, device)
    preds = probs.argmax(axis=1)

    print(f"stage={args.stage} modality={args.modality} n={len(labels):,}")
    if args.stage == 1:
        pos_probs = probs[:, 1]
        class_names = ["normal (SR)", "anomaly"]
        print(f"ROC-AUC: {roc_auc_score(labels, pos_probs):.4f}")
        print(f"PR-AUC:  {average_precision_score(labels, pos_probs):.4f}")
        print(classification_report(labels, preds, target_names=class_names, zero_division=0))
        print("confusion matrix (rows=true, cols=pred):")
        print(class_names)
        print(confusion_matrix(labels, preds, labels=[0, 1]))
    else:
        class_idx = list(range(len(STAGE2_CLASSES)))
        print(classification_report(labels, preds, labels=class_idx, target_names=STAGE2_CLASSES, zero_division=0))
        print("confusion matrix (rows=true, cols=pred):")
        print(STAGE2_CLASSES)
        print(confusion_matrix(labels, preds, labels=class_idx))


if __name__ == "__main__":
    main()
