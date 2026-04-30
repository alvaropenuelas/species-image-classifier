"""
Run inference on a single image or a folder of images.
Usage:
    python src/predict.py --model outputs/best_model.pt --classes outputs/classes.json --image path/to/img.jpg
"""

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from model import build_model
from train import _set_inference_mode

INFERENCE_TRANSFORMS = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def load_model(checkpoint_path: str, num_classes: int, device):
    model = build_model(num_classes=num_classes)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    _set_inference_mode(model)
    return model


def predict_image(model, image_path: str, class_names: list, device) -> tuple[str, float]:
    img = Image.open(image_path).convert("RGB")
    tensor = INFERENCE_TRANSFORMS(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)
        conf, idx = probs.max(1)
    return class_names[idx.item()], conf.item()


def main():
    parser = argparse.ArgumentParser(description="Species image classifier — inference")
    parser.add_argument("--model", required=True, help="Path to best_model.pt")
    parser.add_argument("--classes", required=True, help="Path to classes.json")
    parser.add_argument("--image", help="Single image path")
    parser.add_argument("--folder", help="Folder of images (runs on all .jpg/.png)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(args.classes) as f:
        class_names = json.load(f)

    model = load_model(args.model, num_classes=len(class_names), device=device)

    if args.image:
        species, confidence = predict_image(model, args.image, class_names, device)
        print(f"{Path(args.image).name}: {species} ({confidence:.2%})")

    elif args.folder:
        folder = Path(args.folder)
        images = list(folder.glob("*.jpg")) + list(folder.glob("*.png"))
        for img_path in sorted(images):
            species, confidence = predict_image(model, img_path, class_names, device)
            print(f"{img_path.name}: {species} ({confidence:.2%})")
    else:
        parser.error("Provide --image or --folder")


if __name__ == "__main__":
    main()
