import math
import torch
import open_clip
from PIL import Image

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMAGE_WEIGHT = 0.6
TEXT_WEIGHT  = 0.4
SIGMOID_SCALE = 5.0

model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32",
    pretrained="openai"
)
model = model.to(DEVICE)
model.eval()

tokenizer = open_clip.get_tokenizer("ViT-B-32")

GNOME_PROMPTS = [
    "GNOME desktop environment screenshot",
    "GNOME Shell activities overview",
    "GNOME overview mode",
    "Linux GNOME desktop",
    "GNOME desktop with no windows",
    "GNOME fresh install desktop",
]

NON_GNOME_PROMPTS = [
    "KDE Plasma desktop screenshot",
    "Windows desktop screenshot",
    "macOS desktop screenshot",
    "Linux Cinnamon desktop",
    "i3 window manager desktop",
    "sway window manager desktop",
    "tiling window manager linux",
    "minimal linux desktop",
]

with torch.no_grad():
    gnome_tokens = tokenizer(GNOME_PROMPTS).to(DEVICE)
    not_tokens   = tokenizer(NON_GNOME_PROMPTS).to(DEVICE)

    gnome_features = model.encode_text(gnome_tokens)
    not_features   = model.encode_text(not_tokens)

    gnome_features /= gnome_features.norm(dim=-1, keepdim=True)
    not_features   /= not_features.norm(dim=-1, keepdim=True)

STRONG_KEYWORDS = {
    "gnome": 2.0,
    "gnome-shell": 2.0,
    "adwaita": 1.5,
    "gtk": 1.0,
    "linux mint": 1.5,
    "mint": 1.0,
    "cinnamon": 1.5,
}

CONFUSION_PHRASES = {
    "why is it like this": 1.5,
    "desktop looks different": 1.5,
    "what happened to my desktop": 2.0,
    "everything disappeared": 2.0,
    "desktop is gone": 2.0,
    "tablet ui": 2.0,
    "looks like a tablet": 2.0,
    "weird ui": 1.5,
    "big icons": 1.0,
    "search bar at the top": 1.5,
    "activities overview": 2.0,
    "closed my laptop": 1.0,
    "after restarting": 1.0,
}

NEGATIVE_PHRASES = {
    "enjoying": -1.0,
    "rice": -1.5,
    "unixporn": -2.0,
    "flex": -1.0,
    "tiling wm": -1.5,
}

def image_gnome_score(image_path: str) -> float:
    image = Image.open(image_path).convert("RGB")
    image_input = preprocess(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        image_features = model.encode_image(image_input)
        image_features /= image_features.norm(dim=-1, keepdim=True)

        gnome_sim = (image_features @ gnome_features.T).max().item()
        not_sim   = (image_features @ not_features.T).max().item()

        diff = gnome_sim - not_sim

    return 1 / (1 + math.exp(-SIGMOID_SCALE * diff))


def text_gnome_score(title: str, body: str) -> float:
    text = f"{title} {body}".lower()
    score = 0.0

    for k, w in STRONG_KEYWORDS.items():
        if k in text:
            score += w

    for p, w in CONFUSION_PHRASES.items():
        if p in text:
            score += w

    for p, w in NEGATIVE_PHRASES.items():
        if p in text:
            score += w

    return max(0.0, min(score / 5.0, 1.0))


def combine_scores(img: float, txt: float) -> float:
    score = IMAGE_WEIGHT * img + TEXT_WEIGHT * txt

    if txt == 0.0:
        score *= 0.7

    return score


def severity(score: float) -> str:
    if score < 0.1:
        return "medium"
    elif score < 0.3:
        return "high"
    elif score < 0.7:
        return "severe"
    else:
        return "catastrophe"

if __name__ == "__main__":
    image_path = input('filename')
    reddit_title = (
        input("title ")
    )
    reddit_body = (
        input('body ')
    )

    img_score = image_gnome_score(image_path)
    txt_score = text_gnome_score(reddit_title, reddit_body)
    final_score = combine_scores(img_score, txt_score)

    print(f"Image score: {img_score:.3f}")
    print(f"Text score:  {txt_score:.3f}")
    print(f"Final score: {final_score:.3f}")
    print(f"Severity:    {severity(final_score)}")
