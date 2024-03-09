from pathlib import Path
import torch
from utils import load_image, convert_tensor, save_image, save_gif
from model import get_style, load_model
from tqdm import tqdm
import config
import argparse

def gram_matix(tensor):
    g = None
    _, d, h, w = tensor.size()
    tensor = tensor.view(d, h * w)
    g = torch.mm(tensor, tensor.t())
    return g


def main(
    content_path,
    style_path,
    target_path,
    epoch,
    save_interval,
    save_gif,
):
    if target_path is None:
        target_img  = load_image(content_path).requires_grad_(True)
    else:
        target_img = load_image(target_path).requires_grad_(True)

    content_img = load_image(content_path)
    style_img = load_image(style_path)


    vgg = load_model()

    optimizer = torch.optim.Adam([target_img], lr=0.003)
    content_features = get_style(content_img, vgg)
    style_features = get_style(style_img, vgg)
    style_grams = {layer: gram_matix(style_features[layer]) for layer in style_features}

    for i in tqdm(range(1, epoch + 1)):
        target_features = get_style(target_img, vgg)
        content_loss = torch.mean(
            (content_features["conv4_2"] - content_features["conv4_2"]) ** 2
        )

        style_loss = 0

        for layer in config.STYLE_WEIGHTS:
            target_feature = target_features[layer]
            _, d, h, w = target_feature.shape
            target_gram = gram_matix(target_feature)
            style_gram = style_grams[layer]
            layer_style_loss = config.STYLE_WEIGHTS[layer] * torch.mean(
                (target_gram - style_gram) ** 2
            )

            style_loss += layer_style_loss / (d * h * w)

        total_loss = (
            config.CONTENT_WEIGHT * content_loss + config.STYLE_WEIGHT * style_loss
        )
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        img_tens = convert_tensor(target_img)
        if i % save_interval == 0:
            save_image(img_tens, f"./generated/epoch_{i}.png")

        if i % epoch == 0 and save_gif:
            save_gif("./generated/", "testgif1")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--content", required=True, help="Path to content image", 
       # type=Path
    )
    parser.add_argument(
        "-s", "--style", required=True, help="Path to style image", 
        type=Path
    )
    parser.add_argument(
        "-t",
        "--target",
        required=False,
        help="Path to target image",
        #type=Path,
        default=None
    )
    parser.add_argument(
        "-e",
        "--epoch",
        required=False,
        help="Number of epochs",
        type=int,
        default=config.EPOCH,
    )
    parser.add_argument(
        "-i",
        "--interval",
        required=False,
        help="Interval to save images",
        type=int,
        default=config.SAVE_INTERVAL,
    )
    parser.add_argument(
        "-g",
        "--gif",
        required=False,
        help="Use this to save a gif of the images",
        action="store_true",
    )
    args = parser.parse_args()

    style_img = args.style
    content_img = args.content
    target_img = args.target
    epoch = args.epoch
    save_interval = args.interval
    save_gif = args.gif

    main(content_img, style_img, target_img, epoch, save_interval, save_gif)
