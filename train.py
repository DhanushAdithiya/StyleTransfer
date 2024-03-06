import torch
from utils import load_image, convert_tensor, save_image
from model import get_style, load_model
from tqdm import tqdm
import config

def gram_matix(tensor):
    g = None
    _,d,h,w = tensor.size()
    tensor = tensor.view(d, h*w)
    g = torch.mm(tensor, tensor.t())
    return g


vgg = load_model()

content_img = load_image("content2.png")
style_img = load_image("style_img.png")

content_features = get_style(content_img,vgg)
style_features = get_style(style_img,vgg)
style_grams = {layer: gram_matix(style_features[layer]) for layer in style_features}


target_img = load_image("content2.png").requires_grad_(True)


optimizer = torch.optim.Adam([target_img], lr=0.003)
epoch = 5000

for i in tqdm(range(1, epoch + 1)):
    target_features = get_style(target_img, vgg)
    content_loss = torch.mean((content_features["conv4_2"] - content_features["conv4_2"])**2)


    style_loss = 0

    for layer in config.STYLE_WEIGHTS:
        target_feature = target_features[layer]
        _, d,h,w = target_feature.shape
        target_gram = gram_matix(target_feature)
        style_gram = style_grams[layer]
        layer_style_loss = config.STYLE_WEIGHTS[layer] * torch.mean((target_gram - style_gram) ** 2)

        style_loss += layer_style_loss / (d * h * w)

    total_loss = config.CONTENT_WEIGHT * content_loss + config.STYLE_WEIGHT * style_loss
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()


    img_tens = convert_tensor(target_img)
    save_image(img_tens, f"./generated/epoch_{i}.png")
