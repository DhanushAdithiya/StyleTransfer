from torchvision import models
import config



def load_model():
    vgg = models.vgg19(pretrained=True).features

    for param in vgg.parameters():
        param.requires_grad_(False)

    vgg.to(config.DEVICE)

    return vgg


def get_style(img, model):
    layers = config.LAYERS

    features = {}
    x = img.cuda()
    for name, layer in model._modules.items():
        x = layer(x)

        if name in layers:
            features[layers[name]] = x

    return features



