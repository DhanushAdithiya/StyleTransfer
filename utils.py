from PIL import Image
from torchvision import transforms
import numpy as np

def load_image(img_path, max_size=256):
    img = Image.open(img_path).convert("RGB")

    size = img.size[0]
    if size > max_size:
        size = max_size
    
    transformation = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    image = transformation(img)[:3, :, :].unsqueeze(0)
    
    return image    


def convert_tensor(tensor):
    image = tensor.to("cpu").clone().detach()
    image = image.numpy().squeeze()
    image = image.transpose(1,2,0)
    image = image * np.array((0.5,0.5,0.5)) + np.array((0.5,0.5,0.5))
    image = image * 255  
    image = image.astype(np.uint8)
    return image

def save_image(arr, fp):
    image = Image.fromarray(arr)
    image.save(fp)
