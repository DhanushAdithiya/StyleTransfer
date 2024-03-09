# <div align="center">Style Trasfer</div>
### <div align="center"><a href="https://arxiv.org/abs/1508.06576">A Neural Algorithm of Artistic Style</a> Implementation </div>
</br>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

<div style="text-align: center; width: 100%; display:flex; justify-content: center">
  <img style="padding: 10px; width:40%;" src="https://github.com/DhanushAdithiya/StyleTransfer/assets/84760124/fde6c7e1-9eb5-4557-8196-7dd6c0d72d66" alt="Image 1">
  <img style="padding: 10px;width:40%;" src="https://github.com/DhanushAdithiya/StyleTransfer/assets/84760124/dd4ce706-569d-4ff1-83b2-0a05fdfcbf0b" alt="Image 2">
</div>



This project implements neural style transfer, allowing you to apply the artistic style of one image (style image) to the content of another image (content image). The model is based on the paper titled <a href="https://arxiv.org/abs/1508.06576">A Neural Algorithm of Artistic Style</a> by *Leon A. Gatys, Alexander S. Ecker, Matthias Bethge*

Features:
* Leverages a pretrained VGG19 CNN model. Thus no training requried
* Allows you to specifiy your own style image and content image 
* Generate an animation (GIF) showcasing the style transfer progression (optional).
* Configurable model parameters

### Built With

This section should list any major frameworks/libraries used to bootstrap your project. Leave any add-ons/plugins for the acknowledgements section. Here are a few examples.

* [![Python][Python]][Python-url]
* [![Anaconda][Anaconda]][Anaconda-url]
* ![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)



<!-- GETTING STARTED -->
## Getting Started

To get a local copy up and running follow these simple example steps.

### Prerequisites

This is an example of how to list things you need to use the software and how to install them.

* <a href="https://www.python.org/downloads/release/python-390/">Python 3.9</a>
* <a href="https://docs.anaconda.com/free/anaconda/install/linux/">Conda Enviroment</a>
* Optional: <a href="https://pytorch.org/get-started/locally/">GPU Enabled Env </a>


### Installation

_Below is an example of how you can instruct your audience on installing and setting up your app. This template doesn't rely on any external dependencies or services._

1. Clone the repo
   ```sh
   git clone https://github.com/DhanushAdithiya/StyleTransfer
   ```
2. Activate Conda Environment
    ```sh
    conda activate [YOUR_ENV]
    ```
3. Install Python packages
   ```sh
   pip install -r requirements.txt 
   ```
4. Create a folder called `Generated`
   ```sh
   mkdir Generated
   ```

<!-- USAGE EXAMPLES -->
## Usage

Once you have completed all the installation procedured now you are ready to run the program. This project uses Python's argparse module for command-line arguments.

```sh
python3 train.py -c <content_image_url> -s <style_image_url> [OPTIONS]
```

### Required Arguments:

* `-c, --content`: Path to the content image
* `-s, --style`: Path to the style image 

### Optional Arguments:

* `-t, --target`: Path to the target image (defaults to the content image).
* `-e, --epoch`: Number of training epochs (defaults to a value in config.py).
* `-i, --interval`: Interval (in epochs) for saving intermediate results (defaults to a value in config.py).
* `-g, --gif`: Create a GIF animation of the style transfer process (boolean flag).
#### Example:

```sh
python neural_style_transfer.py -c path/to/content.jpg -s path/to/style.jpg -e 20 -i 5 -g
```

<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.

<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

Use this space to list resources you find helpful and would like to give credit to. I've included a few of my favorites to kick things off!

* [A Neural Algorithm of Artistic Style](https://arxiv.org/abs/1508.06576)
* [Neural Style Transfer with Deep VGG model](https://medium.com/@mirzezadeh.elvin/neural-style-transfer-with-deep-vgg-model-26b11ea06b7e)
* [VGG19 Model](https://arxiv.org/abs/1409.1556v6)

<!-- MARKDOWN LINKS & IMAGES -->
[python]:https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54
[Python-url]: https://www.python.org/
[Anaconda]: https://img.shields.io/badge/Anaconda-%2344A833.svg?style=for-the-badge&logo=anaconda&logoColor=white
[Anaconda-url]: https://www.anaconda.com/download
