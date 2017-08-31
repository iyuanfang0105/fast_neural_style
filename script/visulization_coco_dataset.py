import os

from src import tools


if __name__ == '__main__':
    images_path = '../data/COCO/val2014'
    images = tools.imread_from_folder(images_path, partion=0.01, shuffle=True)
    tools.show_images(images[::20], cols=3)
