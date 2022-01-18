import os

from matplotlib import pyplot as plt


def model_path(model_file):
    module_path = os.path.dirname(__file__)
    return module_path + "../../models/" + model_file


def image_path(image_file):
    module_path = os.path.dirname(__file__)
    return module_path + "/images/" + image_file


def label_path(label_file):
    module_path = os.path.dirname(__file__)
    return module_path + "/labels/" + label_file


def font_path(font_file):
    module_path = os.path.dirname(__file__)
    return module_path + "/fonts/" + font_file


def load_labels(file_name, encoding='utf-8'):
    """Loads labels from file (with or without index numbers).
    Args:
      path: path to label file.
      encoding: label file encoding.
    Returns:
      Dictionary mapping indices to labels.
    """
    with open((label_path(file_name)), 'r', encoding=encoding) as f:
        lines = f.readlines()
        if not lines:
            return {}

        if lines[0].split(' ', maxsplit=1)[0].isdigit():
            pairs = [line.split(' ', maxsplit=1) for line in lines]
            return {int(index): label.strip() for index, label in pairs}
        else:
            return {index: line.strip() for index, line in enumerate(lines)}


def show_image(img):
    plt.figure()
    plt.grid(False)
    plt.imshow(img)
