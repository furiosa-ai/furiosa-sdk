import collections

import numpy as np
from . import font_path
from PIL import ImageFont, Image, ImageColor


def draw_objects(draw, objs, labels):
  """Draws the bounding box and label for each object."""
  colors = list(ImageColor.colormap.values())

  for obj in objs:
    bbox = obj.bbox
    label_name = labels.get(obj.id, obj.id)
    text = '%s: %d%%' % (label_name, int(obj.score * 100))
    color = colors[hash(label_name) % len(colors)]
    draw.rectangle([(bbox.xmin, bbox.ymin), (bbox.xmax, bbox.ymax)], outline=color, width=5)
    font = ImageFont.truetype(font_path('DejaVuSansMono.ttf'), 20)

    left, top, right, bottom = font.getbbox(text)
    text_width, text_height = right - left, bottom - top
    draw.rectangle((bbox.xmin, bbox.ymin, bbox.xmin + text_width + 20, bbox.ymin + text_height + 20), fill=color)
    draw.text((bbox.xmin + 10, bbox.ymin + 10),
              text, font=font, fill='black')


def input_size(session):
  """Returns input image size as (width, height) tuple."""
  _, height, width, _ = session.input(0).shape
  return height, width


def input_tensor(session):
  """Returns input tensor view as numpy array of shape (height, width, 3)."""
  return session.input(0)


def resize_image(session, size, resize):
  """Copies a resized and properly zero-padded image to the input tensor.
  Args:
    interpreter: Interpreter object.
    size: original image size as (width, height) tuple.
    resize: a function that takes a (width, height) tuple, and returns an RGB
      image resized to those dimensions.
  Returns:
    Actual resize ratio, which should be passed to `get_output` function.
  """
  width, height = input_size(session)
  w, h = size
  scale = min(width / w, height / h)
  w, h = int(w * scale), int(h * scale)
  tensor = input_tensor(session)
  _, _, _, channel = tensor.shape
  resized_img = np.reshape(resize((w, h)), (h, w, channel))
  return scale, scale, resized_img


def output_tensor(outputs, i):
  """Returns output tensor view."""
  tensor = outputs[i].numpy()
  return np.squeeze(tensor)


class BBox(collections.namedtuple('BBox', ['xmin', 'ymin', 'xmax', 'ymax'])):
  """Bounding box.
  Represents a rectangle which sides are either vertical or horizontal, parallel
  to the x or y axis.
  """
  __slots__ = ()

  @property
  def width(self):
    """Returns bounding box width."""
    return self.xmax - self.xmin

  @property
  def height(self):
    """Returns bounding box height."""
    return self.ymax - self.ymin

  @property
  def area(self):
    """Returns bound box area."""
    return self.width * self.height

  @property
  def valid(self):
    """Returns whether bounding box is valid or not.
    Valid bounding box has xmin <= xmax and ymin <= ymax which is equivalent to
    width >= 0 and height >= 0.
    """
    return self.width >= 0 and self.height >= 0

  def scale(self, sx, sy):
    """Returns scaled bounding box."""
    return BBox(xmin=sx * self.xmin,
                ymin=sy * self.ymin,
                xmax=sx * self.xmax,
                ymax=sy * self.ymax)

  def translate(self, dx, dy):
    """Returns translated bounding box."""
    return BBox(xmin=dx + self.xmin,
                ymin=dy + self.ymin,
                xmax=dx + self.xmax,
                ymax=dy + self.ymax)

  def map(self, f):
    """Returns bounding box modified by applying f for each coordinate."""
    return BBox(xmin=f(self.xmin),
                ymin=f(self.ymin),
                xmax=f(self.xmax),
                ymax=f(self.ymax))

  @staticmethod
  def intersect(a, b):
    """Returns the intersection of two bounding boxes (may be invalid)."""
    return BBox(xmin=max(a.xmin, b.xmin),
                ymin=max(a.ymin, b.ymin),
                xmax=min(a.xmax, b.xmax),
                ymax=min(a.ymax, b.ymax))

  @staticmethod
  def union(a, b):
    """Returns the union of two bounding boxes (always valid)."""
    return BBox(xmin=min(a.xmin, b.xmin),
                ymin=min(a.ymin, b.ymin),
                xmax=max(a.xmax, b.xmax),
                ymax=max(a.ymax, b.ymax))

  @staticmethod
  def iou(a, b):
    """Returns intersection-over-union value."""
    intersection = BBox.intersect(a, b)
    if not intersection.valid:
      return 0.0
    area = intersection.area
    return area / (a.area + b.area - area)


Object = collections.namedtuple('Object', ['id', 'score', 'bbox'])


def get_output(session, outputs, score_threshold, image_scale=(1.0, 1.0)):
  """Returns list of detected objects."""
  boxes = output_tensor(outputs, 0)
  class_ids = output_tensor(outputs, 1)
  scores = output_tensor(outputs, 2)
  count = int(output_tensor(outputs, 3))

  _, height, width, _ = session.input(0).shape
  image_scale_x, image_scale_y = image_scale
  sx, sy = width / image_scale_x, height / image_scale_y

  def make(i):
    ymin, xmin, ymax, xmax = boxes[i]
    return Object(
        id=int(class_ids[i]),
        score=float(scores[i]),
        bbox=BBox(xmin=xmin,
                  ymin=ymin,
                  xmax=xmax,
                  ymax=ymax).scale(sx, sy).map(int))

  return [make(i) for i in range(count) if scores[i] >= score_threshold]


def get_padded_image(sess, image):
    w_scale, h_scale, resized_image = resize_image(sess, image.size, lambda size: image.resize(size, Image.Resampling.LANCZOS))
    data = np.zeros((300,300, 3), np.uint8)
    data[:resized_image.shape[0],:resized_image.shape[1],:resized_image.shape[2]] = resized_image
    return w_scale, h_scale, np.reshape(data, (1, 300, 300, 3))


def print_objects(labels, objects):
    for obj in objects:
        print("{}: {}% (box: {}, {}, {}, {})"
              .format(labels.get(obj.id, obj.id), int(obj.score * 100),
                      obj.bbox.xmin, obj.bbox.ymin, obj.bbox.xmax, obj.bbox.ymax))
