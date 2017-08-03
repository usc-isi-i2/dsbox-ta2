__path__ = __import__('pkgutil').extend_path(__path__, __name__)

from .net_image_feature import ResNet50ImageFeature, Vgg16ImageFeature
from .image_feature import PcaImageFeature, ColorRegionImageFeature

DSBOX_PRIMITIVES = [
    ResNet50ImageFeature, Vgg16ImageFeature, PcaImageFeature, ColorRegionImageFeature]
