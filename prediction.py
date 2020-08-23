from torchvision import models
from inference import transform_image
import json

# Make sure to pass `pretrained` as `True` to use the pretrained weights:
model = models.resnet34(pretrained=True)
# Since we are using our model only for inference, switch to `eval` mode:
model.eval()

imagenet_class_index = json.load(open('static/imagenet_class_index.json'))


def get_prediction(image_bytes):
    tensor = transform_image(image_bytes=image_bytes)
    outputs = model.forward(tensor)
    _, y_hat = outputs.max(1)
    predicted_idx = str(y_hat.item())
    return imagenet_class_index[predicted_idx]
