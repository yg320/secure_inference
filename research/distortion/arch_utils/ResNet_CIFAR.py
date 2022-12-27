from research.distortion.utils import ArchUtils


class ResNet_CIFAR_Utils(ArchUtils):
    def __init__(self):
        super(ResNet_CIFAR_Utils, self).__init__()

    def run_model_block(self, model, activation, block_name):
        if block_name == "stem":
            activation = model.backbone.conv1(activation)
            activation = model.backbone.norm1(activation)
            activation = model.backbone.relu(activation)
        else:
            res_layer_name, block_name = block_name.split("_")
            layer = getattr(model.backbone, res_layer_name)
            res_block = layer._modules[block_name]
            activation = res_block(activation)

        return activation

    def get_layer(self, model, layer_name):
        if layer_name == "stem":
            return model.backbone.relu
        else:
            res_layer_name, block_name, relu_name = layer_name.split("_")

            layer = getattr(model.backbone, res_layer_name)
            res_block = layer._modules[block_name]
            return getattr(res_block, f"relu_{relu_name}")

    def set_layer(self, model, layer_name, block_relu):
        if layer_name == "stem":
            model.backbone.relu = block_relu
        else:
            res_layer_name, block_name, relu_name = layer_name.split("_")

            layer = getattr(model.backbone, res_layer_name)
            res_block = layer._modules[block_name]
            setattr(res_block, f"relu_{relu_name}", block_relu)
