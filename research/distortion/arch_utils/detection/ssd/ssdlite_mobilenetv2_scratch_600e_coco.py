from research.distortion.utils import ArchUtils

class MobileNetV2_Utils(ArchUtils):
    def __init__(self):
        pass

    def run_model_block(self, model, activation, block_name):
        if block_name == "conv1":
            activation = model.backbone.conv1(activation)
        elif "extra_layer" in block_name:
            assert False
        else:
            res_layer_name, block_name = block_name.split("_")
            layer = getattr(model.backbone, res_layer_name)
            activation = layer[int(block_name)](activation)
        return activation

    def get_layer(self, model, layer_name):
        if "extra_layer" in layer_name:
            assert False
        elif layer_name == "conv1":
            return model.backbone.conv1.activate
        else:
            layer_name, inverted_residual_block, conv_module = layer_name.split("_")
            layer = getattr(model.backbone, layer_name)
            return layer[int(inverted_residual_block)].conv[int(conv_module)].activate

    def set_layer(self, model, layer_name, block_relu):
        if "extra_layer" in layer_name:
            assert False
        elif layer_name == "conv1":
            model.backbone.conv1.activate = block_relu
        else:
            layer_name, inverted_residual_block, conv_module = layer_name.split("_")
            layer = getattr(model.backbone, layer_name)
            layer[int(inverted_residual_block)].conv[int(conv_module)].activate = block_relu
