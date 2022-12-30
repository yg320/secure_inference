from research.distortion.utils import ArchUtils

class MobileNetV2_Utils(ArchUtils):
    def __init__(self):
        pass

    def run_model_block(self, model, activation, block_name):
        if block_name == "conv1":
            activation = model.backbone.conv1(activation)
        elif block_name == "decode":
            activation = model.decode_head([None, None, None, activation])
        else:
            res_layer_name, block_name = block_name.split("_")
            layer = getattr(model.backbone, res_layer_name)
            activation = layer[int(block_name)](activation)
        return activation

    def get_layer(self, model, layer_name):
        if "decode" in layer_name:
            if layer_name == "decode_0":
                return model.decode_head.image_pool[1].activate
            elif layer_name == "decode_5":
                return model.decode_head.bottleneck.activate
            else:
                _, relu_name = layer_name.split("_")
                relu_index = int(relu_name)
                assert relu_index in [1, 2, 3, 4]
                return model.decode_head.aspp_modules[relu_index - 1].activate
        elif layer_name == "conv1":
            return model.backbone.conv1.activate
        else:
            layer_name, inverted_residual_block, conv_module = layer_name.split("_")
            layer = getattr(model.backbone, layer_name)
            return layer[int(inverted_residual_block)].conv[int(conv_module)].activate

    def set_layer(self, model, layer_name, block_relu):
        if "decode" in layer_name:
            if layer_name == "decode_0":
                model.decode_head.image_pool[1].activate = block_relu
            elif layer_name == "decode_5":
                model.decode_head.bottleneck.activate = block_relu
            else:
                _, relu_name = layer_name.split("_")
                relu_index = int(relu_name)
                assert relu_index in [1, 2, 3, 4]
                model.decode_head.aspp_modules[relu_index - 1].activate = block_relu
        elif layer_name == "conv1":
            model.backbone.conv1.activate = block_relu
        else:
            layer_name, inverted_residual_block, conv_module = layer_name.split("_")
            layer = getattr(model.backbone, layer_name)
            layer[int(inverted_residual_block)].conv[int(conv_module)].activate = block_relu
