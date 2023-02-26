from research.distortion.arch_utils.base import ArchUtils


class ResNetUtils(ArchUtils):
    def __init__(self):
        super(ResNetUtils, self).__init__()

    def run_model_block(self, model, activation, block_name):
        if block_name == "stem":
            activation = model.backbone.stem(activation)
            activation = model.backbone.maxpool(activation)
        elif block_name == "decode":
            activation = model.decode_head([None, None, None, activation])
        else:
            res_layer_name, block_name = block_name.split("_")
            layer = getattr(model.backbone, res_layer_name)
            res_block = layer._modules[block_name]
            activation = res_block(activation)

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
        elif "stem" in layer_name:
            _, relu_name = layer_name.split("_")
            return model.backbone.stem._modules[relu_name]
        else:
            res_layer_name, block_name, relu_name = layer_name.split("_")

            layer = getattr(model.backbone, res_layer_name)
            res_block = layer._modules[block_name]
            return getattr(res_block, f"relu_{relu_name}")

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
        elif "stem" in layer_name:
            _, relu_name = layer_name.split("_")
            model.backbone.stem._modules[relu_name] = block_relu
        else:
            res_layer_name, block_name, relu_name = layer_name.split("_")

            layer = getattr(model.backbone, res_layer_name)
            res_block = layer._modules[block_name]
            setattr(res_block, f"relu_{relu_name}", block_relu)