from research.bReLU import BlockRelu


class ArchUtils:
    def __init__(self):
        pass

    def set_layers(self, model, layer_names_to_layers):
        for layer_name, layer in layer_names_to_layers.items():
            self.set_layer(model, layer_name, layer)

    def set_bReLU_layers(self, model, layer_name_to_block_sizes, block_relu_class=BlockRelu):
        layer_name_to_layers = {layer_name: block_relu_class(block_sizes=block_sizes)
                                for layer_name, block_sizes in layer_name_to_block_sizes.items()}
        self.set_layers(model, layer_name_to_layers)



