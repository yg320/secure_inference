from research.distortion.utils import ArchUtils

class Utils(ArchUtils):
    def __init__(self):
        self.relu_layer_to_feature_index = {
            "relu_0": 1,
            "relu_1": 3,
            "relu_2": 6,
            "relu_3": 8,
            "relu_4": 11,
            "relu_5": 13,
            "relu_6": 15,
            "relu_7": 18,
            "relu_8": 20,
            "relu_9": 22,
            "relu_10": 25,
            "relu_11": 27,
            "relu_12": 29,
            "relu_13": 32,
            "relu_14": 34
        }


    def run_model_block(self, model, activation, block_name):

        if block_name == "block_0":
            features = model.features[0:2]
        if block_name == "block_1":
            features = model.features[2:4]
        if block_name == "block_2":
            features = model.features[4:7]
        if block_name == "block_3":
            features = model.features[7:9]
        if block_name == "block_4":
            features = model.features[9:12]
        if block_name == "block_5":
            features = model.features[12:14]
        if block_name == "block_6":
            features = model.features[14:16]
        if block_name == "block_7":
            features = model.features[16:19]
        if block_name == "block_8":
            features = model.features[19:21]
        if block_name == "block_9":
            features = model.features[21:23]
        if block_name == "block_10":
            features = model.features[23:26]
        if block_name == "block_11":
            features = model.features[26:28]
        if block_name == "block_12":
            features = model.features[28:30]
        if block_name == "block_13":
            features = model.features[30:33]
        if block_name == "block_14":
            features = model.features[33:35]

        for feature in features:
            activation = feature(activation)

        return activation

    def get_layer(self, model, layer_name):
        return model.features[self.relu_layer_to_feature_index[layer_name]]

    def set_layer(self, model, layer_name, block_relu):
        model.features[self.relu_layer_to_feature_index[layer_name]] = block_relu
