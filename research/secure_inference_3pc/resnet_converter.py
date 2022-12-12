import torch

def securify_model(model, build_secure_conv, build_secure_relu, crypto_assets, network_assets):
    model.backbone.stem[0] = build_secure_conv(crypto_assets, network_assets, model.backbone.stem[0], model.backbone.stem[1])
    model.backbone.stem[1] = torch.nn.Identity()
    model.backbone.stem[2] = build_secure_relu(crypto_assets=crypto_assets, network_assets=network_assets)

    model.backbone.stem[3] = build_secure_conv(crypto_assets, network_assets, model.backbone.stem[3], model.backbone.stem[4])
    model.backbone.stem[4] = torch.nn.Identity()
    model.backbone.stem[5] = build_secure_relu(crypto_assets=crypto_assets, network_assets=network_assets)

    model.backbone.stem[6] = build_secure_conv(crypto_assets, network_assets, model.backbone.stem[6], model.backbone.stem[7])
    model.backbone.stem[7] = torch.nn.Identity()
    model.backbone.stem[8] = build_secure_relu(crypto_assets=crypto_assets, network_assets=network_assets)

    for layer in [1, 2, 3, 4]:
        for block in [0, 1]:
            cur_res_layer = getattr(model.backbone, f"layer{layer}")
            cur_res_layer[block].conv1 = build_secure_conv(crypto_assets, network_assets, cur_res_layer[block].conv1, cur_res_layer[block].bn1)
            cur_res_layer[block].bn1 = torch.nn.Identity()
            cur_res_layer[block].relu_1 = build_secure_relu(crypto_assets=crypto_assets, network_assets=network_assets)

            cur_res_layer[block].conv2 = build_secure_conv(crypto_assets, network_assets, cur_res_layer[block].conv2, cur_res_layer[block].bn2)
            cur_res_layer[block].bn2 = torch.nn.Identity()
            cur_res_layer[block].relu_2 = build_secure_relu(crypto_assets=crypto_assets, network_assets=network_assets)

            if cur_res_layer[block].downsample:
                cur_res_layer[block].downsample = build_secure_conv(crypto_assets, network_assets, cur_res_layer[block].downsample[0], cur_res_layer[block].downsample[1])

    model.decode_head.image_pool[1].conv = build_secure_conv(crypto_assets, network_assets, model.decode_head.image_pool[1].conv, model.decode_head.image_pool[1].bn)
    model.decode_head.image_pool[1].bn = torch.nn.Identity()
    model.decode_head.image_pool[1].activate = build_secure_relu(crypto_assets=crypto_assets, network_assets=network_assets)

    for i in range(4):
        model.decode_head.aspp_modules[i].conv = build_secure_conv(crypto_assets, network_assets, model.decode_head.aspp_modules[i].conv, model.decode_head.aspp_modules[i].bn)
        model.decode_head.aspp_modules[i].bn = torch.nn.Identity()
        model.decode_head.aspp_modules[i].activate = build_secure_relu(crypto_assets=crypto_assets, network_assets=network_assets)

    model.decode_head.bottleneck.conv = build_secure_conv(crypto_assets, network_assets, model.decode_head.bottleneck.conv, model.decode_head.bottleneck.bn)
    model.decode_head.bottleneck.bn = torch.nn.Identity()
    model.decode_head.bottleneck.activate = build_secure_relu(crypto_assets=crypto_assets, network_assets=network_assets)

    model.decode_head.conv_seg = build_secure_conv(crypto_assets, network_assets, model.decode_head.conv_seg, None)
    model.decode_head.image_pool[0].forward = lambda x: x.sum(dim=[2, 3], keepdims=True) // (x.shape[2] * x.shape[3])


