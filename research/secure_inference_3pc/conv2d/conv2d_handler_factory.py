from research.secure_inference_3pc.conv2d.cuda_conv2d import Conv2DHandler as CudaConv2DHandler
from research.secure_inference_3pc.conv2d.numba_conv2d import Conv2DHandler as NumbaConv2DHandler


class Conv2DHandlerFactory:
    def __init__(self):
        pass

    def create(self, device):
        if 'cuda' in device:
            return CudaConv2DHandler(device)
        elif device == 'cpu':
            return NumbaConv2DHandler()
        else:
            raise ValueError('Invalid device')


conv2d_handler_factory = Conv2DHandlerFactory()