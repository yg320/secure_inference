from research.secure_inference_3pc.const import TRUNC, NUM_BITS
import torch
import numpy as np
from research.secure_inference_3pc.base import SecureModule


class PRFFetcherModule(SecureModule):
    def __init__(self, **kwargs):
        super(PRFFetcherModule, self).__init__(**kwargs)


