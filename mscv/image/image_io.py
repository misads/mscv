import numpy as np


def tensor2im(x, index=0):
    """Convert tensor to image.

    Args:
        x(torch.Tensor): input tensor, [n, c, h, w] float32 type.

    Returns:
        an image in shape of [h, w, c].

    """
    x = x.data.cpu().numpy()[index]
    x[x > 1] = 1
    x[x < 0] = 0
    x *= 255
    x = x.astype(np.uint8)
    x = x.transpose((1, 2, 0))

    return x

