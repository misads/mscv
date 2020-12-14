import torch.nn.functional as F


def padding_forward(times=8):
    def decorator(fn):
        def pad_image(*args, **kwargs):

            input = args[1]  # args = [self, input]
            _, _, pad_h, pad_w = input.size()
            if pad_h % times != 0 or pad_w % times != 0:
                h_pad_len = times - pad_h % times
                w_pad_len = times - pad_w % times

                input = F.pad(input, (0, w_pad_len, 0, h_pad_len), mode='reflect')

            args = list(args)
            args[1] = input
            args = tuple(args)

            output = fn(*args, **kwargs)

            output = output[:, :, :pad_h, :pad_w]
            return output

        return pad_image

    return decorator


