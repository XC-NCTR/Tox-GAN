from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn as nn

import adf


def spaced_elements(array, num_elems=4):
    return [x[len(x) // 2] for x in np.array_split(np.array(array), num_elems)]


class AbstractPlayerIterator(ABC):

    def __init__(self, inputs, random=False):
        self._assert_input_compatibility(inputs)
        self.input_shape = inputs.shape[1:]
        self.random = random
        self.n_players = self._get_number_of_players_from_shape()
        self.permutation = np.array(range(self.n_players), 'int32')
        if random is True:
            self.permutation = np.random.permutation(self.permutation)
        self.i = 0
        self.kn = self.n_players
        self.ks = spaced_elements(range(self.n_players), self.kn)

    def set_n_steps(self, steps):
        self.kn = steps
        self.ks = spaced_elements(range(self.n_players), self.kn)

    def get_number_of_players(self):
        return self.n_players

    def get_explanation_shape(self):
        return self.input_shape

    def get_coalition_size(self):
        return 1

    def get_steps_list(self):
        return self.ks

    def __iter__(self):
        self.i = 0
        return self

    def __next__(self):
        if self.i == self.n_players:
            raise StopIteration
        m = self._get_masks_for_index(self.i)
        self.i = self.i + 1
        return m

    @abstractmethod
    def _assert_input_compatibility(self, inputs):
        pass

    @abstractmethod
    def _get_masks_for_index(self, i):
        pass

    @abstractmethod
    def _get_number_of_players_from_shape(self):
        pass


class DefaultPlayerIterator(AbstractPlayerIterator):

    def _assert_input_compatibility(self, inputs):
        assert len(inputs.shape) > 1, 'DefaultPlayerIterator requires an input with 2 or more dimensions'

    def _get_number_of_players_from_shape(self):
        return int(np.prod(self.input_shape))

    def _get_masks_for_index(self, i):
        mask = np.zeros(self.n_players, dtype='int32')
        mask[self.permutation[i]] = 1
        return mask.reshape(self.input_shape), mask.reshape(self.input_shape)


def keep_variance(x, min_variance):
    return x + min_variance


def convert_2_lpdn(model: nn.Module, convert_weights: bool = True) -> nn.Module:
    """
    Convert the model into a LPDN
    Conversion code skeleton from https://discuss.pytorch.org/t/how-can-i-replace-an-intermediate-layer-in-a-pre-trained-network/3586/7
    :param model: The model to convert
    :param convert_weights:
    :return: converted LPDN
    """
    min_variance = 1e-3
    keep_variance_fn = lambda x: keep_variance(x, min_variance)
    for name, module in reversed(model._modules.items()):
        if len(list(module.children())) > 0:
            # recurse
            model._modules[name] = convert_2_lpdn(module, convert_weights)
        else:
            if isinstance(module, nn.Conv2d):
                layer_new = adf.Conv2d(module.in_channels, module.out_channels, module.kernel_size, module.stride,
                                       module.padding, module.dilation, module.groups,
                                       module.bias is not None, module.padding_mode, keep_variance_fn=keep_variance_fn)
            elif isinstance(module, nn.Linear):
                layer_new = adf.Linear(module.in_features, module.out_features, module.bias is not None,
                                       keep_variance_fn=keep_variance_fn)
            elif isinstance(module, nn.ReLU):
                layer_new = adf.ReLU(keep_variance_fn=keep_variance_fn)
            elif isinstance(module, nn.LeakyReLU):
                layer_new = adf.LeakyReLU(negative_slope=module.negative_slope, keep_variance_fn=keep_variance_fn)
            elif isinstance(module, nn.Dropout):
                layer_new = adf.Dropout(module.p, keep_variance_fn=keep_variance_fn)
            elif isinstance(module, nn.MaxPool2d):
                layer_new = adf.MaxPool2d(keep_variance_fn=keep_variance_fn)
            elif isinstance(module, nn.ConvTranspose2d):
                layer_new = adf.ConvTranspose2d(module.in_channels, module.out_channels, module.kernel_size,
                                                module.stride, module.padding, module.output_padding, module.groups,
                                                module.bias, module.dilation, keep_variance_fn=keep_variance_fn)
            elif isinstance(module, nn.BatchNorm1d):
                continue
            else:
                raise NotImplementedError(f"Layer type {module} not supported")
            layer_old = module
            try:
                if convert_weights:
                    layer_new.weight = layer_old.weight
                    layer_new.bias = layer_old.bias
            except AttributeError:
                pass

            model._modules[name] = layer_new

    return model


class DASPModel(nn.Module):
    def __init__(self, first_layer, lpdn_model):
        super(DASPModel, self).__init__()
        self.first_layer = ProbDenseInput(first_layer.in_features, first_layer.out_features,
                                          bias=first_layer.bias is not None)
        self.lpdn_model = lpdn_model
        self.first_layer.weight = first_layer.weight
        self.first_layer.bias = first_layer.bias

    def forward(self, inputs: torch.Tensor, mask: torch.Tensor, k: int):
        x1_mean, x1_var, x2_mean, x2_var = self.first_layer(inputs, mask, k)
        y1_mean, y1_var = self.lpdn_model(x1_mean, x1_var)
        y2_mean, y2_var = self.lpdn_model(x2_mean, x2_var)

        return torch.stack([y1_mean, y1_var], -1), torch.stack([y2_mean, y2_var], -1)


class DASP(object):
    def __init__(self, model: nn.Module):
        self.model = model
        self._build_dasp_model()

    def _build_dasp_model(self):
        first_layer: nn.Linear = self.model.linear
        lpdn_model = self._convert_to_lpdn(self.model)
        lpdn_model.noise_variance = 1e-3
        self.dasp_model = DASPModel(first_layer, lpdn_model=lpdn_model)

    def _convert_to_lpdn(self, model: nn.Module):
        return convert_2_lpdn(model, True)

    def __call__(self, x, steps=None):
        player_generator = DefaultPlayerIterator(x)
        player_generator.set_n_steps(steps if x.shape[1] > steps else x.shape[1])
        ks = player_generator.get_steps_list()
        result = None
        tile_input = [len(ks)] + (len(x.shape) - 1) * [1]
        tile_mask = [len(ks) * x.shape[0]] + (len(x.shape) - 1) * [1]
        for i, (mask, mask_output) in enumerate(player_generator):
            # This line is from Keras implementation and will be updated soon
            # Workaround: as Keras requires the first dimension of the inputs to be the same,
            # we tile and repeat the input, mask and ks vector to have them aligned.
            y1, y2 = self.dasp_model(inputs=torch.tensor(np.tile(x, tile_input)),
                                     mask=torch.tensor(np.tile(mask, tile_mask)),
                                     k=torch.tensor(np.repeat(ks, x.shape[0])))
            y1 = y1.reshape(len(ks), x.shape[0], -1, 2)
            y2 = y2.reshape(len(ks), x.shape[0], -1, 2)
            y = torch.mean(y2[..., 0] - y1[..., 0], 0)
            if torch.isnan(y).any():
                raise RuntimeError('Result contains nans! This should not happen...')

            # Compute Shapley Values as mean of all coalition sizes
            if result is None:
                result = torch.zeros(y.shape + mask_output.shape)

            shape_mask = [1] * len(y.shape)
            shape_mask += list(mask_output.shape)

            shape_out = list(y.shape)
            shape_out += [1] * len(mask_output.shape)

            result += torch.reshape(y, shape_out) * torch.tensor(mask_output)

        return result


if __name__ == "__main__":
    pass
