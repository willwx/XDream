import numpy as np
import net_catalogue
from utils import resize_image

defined_generators = list(net_catalogue.defined_generators) + ['raw_pixel']


class Generator:
    def __init__(self, digitize_image, load_on_init):
        self._digitize = bool(digitize_image)
        if load_on_init:
            self.load_generator()

    def load_generator(self):
        pass

    def visualize(self, code):
        raise NotImplementedError

    def encode(self, im):
        raise NotImplementedError

    def reencode(self, code):
        raise NotImplementedError

    @property
    def name(self):
        raise NotImplementedError

    @property
    def code_shape(self):
        raise NotImplementedError

    @property
    def digitize_image(self):
        raise NotImplementedError

    @property
    def parameters(self):
        return {'class': self.__class__.__name__,
                'name': self.name, 'digitize_image': self.digitize_image, 'code_shape': self.code_shape}

    @property
    def loaded(self):
        return True

    @property
    def dtype(self):
        raise NotImplementedError


class NNGenerator(Generator):
    def __init__(self, gnn_name, digitize_image=True, load_on_init=True, fresh_copy=False):
        self._gnn_name = gnn_name
        self._fresh_copy = bool(fresh_copy)
        self._GNN = None
        self._detransformer = None
        self._input_layer_name = net_catalogue.net_io_layers[gnn_name]['input_layer_name']
        self._output_layer_name = net_catalogue.net_io_layers[gnn_name]['output_layer_name']
        self._code_shape = net_catalogue.net_io_layers[gnn_name]['input_layer_shape']
        self._input_layer_shape = (1, *self._code_shape)    # add batch dimension

        super(NNGenerator, self).__init__(digitize_image=digitize_image, load_on_init=load_on_init)

    def load_generator(self):
        if self._GNN is None:
            import net_loader
            generator = net_loader.load(self._gnn_name, self._fresh_copy)
            detransformer = net_loader.get_transformer(self._gnn_name, outputs_image=True)
            self._GNN = generator
            self._detransformer = detransformer

    def visualize(self, code):
        if self._GNN is None:
            raise RuntimeError('please load generator first')

        code = code.reshape(self._input_layer_shape)
        x = self._GNN.forward(**{self._input_layer_name: code})[self._output_layer_name]
        x = self._detransformer.deprocess('data', x)
        x = np.clip(x, 0, 1) * 255
        # x = x[14:241, 14:241, :]
        if self._digitize:
            return x.astype('uint8')
        else:
            return x

    def encode(self, im, steps=100):
        """
        encodes image into code space of generator by minimizing synthesized/target pixelwise difference
        :param im: array containing RGB image of type uint8 or type float (0 - 1)
        :param steps: number of optimization steps
        :return: generator code
        """
        if self._GNN is None:
            raise RuntimeError('please load generator first')

        if im.dtype == np.uint8:
            im = im.astype(float) / 255
        gen_tgt = self._detransformer.preprocess('data', im)
        code = np.zeros(shape=self._input_layer_shape)
        for i, step in enumerate(np.linspace(8, 1e-10, steps)):
            x0 = self._GNN.forward(**{self._input_layer_name: code})[self._output_layer_name]
            self._GNN.blobs[self._output_layer_name].diff[...] = gen_tgt - x0
            self._GNN.backward()
            gradient = self._GNN.blobs[self._input_layer_name].diff
            self._GNN.blobs[self._input_layer_name].data[:] += step / np.linalg.norm(gradient) * gradient
            code = self._GNN.blobs[self._input_layer_name].data
        return code.copy()

    def reencode(self, code, steps=100):
        if self._GNN is None:
            raise RuntimeError('please load generator first')

        im = self.visualize(code)
        recode = self.encode(im, steps=steps)
        return recode

    @property
    def name(self):
        return self._gnn_name

    @property
    def code_shape(self):
        return self._code_shape

    @property
    def digitize_image(self):
        return self._digitize

    @property
    def loaded(self):
        return self._GNN is not None

    @property
    def dtype(self):
        # caffe models use single-precision float
        return self._GNN.blobs[self._input_layer_name].data.dtype


class RawPixGenerator(Generator):
    def __init__(self, size=256, code_range=1, digitize_image=True, load_on_init=True):
        assert isinstance(size, int) and size > 0
        # warning: use of large code_range is discouraged due to potential problems when saving as float16
        # see self.dtype
        assert code_range > 0

        self._im_shape = (size, size, 3)
        self._drange = float(code_range)

        super(RawPixGenerator, self).__init__(digitize_image=digitize_image, load_on_init=load_on_init)

    def visualize(self, code):
        im = code.reshape(self._im_shape)
        im = np.clip(im, 0, self._drange) * 255.
        if self._digitize:
            return im.astype('uint8')
        return im

    def encode(self, im):
        code = resize_image(im, self._im_shape[0])
        if issubclass(code.dtype.type, np.floating):
            code = np.clip(code, 0, 1)
            if self._digitize:
                code = (code * 255.).astype('uint8') / 255.
        else:
            code = np.clip(code, 0, 255)
            if self._digitize:
                code = code.astype('uint8')
            code /= 255.
        return code * self._drange

    def reencode(self, code):
        im = self.visualize(code)
        recode = self.encode(im)
        return recode

    @property
    def code_range(self):
        return self._drange

    @property
    def name(self):
        return 'raw_pixel'

    @property
    def code_shape(self):
        return self._im_shape

    @property
    def digitize_image(self):
        return self._digitize

    @property
    def dtype(self):
        # half-precision float should be sufficient for saving raw_pix code, which maps to 8-bit images
        return np.float16


def get_generator(generator_name, *args, **kwargs):
    assert generator_name in defined_generators
    if generator_name == 'raw_pixel':
        return RawPixGenerator(*args, **kwargs)
    else:
        return NNGenerator(generator_name, *args, **kwargs)

