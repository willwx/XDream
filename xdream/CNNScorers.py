import os
import numpy as np
from net_utils import net_loader
import utils
from Scorers import Scorer, WithIOScorer

gpu_available = net_loader.gpu_available


class ForwardHook:
    def __init__(self):
        self.i = None
        self.o = None

    def hook(self, module, i, o):
        self.i = i
        self.o = o


class NoIOCNNScorer(Scorer):
    """
    Scores images using a CNN unit without writing the images to disk
    """

    def __init__(self, log_dir, target_neuron, engine='caffe', image_size=None,
                 stochastic=False, stochastic_random_seed=None, stochastic_scale=None, reps=1,
                 load_on_init=True, **kwargs):
        """
        :param log_dir: str (path), directory to which to backup images and scores
        :param target_neuron: 3- or 5-tuple
            (network_name (str), layer_name (str), unit_index (int) [, x_index (int), y_index (int)])
        :param engine: 'caffe' or 'pytorch'
        :param image_size: int, size in pixels to which to resize all images; default is to leave unchanged
        :param stochastic: bool, whether to inject random Poisson noise to CNN responses; default is False
        :param stochastic_random_seed: int
            when set, the pseudo-stochastic noice will be deterministic; default is to set to `random_seed`
        :param stochastic_scale: float
            the score will be multipled by the scale before applying stochastic (Poisson) noise;
            default is None (not scaled)
        :param reps: int
            number of stochastic scores to produce for the same image
            only meaningful if stochastic == True
        """
        assert engine in ('caffe', 'pytorch'), f'CNNScorer for engine {engine} is not currently supported'

        super().__init__(log_dir, image_size=image_size,
                         stochastic=stochastic, stochastic_random_seed=stochastic_random_seed,
                         stochastic_scale=stochastic_scale, reps=reps)

        self._classifier_name = str(target_neuron[0])
        self._net_layer = str(target_neuron[1])
        self._net_iunit = target_neuron[2]
        if len(target_neuron) == 5:
            self._net_unit_x = int(target_neuron[3])
            self._net_unit_y = int(target_neuron[4])
            self._target_neuron = (self._classifier_name, self._net_layer, self._net_iunit,
                                   self._net_unit_x, self._net_unit_y)
        else:
            self._net_unit_x = None
            self._net_unit_y = None
            self._target_neuron = (self._classifier_name, self._net_layer, self._net_iunit)

        self._classifier = None
        self._engine = str(engine)
        self._transformer = None

        # torch specific
        self._torch_lib = None
        self._torch_dtype = None
        self._fwd_hk = None

        if load_on_init:
            self.load_classifier()

    def load_classifier(self):
        """
        Load the underlying neural network; initialize dependent attributes
        """
        self._classifier = net_loader.load_net(self._classifier_name, engine=self._engine)[0]
        self._transformer = net_loader.get_transformer(self._classifier_name, self._engine, self._classifier)
        if self._engine == 'pytorch':
            import torch
            self._torch_lib = torch
            fwd_hk = ForwardHook()
            layer_found = False
            for layer_name, layer in self._classifier.named_modules():
                if layer_name == self._net_layer:
                    layer.register_forward_hook(fwd_hk.hook)
                    layer_found = True
                    break
            assert layer_found, f'layer {self._net_layer} not found in pytoch network; ' +\
                f'available layers: {[l[0] for l in self._classifier.named_modules()]}'
            self._fwd_hk = fwd_hk
            self._torch_dtype = self._classifier.parameters().__iter__().__next__().dtype

    def _score_image(self, im):
        """
        :param im: a numpy array representing an 8-bit image
        :return: score for the given image judged by the target neuron
        """
        tim = self._transformer.preprocess('data', im / 255.)
        if self._engine == 'caffe':
            self._classifier.forward(data=np.array([tim]), end=self._net_layer)
            y = self._classifier.blobs[self._net_layer].data
        else:
            tim = self._torch_lib.tensor(tim[None, ...], dtype=self._torch_dtype)
            if gpu_available:
                self._classifier.forward(tim.cuda())
                y = self._fwd_hk.o.detach().cpu().numpy()
            else:
                self._classifier.forward(tim)
                y = self._fwd_hk.o.detach().numpy()
        score = y[0, self._net_iunit]
        if self._net_unit_x is not None:
            score = score[..., self._net_unit_x, self._net_unit_y]
        return score.copy()

    @property
    def parameters(self):
        params = super().parameters
        params.update({'target_neuron': self._target_neuron})
        return params


class WithIOCNNScorer(WithIOScorer, NoIOCNNScorer):
    """
    Scores images using a CNN unit
    Images are written to disk, read from disk, then evaluated with the same behavior as WithIOScorer/EphysScorer
    """

    def __init__(self, log_dir, target_neuron, write_dir, engine='caffe', image_size=None, random_seed=None,
                 stochastic=False, stochastic_random_seed=None, stochastic_scale=None, reps=1):
        """
        :param log_dir: str (path), directory to which to backup images and scores
        :param target_neuron: 5-tuple
            (network_name (str), layer_name (str), unit_index (int) [, x_index (int), y_index (int)])
        :param engine: 'caffe' or 'pytorch'
        :param write_dir: str (path), directory to which to write images
        :param image_size: int, size in pixels to which to resize all images; default is to leave unchanged
        :param random_seed: int
            when set, the scorer will have deterministic behavior (when writing images in a pseudo-random order)
            default is an arbitrary integer
        :param stochastic: bool, whether to inject random Poisson noise to CNN responses; default is False
        :param stochastic_random_seed: int
            when set, the pseudo-stochastic noice will be deterministic; default is to set to `random_seed`
        :param stochastic_scale: float
            the score will be multipled by the scale before applying stochastic (Poisson) noise;
            default is None (not scaled)
        :param reps: int
            number of stochastic scores to produce for the same image
            only meaningful if stochastic == True
        """
        NoIOCNNScorer.__init__(self, log_dir, target_neuron, engine=engine,
                               stochastic=stochastic, stochastic_random_seed=stochastic_random_seed,
                               stochastic_scale=stochastic_scale, reps=reps)
        WithIOScorer.__init__(self, write_dir, log_dir, image_size=image_size, random_seed=random_seed)

    def _with_io_get_scores(self):
        imgid_2_local_idx = {imgid: i for i, imgid in enumerate(self._curr_imgids)}
        organized_scores = []
        scores_local_idx = []
        novel_imgfns = []

        for imgfn in self._curr_imgfn_2_imgid.keys():
            im = utils.read_image(os.path.join(self._writedir, imgfn))
            score = self._score_image(im)
            try:
                imgid = self._curr_imgfn_2_imgid[imgfn]
                local_idx = imgid_2_local_idx[imgid]
                if self._stoch:
                    organized_scores += list(score)
                    scores_local_idx += [local_idx] * self._reps
                else:
                    organized_scores.append(score)
                    scores_local_idx.append(local_idx)
            except KeyError:
                novel_imgfns.append(imgfn)
        return organized_scores, scores_local_idx, novel_imgfns

    @property
    def parameters(self):
        params = super(WithIOCNNScorer, self).parameters
        return params


get_scorer = {'cnn_no_io': NoIOCNNScorer, 'cnn_with_io': WithIOCNNScorer}
defined_scorers = tuple(get_scorer.keys())


def load_scorer(scorer_name, scorer_parameters):
    """
    :param scorer_name: see `defined_scorers` in `Scorers.py`
    :param scorer_parameters: dict, kwargs passed when initializing the scorer
    :return: a Scorer object
    """
    return get_scorer[scorer_name](**scorer_parameters)
