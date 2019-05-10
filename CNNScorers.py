import os
import numpy as np
import net_loader
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
                 stochastic=False, stochastic_random_seed=None, reps=1, **kwargs):
        """
        :param log_dir: str (path), directory to which to backup images and scores
        :param target_neuron: 3- or 5-tuple
            (network_name (str), layer_name (str), unit_index (int) [, x_index (int), y_index (int)])
        :param engine: 'caffe' or 'pytorch'
        :param image_size: int, size in pixels to which to resize all images; default is to leave unchanged
        :param stochastic: bool, whether to inject random Poisson noise to CNN responses; default is False
        :param stochastic_random_seed: int
            when set, the pseudo-stochastic noice will be deterministic; default is to set to `random_seed`
        :param reps: int
            number of stochastic scores to produce for the same image
            only meaningful if stochastic == True
        """
        assert engine in ('caffe', 'pytorch'), f'CNNScorer for engine {engine} is not currently supported'

        Scorer.__init__(self, log_dir)

        self._classifier_name = str(target_neuron[0])
        self._net_layer = str(target_neuron[1])
        self._net_iunit = int(target_neuron[2])
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

        if image_size is None:
            self._imsize = None
        else:
            self._imsize = abs(int(image_size))

        self._stoch = bool(stochastic)
        self._stoch_rand_seed = None
        self._reps = 1
        if self._stoch:
            self._reps = int(max(1, reps))    # handles reps=None correctly
            print('%s: stochastic; %d reps' % (self.__class__.__name__, self._reps))
            if stochastic_random_seed is None:
                stochastic_random_seed = np.random.randint(100000)
                print('%s: stochastic random seed not provided, using %d for reproducibility' %
                      (self.__class__.__name__, stochastic_random_seed))
            else:
                stochastic_random_seed = abs(int(stochastic_random_seed))
                print('%s: stochastic random seed set to %d' % (self.__class__.__name__, stochastic_random_seed))
            self._stoch_rand_seed = stochastic_random_seed
            self._stoch_rand_gen = np.random.RandomState(seed=stochastic_random_seed)
        else:
            if stochastic_random_seed is not None:
                print('%s: not stochastic; stochastic random seed %s not used' %
                      (self.__class__.__name__, stochastic_random_seed))
            if reps is not None and reps != 1:
                print('%s: not stochastic; reps = %d not used' %
                      (self.__class__.__name__, reps))
                self._stoch_rand_seed = None
                self._stoch_rand_gen = None

        self._istep = -1
        self._curr_scores_mat = None
        self._curr_nscores = None

    def load_classifier(self):
        """
        Load the underlying neural network; initialize dependent attributes
        """
        self._classifier = net_loader.load(self._classifier_name, engine=self._engine)[0]
        self._transformer = net_loader.get_transformer(self._classifier_name, self._engine)
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

    def _score_image_by_CNN(self, im):
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
            score = score[self._net_unit_x, self._net_unit_y]
        if self._stoch:
            score = self._stoch_rand_gen.poisson(max(0, score), size=self._reps)
        return score

    def score(self, images, image_ids):
        nimgs = len(images)
        assert len(image_ids) == nimgs
        for imgid in image_ids:
            if not isinstance(imgid, str):
                raise ValueError('image_id should be str; got %s ' % str(type(imgid)))
        image_ids = utils.make_ids_unique(image_ids)

        scores_mat = []
        for im in images:
            im = utils.resize_image(im, self._imsize)
            score = self._score_image_by_CNN(im)
            scores_mat.append(score)
        scores_mat = np.array(scores_mat)
        if self._stoch:
            scores = np.mean(scores_mat, axis=1)
        else:
            scores = scores_mat
        self._curr_images = images
        self._curr_imgids = image_ids
        self._curr_scores = scores
        self._curr_scores_mat = scores_mat
        self._curr_nscores = np.full(len(images), self._reps)
        self._istep += 1
        return scores

    def save_current_scores(self):
        """
        Save scores for current images to log_dir
        """
        if self._istep < 0:
            raise RuntimeWarning('no scores evaluated; scores not saved')
        else:
            save_kwargs = {'image_ids': self._curr_imgids, 'scores': self._curr_scores}
            if self._stoch:
                save_kwargs.update({'scores_mat': self._curr_scores_mat, 'nscores': self._curr_nscores})
            savefpath = os.path.join(self._logdir, 'scores_step%03d.npz' % self._istep)
            print('saving scores to %s' % savefpath)
            utils.save_scores(savefpath, save_kwargs)

    @property
    def parameters(self):
        params = super(NoIOCNNScorer, self).parameters
        params.update({'target_neuron': self._target_neuron, 'image_size': self._imsize,
                       'stochastic': self._stoch, 'reps': self._reps, 'stochastic_random_seed': self._stoch_rand_seed})
        return params


class WithIOCNNScorer(WithIOScorer, NoIOCNNScorer):
    """
    Scores images using a CNN unit
    Images are written to disk, read from disk, then evaluated with the same behavior as WithIOScorer/EphysScorer
    """

    def __init__(self, log_dir, target_neuron, write_dir, engine='caffe', image_size=None, random_seed=None,
                 stochastic=False, stochastic_random_seed=None, reps=1):
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
        :param reps: int
            number of stochastic scores to produce for the same image
            only meaningful if stochastic == True
        """
        NoIOCNNScorer.__init__(self, log_dir, target_neuron, engine=engine,
                               stochastic=stochastic, stochastic_random_seed=stochastic_random_seed, reps=reps)
        WithIOScorer.__init__(self, write_dir, log_dir, image_size=image_size, random_seed=random_seed)

    def _get_scores(self):
        imgid_2_local_idx = {imgid: i for i, imgid in enumerate(self._curr_imgids)}
        organized_scores = []
        scores_local_idx = []
        novel_imgfns = []

        for imgfn in self._curr_imgfn_2_imgid.keys():
            im = utils.read_image(os.path.join(self._writedir, imgfn))
            score = self._score_image_by_CNN(im)
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
