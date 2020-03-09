import os
import shutil
from time import time, sleep
import numpy as np
import utils


class Scorer:
    def __init__(self, log_dir, **kwargs):
        """
        :param log_dir: str (path), directory to which to backup images and scores
        """
        assert os.path.isdir(log_dir), 'invalid record directory: %s' % log_dir

        self._curr_images = None
        self._curr_imgids = None
        self._curr_scores = None
        self._logdir = log_dir

    def score(self, images, image_ids):
        """
        :param images: a list of numpy arrays of dimensions (h, w, c) and type uint8 containing images to be scored
        :param image_ids: a list of strs as unique identifier of each image
        :return: scores for each image
        """
        raise NotImplementedError

    def save_current_scores(self):
        """
        Save scores for current images to log_dir
        """
        raise NotImplementedError

    @property
    def curr_scores(self):
        return self._curr_scores.copy()

    @property
    def parameters(self):
        return {'class': self.__class__.__name__}    # no other parameters


class BlockWriter:
    """
    Utilities for writing images to disk in specified block sizes, num of reps, random order, & image size
    Initialize with blockwriter.show_images(images, IDs), show with blockwriter.write_block() until blockwriter.done
    """
    def __init__(self, write_dir, backup_dir, block_size=None,
                 reps=1, image_size=None, random_seed=None, cleanup_dir=None):
        """
        :param write_dir: str (path), directory to which to write images
        :param backup_dir: str (path), directory to which to backup images and scores
        :param block_size: int, number of images to write per block; default is to write all images
        :param reps: int, number of times to show each image repeatedly
        :param image_size: int, size in pixels to which to resize all images; default is to leave unchanged
        :param random_seed: int
            when set, BlockWriter will have deterministic behavior (when writing images in a pseudo-random order)
            default is an arbitrary integer
        :param cleanup_dir: str (path)
            path to which images will be saved when calling cleanup; default is to delete images
        """
        self._writedir = write_dir
        self._backupdir = backup_dir
        self._cleanupdir = cleanup_dir
        self._images = None
        self._imgids = None
        self._nimgs = None
        self._imsize = None
        self._curr_block_imgfns = []
        self._imgfn_2_imgid = {}
        self._imgid_2_local_idx = None
        self._remaining_times_toshow = None
        self._iblock = -1
        self._iloop = -1
        self._blocksize = None
        self._reps = None

        self.block_size = block_size
        self.reps = reps
        self.image_size = image_size
        if random_seed is None:
            random_seed = np.random.randint(100000)
            print('%s: random seed not provided, using %d for reproducibility' %
                  (self.__class__.__name__, random_seed))
        else:
            print('%s: random seed set to %d' % (self.__class__.__name__, random_seed))
        self._random_generator = np.random.RandomState(seed=random_seed)
        self._random_seed = random_seed

    def cleanup(self):
        """
        Move all image files (with '.bmp' extension) in write_dir to cleanup_dir if it is not None, else remove them
        """
        for image_fn in [fn for fn in os.listdir(self._writedir) if os.path.splitext(fn)[-1] == '.bmp']:
            try:
                if self._cleanupdir is None:
                    os.remove(os.path.join(self._writedir, image_fn))
                else:
                    shutil.move(os.path.join(self._writedir, image_fn), os.path.join(self._cleanupdir, image_fn))
            except OSError:
                print('failed to clean up %s' % image_fn)

    def show_images(self, images, imgids):
        """
        Reset blockwriter to show the given images
        :param images: array of images
        :param imgids: array of strs containing one id for each image
        """
        nimgs = len(images)
        assert len(imgids) == nimgs
        if self._blocksize is not None:
            assert nimgs >= self._blocksize, 'not enough images for block'

        self._images = images
        self._imgids = imgids
        self._nimgs = nimgs
        self._remaining_times_toshow = np.full(len(images), self._reps, dtype=int)
        self._imgid_2_local_idx = {imgid: i for i, imgid in enumerate(imgids)}
        self._iloop = -1

    def write_block(self):
        """
        Writes a block of images to disk
        :return:
            imgfn_2_imgid: dict mapping each image filename written to each image id written
        """
        assert self._images is not None, 'no images loaded'
        if self._blocksize is not None:
            blocksize = self._blocksize
        else:
            blocksize = len(self._images)

        self._iblock += 1
        self._iloop += 1

        view = self._random_generator.permutation(self._nimgs)
        prioritized_view = np.argsort(self._remaining_times_toshow[view])[::-1][:blocksize]
        block_images = self._images[view[prioritized_view]]
        block_imgids = self._imgids[view[prioritized_view]]
        block_ids = ['block%03d_%03d' % (self._iblock, i) for i in range(blocksize)]
        block_imgfns = ['%s_%s.bmp' % (blockid, imgid) for blockid, imgid in zip(block_ids, block_imgids)]
        imgfn_2_imgid = {name: imgid for name, imgid in zip(block_imgfns, block_imgids)}

        # images written here
        utils.write_images(block_images, block_imgfns, self._writedir, self._imsize)

        self._curr_block_imgfns = block_imgfns
        self._imgfn_2_imgid = imgfn_2_imgid
        self._remaining_times_toshow[view[prioritized_view]] -= 1
        return imgfn_2_imgid

    def backup_images(self):
        """
        Move all image files in the current block to backup_dir
        """
        for imgfn in self._curr_block_imgfns:
            try:
                shutil.copyfile(os.path.join(self._writedir, imgfn), os.path.join(self._backupdir, imgfn))
            except OSError:
                print('%s: failed to backup image %s' % (self.__class__.__name__, imgfn))

    def show_again(self, imgids):
        """
        Show the given image ids one more time
        :param imgids: array of strs, must be one of the current image ids being shown
        """
        for imgid in imgids:
            try:
                local_idx = self._imgid_2_local_idx[imgid]
                self._remaining_times_toshow[local_idx] += 1
            except KeyError:
                print('%s: warning: cannot show image %s again; image is not registered'
                      % (self.__class__.__name__, imgid))

    @property
    def iblock(self):
        return self._iblock

    @property
    def iloop(self):
        return self._iloop

    @property
    def done(self):
        return np.all(self._remaining_times_toshow <= 0)

    @property
    def block_size(self):
        return self._blocksize

    @block_size.setter
    def block_size(self, block_size):
        assert isinstance(block_size, int) or block_size is None, 'block_size must be an integer or None'
        self._blocksize = block_size

    @property
    def reps(self):
        return self._reps

    @reps.setter
    def reps(self, reps):
        assert isinstance(reps, int), 'reps must be an integer'
        self._reps = reps

    @property
    def image_size(self):
        return self._imsize

    @image_size.setter
    def image_size(self, image_size):
        assert isinstance(image_size, int) or image_size is None, 'block_size must be an integer or None'
        self._imsize = image_size

    @property
    def random_seed(self):
        return self._random_seed


class WithIOScorer(Scorer):
    """ Base class for a Scorer that writes images to disk and waits for scores """
    def __init__(self, write_dir, log_dir, image_size=None, random_seed=None, **kwargs):
        """
        :param write_dir: str (path), directory to which to write images
        :param log_dir: str (path), directory to which to write log files
        :param image_size: int, size in pixels to which to resize all images; default is to leave unchanged
        :param random_seed: int
            when set, the scorer will have deterministic behavior (when writing images in a pseudo-random order)
            default is an arbitrary integer
        """
        # super(WithIOScorer, self).__init__(log_dir)
        Scorer.__init__(self, log_dir)

        assert os.path.isdir(write_dir), 'invalid write directory: %s' % write_dir
        self._writedir = write_dir
        self._score_shape = tuple()    # use an empty tuple to indicate shape of score is a scalar (not even a 1d array)
        self._curr_nimgs = None
        self._curr_listscores = None
        self._curr_cumuscores = None
        self._curr_nscores = None
        self._curr_scores_mat = None
        self._curr_imgfn_2_imgid = None
        self._istep = -1

        self._blockwriter = BlockWriter(self._writedir, self._logdir, random_seed=random_seed)
        self._blockwriter.image_size = image_size

        self._require_response = False
        self._verbose = False

    def score(self, images, image_ids):
        nimgs = len(images)
        assert len(image_ids) == nimgs
        if self._blockwriter.block_size is not None:
            assert nimgs >= self._blockwriter.block_size, 'too few images for block'
        for imgid in image_ids:
            if not isinstance(imgid, str):
                raise ValueError('image_id should be str; got %s ' % str(type(imgid)))
        image_ids = utils.make_ids_unique(image_ids)
        self._istep += 1

        self._curr_imgids = np.array(image_ids, dtype=str)
        self._curr_images = np.array(images)
        self._curr_nimgs = nimgs
        blockwriter = self._blockwriter
        blockwriter.show_images(self._curr_images, self._curr_imgids)
        self._curr_listscores = [[] for _ in range(nimgs)]
        self._curr_cumuscores = np.zeros((nimgs, *self._score_shape), dtype='float')
        self._curr_nscores = np.zeros(nimgs, dtype='int')
        while not blockwriter.done:
            t0 = time()

            self._curr_imgfn_2_imgid = blockwriter.write_block()
            t1 = time()

            blockwriter.backup_images()
            t2 = time()

            scores, scores_local_idx, novel_imgfns = self._get_scores()
            if self._score_shape == (-1,) and len(scores) > 0:    # if score_shape is the inital placeholder
                self._score_shape = scores[0].shape
                self._curr_cumuscores = np.zeros((nimgs, *self._score_shape), dtype='float')
            for score, idx in zip(scores, scores_local_idx):
                self._curr_listscores[idx].append(score)
                self._curr_cumuscores[idx] += score
                self._curr_nscores[idx] += 1
            if self._require_response:
                unscored_imgids = set(self._curr_imgfn_2_imgid.values()) - set(self._curr_imgids[scores_local_idx])
                blockwriter.show_again(unscored_imgids)
            t3 = time()

            blockwriter.cleanup()
            t4 = time()

            # report delays
            if self._verbose:
                print(('block %03d time: total %.2fs | ' +
                       'write images %.2fs  backup images %.2fs  ' +
                       'wait for results %.2fs  clean up images %.2fs  (loop %d)') %
                      (blockwriter.iblock, t4 - t0, t1 - t0, t2 - t1, t3 - t2, t4 - t3, blockwriter.iloop))
                if len(novel_imgfns) > 0:
                    print('novel images:  {}'.format(sorted(novel_imgfns)))

        # consolidate & save data before returning
        # calculate average score
        scores = np.empty(self._curr_cumuscores.shape)
        valid_mask = self._curr_nscores != 0
        scores[~valid_mask] = np.nan
        if np.sum(valid_mask) > 0:    # if any valid scores
            if len(self._curr_cumuscores.shape) == 2:
                # if multiple channels, need to reshape nscores for correct array broadcasting
                scores[valid_mask] = self._curr_cumuscores[valid_mask] / self._curr_nscores[:, np.newaxis][valid_mask]
            else:
                scores[valid_mask] = self._curr_cumuscores[valid_mask] / self._curr_nscores[valid_mask]
        # make matrix of all individual scores
        scores_mat = np.full((*scores.shape, max(self._curr_nscores)), np.nan)
        for i in range(len(self._curr_imgids)):
            if self._curr_nscores[i] > 0:
                scores_mat[i, ..., :self._curr_nscores[i]] = np.array(self._curr_listscores[i]).T
        # record scores
        self._curr_scores = scores
        self._curr_scores_mat = scores_mat
        return scores    # shape of (nimgs, [nchannels,])

    def _get_scores(self):
        """
        Method for obtaining scores for images shown, for ex by reading a datafile from disk
        :return:
            organized_scores: list of scores that match one of the expected image fns; see `_match_imgfn_2_imgid()`
            scores_local_idx: for each score in `organized_scores`, the index of the image_id it matches
            novel_imgfns: loaded image fns that do not match what is expected
        """
        raise NotImplementedError

    def save_current_scores(self):
        if self._istep < 0:
            raise RuntimeWarning('no scores evaluated; scores not saved')
        else:
            savefpath = os.path.join(self._logdir, 'scores_end_block%03d.npz' % self._blockwriter.iblock)
            save_kwargs = {'image_ids': self._curr_imgids, 'scores': self._curr_scores,
                           'scores_mat': self._curr_scores_mat, 'nscores': self._curr_nscores}
            print('saving scores to %s' % savefpath)
            utils.save_scores(savefpath, save_kwargs)

    @property
    def parameters(self):
        params = super(WithIOScorer, self).parameters
        params.update({'image_size': self._blockwriter.image_size,
                       'blockwriter_random_seed': self._blockwriter.random_seed})
        return params


class EPhysScorer(WithIOScorer):
    """
    WithIOScorer that expects scores saved in the .mat file 'block%03d.mat' % block_index, containing
        - filename of images in the cell array 'stimulusID'
        - responses in the matrix 'tEvokedResp' with shape (imgs, channels), aligned to stimulusID
    """

    _supported_match_imgfn_policies = ('strict', 'loose')

    def __init__(self, write_dir, log_dir, block_size=None, channel=None, reps=1, image_size=None, random_seed=None,
                 mat_dir=None, require_response=False, verbose=True, match_imgfn_policy='strict'):
        """
        :param write_dir: str (path), directory to which to write images
        :param log_dir: str (path), directory to which to write log files
        :param block_size: int, number of images to write per block; default is all images to score
        :param channel: int, iterable of [iterable of] ints, None, or Ellipsis
            index of channels on which to listen for responses given response matrix of dimension n_images x n_channels
            int means to return the score at the index
            iterable of [iterable of] ints means, for each index/iterable of indices:
                if index, to return the score at the index
                if iterable of indices, to return the score averaged over the indices
            None means to average over all channels
            Ellipsis means to return all channels
        :param reps: int, number of times to show each image
        :param image_size: int, size in pixels to which to resize all images; default is to leave unchanged
        :param random_seed: int
            when set, the scorer will have deterministic behavior (when writing images in a pseudo-random order)
            default is an arbitrary integer
        :param mat_dir: str (path), directory in which to expect the .mat file; default is write_dir
        :param require_response: bool
            if True, images are shown until all receive at least one score
        :param verbose: bool, whether to report delays
        :param match_imgfn_policy: see `_match_imgfn_2_imgid`
        """
        super(EPhysScorer, self).__init__(write_dir, log_dir, image_size, random_seed)
        self._blockwriter.reps = reps
        self._blockwriter.block_size = block_size

        if channel is None:
            self._channel = channel                # self._score_shape defaults to empty tuple()
        elif isinstance(channel, int):
            self._channel = channel
        elif channel is ...:
            self._channel = channel
            self._score_shape = (-1,)               # placeholder; will be overwritten later
        elif hasattr(channel, '__iter__'):
            channel_new = []
            for c in channel:
                if hasattr(c, '__iter__'):
                    channel_new.append(np.array(c, dtype=int))
                else:
                    channel_new.append(int(c))
            self._channel = channel_new
            self._score_shape = (len(channel),)    # each score is a vector of dimension len(channel)
        else:
            raise ValueError('channel must be one of int, iterable of ints, None, or Ellipsis')

        if mat_dir is None:
            self._respdir = write_dir
        else:
            assert os.path.isdir(mat_dir), 'invalid response directory: %s' % mat_dir
            self._respdir = mat_dir

        assert isinstance(require_response, bool), 'require_response must be True or False'
        assert isinstance(verbose, bool), 'verbose must be True or False'
        assert match_imgfn_policy in self._supported_match_imgfn_policies,\
            'match_imgfn_policy %s not supported; must be one of %s'\
            % (match_imgfn_policy, str(self._supported_match_imgfn_policies))
        self._match_imgfn_policy = match_imgfn_policy
        self._verbose = verbose
        self._require_response = require_response

    def _match_imgfn_2_imgid(self, result_imgfn):
        # stric: result_imgfn must exactly match what is expected
        if self._match_imgfn_policy == 'strict':
            return self._curr_imgfn_2_imgid[result_imgfn]
        # loose: result_imgfn need only match the part after block###_##_ and before the file extension
        elif self._match_imgfn_policy == 'loose':
            try:
                return self._curr_imgfn_2_imgid[result_imgfn]
            except KeyError:
                imgid = os.path.splitext(result_imgfn)[0]
                if imgid.find('block') == 0:
                    imgid = imgid[imgid.find('_') + 1:]
                    imgid = imgid[imgid.find('_') + 1:]
                return imgid

    def _get_scores(self):
        """
        :return:
            organized_scores: list of scores that match one of the expected image fns; see `_match_imgfn_2_imgid()`
            scores_local_idx: for each score in `organized_scores`, the index of the image_id it matches
            novel_imgfns: loaded image fns that do not match what is expected
        """
        imgid_2_local_idx = {imgid: i for i, imgid in enumerate(self._curr_imgids)}
        t0 = time()

        # wait for matf
        matfn = 'block%03d.mat' % self._blockwriter.iblock
        matfpath = os.path.join(self._respdir, matfn)
        print('waiting for %s' % matfn)
        while not os.path.isfile(matfpath):
            sleep(0.001)
        sleep(0.5)    # ensures mat file finish writing
        t1 = time()

        # load .mat file results
        result_imgfns, scores = utils.load_block_mat(matfpath)
        #    select the channel(s) to use
        if self._channel is None:
            scores = np.mean(scores, axis=-1)
        elif hasattr(self._channel, '__iter__'):
            scores_new = np.empty((scores.shape[0], len(self._channel)))
            for ic, c in enumerate(self._channel):
                if hasattr(c, '__iter__'):
                    scores_new[:, ic] = np.mean(scores[:, c], axis=-1)
                else:
                    scores_new[:, ic] = scores[:, c]
            scores = scores_new
        else:
            scores = scores[:, self._channel]
        print('read from %s: stimulusID %s  tEvokedResp %s' % (matfn, str(result_imgfns.shape), str(scores.shape)))
        t2 = time()

        # organize results
        organized_scores = []
        scores_local_idx = []
        novel_imgfns = []
        for result_imgfn, score in zip(result_imgfns, scores):
            try:
                imgid = self._match_imgfn_2_imgid(result_imgfn)
                local_idx = imgid_2_local_idx[imgid]
                organized_scores.append(score)
                scores_local_idx.append(local_idx)
            except KeyError:
                novel_imgfns.append(result_imgfn)
        t3 = time()

        print('wait for .mat file %.2fs  load .mat file %.2fs  organize results %.2fs' %
              (t1 - t0, t2 - t1, t3 - t2))
        return organized_scores, scores_local_idx, novel_imgfns

    @property
    def parameters(self):
        params = super(EPhysScorer, self).parameters
        params.update({'block_size': self._blockwriter.block_size, 'channel': self._channel,
                       'reps': self._blockwriter.reps})
        return params


class WithIODummyScorer(WithIOScorer):
    def __init__(self, write_dir, log_dir):
        super(WithIODummyScorer, self).__init__(write_dir, log_dir)

    def _get_scores(self):
        return np.ones(self._curr_nimgs), np.arange(self._curr_nimgs), []


class ShuffledEPhysScorer(EPhysScorer):
    """
    Shuffles scores returned by EPhysScorer; for use as a control experiment
    """

    _supported_match_imgfn_policies = ('strict', 'loose', 'gen_nat', 'no_check')

    def __init__(self, *args, shuffle_first_n=None, match_imgfn_policy='loose', **kwargs):
        """
        :param shuffle_first_n: only shuffle the first n responses; default is to shuffle all
        :param match_imgfn_policy: 'strict', 'loose', 'gen_nat', or 'no_check'
        """
        super(ShuffledEPhysScorer, self).__init__(*args, **kwargs)
        # for shuffling, use a separate random generator from the one generator in EPhysScorer
        self._shuffle_random_seed = self._blockwriter.random_seed
        self._random_generator = np.random.RandomState(seed=self._shuffle_random_seed)
        print('%s: shuffle random seed set to %d' % (self.__class__.__name__, self._shuffle_random_seed))
        if shuffle_first_n is not None:
            self._first = int(shuffle_first_n)
        else:
            self._first = None
        assert match_imgfn_policy in self._supported_match_imgfn_policies,\
            'match_imgfn_policy %s not supported; must be one of %s'\
            % (match_imgfn_policy, str(self._supported_match_imgfn_policies))
        self._match_imgfn_policy = match_imgfn_policy

        self._verbose = False

        # specifically used in self._match_imgfn_2_imgid()
        self._matcher_istep = self._istep - 1
        self._matcher_curr_idc = None
        self._matcher_curr_gen_idc = None
        self._matcher_curr_nat_idc = None
        self._matcher_curr_igen = None
        self._matcher_curr_inat = None
        self._matcher_curr_iall = None

    def _match_imgfn_2_imgid(self, result_imgfn):
        if self._match_imgfn_policy in ('strict', 'loose'):
            return super(ShuffledEPhysScorer, self,)._match_imgfn_2_imgid(result_imgfn)
        # match generated and natural images separately, but make no distinction within each category
        # behavior is undefined if # of gen and nat images expected is not equal to the same received
        elif self._match_imgfn_policy == 'gen_nat':
            # when step advanced, reinitialize
            if self._matcher_istep < self._istep:
                self._matcher_istep = self._istep
                self._matcher_curr_idc = np.arange(self._curr_nimgs)
                is_generated = np.array(['gen' in imgid for imgid in self._curr_imgids], dtype=bool)
                gen_idc = self._matcher_curr_idc[is_generated]
                nat_idc = self._matcher_curr_idc[~is_generated]
                gen_idc = gen_idc[np.argsort(self._curr_nscores[is_generated])]
                nat_idc = nat_idc[np.argsort(self._curr_nscores[~is_generated])]
                self._matcher_curr_gen_idc = gen_idc
                self._matcher_curr_nat_idc = nat_idc
                self._matcher_curr_igen = 0
                self._matcher_curr_inat = 0
            if 'gen' in result_imgfn:
                imgid = self._curr_imgids[self._matcher_curr_gen_idc[self._matcher_curr_igen]]
                self._matcher_curr_igen += 1
                self._matcher_curr_igen %= len(self._matcher_curr_gen_idc)
            else:
                imgid = self._curr_imgids[self._matcher_curr_nat_idc[self._matcher_curr_inat]]
                self._matcher_curr_inat += 1
                self._matcher_curr_inat %= len(self._matcher_curr_nat_idc)
            return imgid
        # match completely arbitrarily
        elif self._match_imgfn_policy == 'no_check':
            if self._matcher_istep < self._istep:
                self._matcher_istep = self._istep
                self._matcher_curr_idc = np.arange(self._curr_nimgs)
                self._matcher_curr_iall = 0
            imgid = self._curr_imgids[self._matcher_curr_idc[self._matcher_curr_iall]]
            self._matcher_curr_iall += 1
            self._matcher_curr_iall %= self._curr_nimgs
            return imgid

    def score(self, *args, **kwargs):
        scores = super(ShuffledEPhysScorer, self).score(*args, **kwargs)

        shuffled_view = np.arange(len(scores))
        if self._first is None:
            self._random_generator.shuffle(shuffled_view)
        else:
            self._random_generator.shuffle(shuffled_view[:self._first])
        shuffled_scores = scores[shuffled_view]
        return shuffled_scores

    @property
    def parameters(self):
        params = super(ShuffledEPhysScorer, self).parameters
        params.update({'shuffle_random_seed': self._shuffle_random_seed})
        return params


get_scorer = {'ephys': EPhysScorer, 'dummy': WithIODummyScorer, 'shuffled_ephys': ShuffledEPhysScorer}
defined_scorers = tuple(get_scorer.keys())


def load_scorer(scorer_name, scorer_parameters):
    """
    :param scorer_name: see `defined_scorers` in `Scorers.py`
    :param scorer_parameters: dict, kwargs passed when initializing the scorer
    :return: a Scorer object
    """
    return get_scorer[scorer_name](**scorer_parameters)
