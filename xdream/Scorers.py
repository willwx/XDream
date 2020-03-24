import os
from os import path as ospath
import shutil
from time import time, sleep

import numpy as np
import h5py as h5
import utils


class Scorer:
    def __init__(self, log_dir, image_size=None,
                 stochastic=False, stochastic_random_seed=None, stochastic_scale=None, reps=1):
        """
        :param log_dir: str (path), directory to which to backup images and scores
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
        assert ospath.isdir(log_dir), f'invalid record directory: {log_dir}'
        if stochastic_scale is not None:
            assert stochastic_scale > 0
            stochastic_scale = float(stochastic_scale)
        else:
            stochastic_scale = 1

        if image_size is None:
            self._imsize = None
        else:
            self._imsize = abs(int(image_size))

        self._istep = -1
        self._curr_images = None
        self._curr_imgids = None
        self._curr_scores = None
        self._curr_scores_reps = None    # shape (n_ims, n_reps)
        self._curr_nscores = None
        self._logdir = log_dir

        self._stoch = bool(stochastic)
        self._stoch_rand_seed = None
        self._stoch_scale = stochastic_scale
        self._curr_scores_no_stoch = None
        self._reps = 1
        if self._stoch:
            self._reps = int(max(1, reps))    # handles reps=None correctly
            print(f'{self.__class__.__name__}: stochastic; {self._reps} reps')
            if stochastic_random_seed is None:
                stochastic_random_seed = np.random.randint(100000)
                print(f'{self.__class__.__name__}: stochastic random seed not provided, '
                      'using {stochastic_random_seed} for reproducibility')
            else:
                stochastic_random_seed = abs(int(stochastic_random_seed))
                print(f'{self.__class__.__name__}: stochastic random seed set to {stochastic_random_seed}')
            self._stoch_rand_seed = stochastic_random_seed
            self._stoch_rand_gen = np.random.RandomState(seed=stochastic_random_seed)
        else:
            if stochastic_random_seed is not None:
                print(f'{self.__class__.__name__}: not stochastic; '
                      f'stochastic random seed {stochastic_random_seed} not used')
            if reps is not None and reps != 1:
                print(f'{self.__class__.__name__}: not stochastic; reps = {reps} not used')
                self._stoch_rand_seed = None
                self._stoch_rand_gen = None

    def _score_image(self, im):
        raise NotImplemented

    def score(self, images, image_ids):
        """
        :param images: a list of numpy arrays of dimensions (h, w, c) and type uint8 containing images to be scored
        :param image_ids: a list of strs as unique identifier of each image;
            using non-unique ids will lead to undefined behavior
        :param skip_stoch: skip making scores stochastic; has not effect if scorer is not stochastic
        :return: scores for each image
        """
        nimgs = len(images)
        assert len(image_ids) == nimgs
        for imgid in image_ids:
            if not isinstance(imgid, str):
                raise ValueError(f'image_id should be str; got {type(imgid)}')

        scores = []
        scores_no_stoch = []
        for im in images:
            im = utils.resize_image(im, self._imsize)
            score = self._score_image(im)
            if self._stoch:
                scores_no_stoch.append(score)
                score = self._stoch_rand_gen.poisson(
                    max(0, score * self._stoch_scale),
                    size=(self._reps, *score.shape)
                ) / self._stoch_scale
            scores.append(score)
        scores = np.array(scores)
        if self._stoch:
            scores_reps = np.moveaxis(scores, 1, -1)
            scores = np.mean(scores, axis=1)
            scores_no_stoch = np.array(scores_no_stoch)
            self._curr_scores_reps = scores_reps
            self._curr_scores_no_stoch = scores_no_stoch
        else:
            self._curr_scores_reps = scores[..., None]
        self._curr_images = images
        self._curr_imgids = image_ids
        self._curr_scores = scores
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
                save_kwargs.update({'scores_reps': self._curr_scores_reps,
                                    'scores_no_stoch': self._curr_scores_no_stoch})
            savefpath = ospath.join(self._logdir, f'scores_step{self._istep:03d}.npz')
            print('saving scores to', savefpath)
            utils.save_scores(savefpath, save_kwargs)

    @property
    def curr_scores(self):
        return self._curr_scores.copy()

    @property
    def parameters(self):
        params = {'class': self.__class__.__name__, 'image_size': self._imsize,
                  'stochastic': self._stoch, 'reps': self._reps, 'stochastic_random_seed': self._stoch_rand_seed}
        if self._stoch_scale != 1:
            params['stochastic_scale'] = self._stoch_scale
        return params


class BlockWriter:
    """
    Utilities for writing images to disk in specified block sizes, num of reps, random order, & image size
    Initialize with blockwriter.show_images(images, IDs), show with blockwriter.write_block() until blockwriter.done
    """
    _available_image_formats = ('bmp', 'png')

    def __init__(self, write_dir, backup_dir, block_size=None,
                 reps=1, image_size=None, random_seed=None, cleanup_dir=None,
                 image_format='png', image_gray=False):
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
        assert image_format in self._available_image_formats, \
            f'image format {image_format} is not supported; ' \
            f'available formats: {self._available_image_formats}. ' \
            f'Please add new formats explicitly.'

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
        self._fmt = image_format
        self._grayscale = bool(image_gray)

        self.block_size = block_size
        self.reps = reps
        self.image_size = image_size
        if random_seed is None:
            random_seed = np.random.randint(100000)
            print(f'{self.__class__.__name__}: random seed not provided, using {random_seed} for reproducibility')
        else:
            print(f'{self.__class__.__name__}: random seed set to {random_seed}')
        self._random_generator = np.random.RandomState(seed=random_seed)
        self._random_seed = random_seed

    def check_write_dir_empty(self):
        for fn in os.listdir(self._writedir):
            if ospath.splitext(fn)[-1] == f'.{self._fmt}':
                return False
        return True

    def cleanup(self, write_dir=None):
        """
        Move all image files (extension matching image_format) in write_dir
        to cleanup_dir if it is not None, else remove them
        """
        write_dir = write_dir if write_dir is not None and os.path.isdir(write_dir) else self._writedir
        for image_fn in [fn for fn in os.listdir(write_dir) if ospath.splitext(fn)[-1] == f'.{self._fmt}']:
            try:
                if self._cleanupdir is None:
                    os.remove(ospath.join(write_dir, image_fn))
                else:
                    shutil.move(ospath.join(write_dir, image_fn), ospath.join(self._cleanupdir, image_fn))
            except OSError:
                print('failed to clean up', image_fn)

    def show_images(self, images, imgids, reps=None):
        """
        Reset blockwriter to show the given images
        :param images: array of images
        :param imgids: array of strs containing one id for each image
        """
        nimgs = len(images)
        assert len(imgids) == nimgs
        if self._blocksize is not None and nimgs >= self._blocksize:
            print('warning: not enough images for block', f'({nimgs} vs. {self._blocksize})')
        images = np.array([utils.resize_image(im, self._imsize) for im in images])

        self._images = images
        self._imgids = imgids
        self._nimgs = nimgs
        if isinstance(reps, int):
            self._remaining_times_toshow = np.full(len(images), reps, dtype=int)
        else:
            try:
                self._remaining_times_toshow = reps.astype(int)
            except (ValueError, AttributeError):
                if reps is not None:
                    print(f'ignored invalid value for reps: {reps}')
            self._remaining_times_toshow = np.full(len(images), self._reps, dtype=int)
        self._imgid_2_local_idx = {imgid: i for i, imgid in enumerate(imgids)}
        self._iloop = -1

    def write_block(self, wait_for_empty=-1):
        """
        Writes a block of images to disk
        :param wait_for_empty: float
            if positive, check write_dir every so many seconds until it is empty
        :return:
            imgfn_2_imgid: dict mapping each image filename written to each image id written
        """
        assert self._images is not None, 'no images loaded'
        blocksize = self._blocksize if self._blocksize is not None else len(self._images)
        self.iblock += 1
        self._iloop += 1

        try:
            wait_for_empty = float(wait_for_empty)
            if wait_for_empty > 0:
                while not self.check_write_dir_empty():
                    sleep(wait_for_empty)
        except ValueError:
            pass

        view = self._random_generator.permutation(self._nimgs)
        prioritized_view = np.argsort(self._remaining_times_toshow[view])[::-1][:blocksize]
        block_images = self._images[view[prioritized_view]]
        block_imgids = self._imgids[view[prioritized_view]]
        block_ids = [f'block{self._iblock:03d}_{i:03d}' for i in range(blocksize)]
        block_imgfns = [f'{blockid}_{imgid}.{self._fmt}' for blockid, imgid in zip(block_ids, block_imgids)]
        imgfn_2_imgid = {name: imgid for name, imgid in zip(block_imgfns, block_imgids)}

        # images written here
        utils.write_images(block_images, block_imgfns, self._writedir,
                           size=self._imsize, grayscale=self._grayscale, fmt=self._fmt)

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
                shutil.copyfile(ospath.join(self._writedir, imgfn), ospath.join(self._backupdir, imgfn))
            except OSError:
                print(f'{self.__class__.__name__}: failed to backup images {imgfn}')

    def backup_image_filenames(self):
        im_fns = np.array(self._curr_block_imgfns)
        with h5.File(os.path.join(self._backupdir, f'block{self._iblock:03d}-im_names.h5'), 'w') as f:
            try:
                f.create_dataset('image_names', data=im_fns.astype(np.bytes_))
            except UnicodeEncodeError:    # unicode chars in im_fns
                utils.save_unicode_array_to_h5(f, 'image_names', im_fns)

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
                print(f'{self.__class__.__name__}: warning: cannot show image {imgid} again; image is not registered')

    @property
    def iblock(self):
        return self._iblock

    @iblock.setter
    def iblock(self, iblock):
        self._iblock = iblock

    @property
    def iloop(self):
        return self._iloop

    @property
    def done(self):
        if self._remaining_times_toshow is None:
            return True
        return np.all(self._remaining_times_toshow <= 0)

    @property
    def block_size(self):
        return self._blocksize

    @block_size.setter
    def block_size(self, block_size):
        assert isinstance(block_size, int) or block_size is None, \
            f'block_size must be an integer or None; got {block_size}'
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
    def __init__(self, write_dir, log_dir,
                 image_size=None, image_format='png', backup_images=True,
                 random_seed=None, **kwargs):
        """
        :param write_dir: str (path), directory to which to write images
        :param log_dir: str (path), directory to which to write log files
        :param image_size: int, size in pixels to which to resize all images; default is to leave unchanged
        :param random_seed: int
            when set, the scorer will have deterministic behavior (when writing images in a pseudo-random order)
            default is an arbitrary integer
        """
        Scorer.__init__(self, log_dir, **kwargs)    # explicite call to superclass to avoid confusing MRO

        assert ospath.isdir(write_dir), f'invalid write directory: {write_dir}'
        self._writedir = write_dir
        self._score_shape = tuple()    # use an empty tuple to indicate shape of score is a scalar (not even a 1d array)
        self._curr_nimgs = None
        self._curr_listscores = None
        self._curr_cumuscores = None
        self._curr_imgfn_2_imgid = None
        self._istep = -1

        self._blockwriter = BlockWriter(
            self._writedir, self._logdir, image_format=image_format,
            random_seed=random_seed)
        self._blockwriter.image_size = image_size
        self._backup_ims = bool(backup_images)

        self._require_response = False
        self._verbose = False

    def score(self, images, image_ids):
        nimgs = len(images)
        assert len(image_ids) == nimgs
        if self._blockwriter.block_size is not None:
            assert nimgs >= self._blockwriter.block_size, 'too few images for block'
        for imgid in image_ids:
            if not isinstance(imgid, str):
                raise ValueError(f'image_id should be str; got {type(imgid)}')
        self._istep += 1

        self._curr_imgids = np.array(image_ids, dtype=str)
        self._curr_images = np.array(images)
        self._curr_nimgs = nimgs
        blockwriter = self._blockwriter
        blockwriter.show_images(self._curr_images, self._curr_imgids)
        self._curr_listscores = [[] for _ in range(nimgs)]
        try:
            self._curr_cumuscores = np.zeros((nimgs, *self._score_shape), dtype='float')
        except ValueError:        # if score_shape is the initial placeholder
            pass
        self._curr_nscores = np.zeros(nimgs, dtype='int')
        while not blockwriter.done:
            t0 = time()

            self._curr_imgfn_2_imgid = blockwriter.write_block()
            t1 = time()

            if self._backup_ims:
                blockwriter.backup_images()
            t2 = time()

            scores, scores_local_idx, novel_imgfns = self._with_io_get_scores()
            if self._score_shape == () and len(scores) > 0:    # if score_shape is the initial placeholder
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
                print(f'block {blockwriter.iblock:03d} time: total {t4 - t0:.2f}s | ' +
                      f'write images {t1 - t0:.2f}s  ' +
                      (f'backup images {t2 - t1:.2f}s  ' if self._backup_ims else '') +
                      f'wait for results {t3 - t2:.2f}s  clean up images {t4 - t3:.2f}s  (loop {blockwriter.iloop})')
                if len(novel_imgfns) > 0:
                    print('novel images: ', sorted(novel_imgfns))

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
        self._curr_scores_reps = scores_mat
        return scores    # shape of (nimgs, [nchannels,])

    def _with_io_get_scores(self):
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
            savefpath = ospath.join(self._logdir, f'scores_end_block{self._blockwriter.iblock:03d}.npz')
            save_kwargs = {'image_ids': self._curr_imgids, 'scores': self._curr_scores,
                           'scores_reps': self._curr_scores_reps, 'nscores': self._curr_nscores}
            print('saving scores to', savefpath)
            utils.save_scores(savefpath, save_kwargs)

    @property
    def parameters(self):
        params = super().parameters
        params.update({'image_size': self._blockwriter.image_size,
                       'blockwriter_random_seed': self._blockwriter.random_seed})
        return params


class EPhysScorer(WithIOScorer):
    """
    WithIOScorer that expects scores saved in the .mat file 'block%03d.mat' % block_index, containing
        - filename of images in the cell array 'stimulusID'
        - responses in the matrix 'tEvokedResp' with shape (imgs, channels), aligned to stimulusID
    """

    _available_score_formats = ('h5', 'mat')
    _supported_match_imgfn_policies = ('strict', 'loose')

    def __init__(self, write_dir, log_dir, score_dir=None, score_format='mat',
                 block_size=None, channel=None, reps=1,
                 image_size=None, image_format='png', backup_images=True,
                 require_response=False, match_imgfn_policy='strict',
                 random_seed=None, verbose=True):
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
        :param score_dir: str (path), directory in which to expect the .mat file; default is write_dir
        :param require_response: bool
            if True, images are shown until all receive at least one score
        :param verbose: bool, whether to report delays
        :param match_imgfn_policy: see `_match_imgfn_2_imgid`
        """
        super().__init__(
            write_dir, log_dir, image_size=image_size, image_format=image_format,
            backup_images=backup_images, random_seed=random_seed)
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

        if score_dir is None:
            self._respdir = write_dir
        else:
            assert ospath.isdir(score_dir), f'invalid response directory: {score_dir}'
            self._respdir = score_dir
        assert score_format in self._available_score_formats, f'invalid score format {score_format}; ' \
            f'options: {self._available_score_formats}'
        self._fmt = score_format

        assert isinstance(require_response, bool), 'require_response must be True or False'
        assert isinstance(verbose, bool), 'verbose must be True or False'
        assert match_imgfn_policy in self._supported_match_imgfn_policies, \
            f'match_imgfn_policy {match_imgfn_policy} not supported; ' \
            f'must be one of {self._supported_match_imgfn_policies}'
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
                imgid = ospath.splitext(result_imgfn)[0]
                if imgid.find('block') == 0:
                    imgid = imgid[imgid.find('_') + 1:]
                    imgid = imgid[imgid.find('_') + 1:]
                return imgid

    def _with_io_get_scores(self):
        """
        :return:
            organized_scores: list of scores that match one of the expected image fns; see `_match_imgfn_2_imgid()`
            scores_local_idx: for each score in `organized_scores`, the index of the image_id it matches
            novel_imgfns: loaded image fns that do not match what is expected
        """
        imgid_2_local_idx = {imgid: i for i, imgid in enumerate(self._curr_imgids)}
        t0 = time()

        # wait for matf
        score_fn = f'block{self._blockwriter.iblock:03d}.{self._fmt}'
        score_fp = ospath.join(self._respdir, score_fn)
        print('waiting for', score_fp)
        while not ospath.isfile(score_fp):
            sleep(0.1)
        sleep(0.5)    # ensures mat file finish writing
        t1 = time()

        # load .mat file results
        if self._fmt == 'h5':
            block_idc, scores = utils.load_h5_score_file(score_fp)
            imfns = np.array(sorted(self._blockwriter._curr_block_imgfns))
            if block_idc is None:
                assert len(scores) == len(imfns), \
                    'scores must correspond to imfns if no indices in ' \
                    f'score file; got len {len(scores)} vs. {len(imfns)}'
            else:
                imfns = imfns[block_idc]
            result_imfns = imfns
        elif self._fmt == 'mat':
            result_imfns, scores = utils.load_block_mat(score_fp)
        else:
            raise RuntimeError(f'loading scores is not implemented for format {self._fmt}')

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
        if self._verbose:
            print(f'read from {score_fn}: '
                  f'image filenames shape {result_imfns.shape} '
                  f'scores shape {scores.shape}')
        t2 = time()

        # organize results
        organized_scores = []
        scores_local_idx = []
        novel_imgfns = []
        for result_imgfn, score in zip(result_imfns, scores):
            try:
                imgid = self._match_imgfn_2_imgid(result_imgfn)
                local_idx = imgid_2_local_idx[imgid]
                organized_scores.append(score)
                scores_local_idx.append(local_idx)
            except KeyError:
                novel_imgfns.append(result_imgfn)
        t3 = time()

        print(f'wait for .mat file {t1 - t0:.2f}s  load .mat file {t2 - t1:.2f}s  organize results {t3 - t2:.2f}s')
        return organized_scores, scores_local_idx, novel_imgfns

    @property
    def parameters(self):
        params = super().parameters
        params.update({'block_size': self._blockwriter.block_size, 'channel': self._channel,
                       'reps': self._blockwriter.reps, 'score_format': self._fmt})
        return params


class WithIODummyScorer(WithIOScorer):
    def __init__(self, write_dir, log_dir):
        super().__init__(write_dir, log_dir)

    def _with_io_get_scores(self):
        return np.ones(self._curr_nimgs), np.arange(self._curr_nimgs), []


get_scorer = {'ephys': EPhysScorer, 'dummy': WithIODummyScorer}
defined_scorers = tuple(get_scorer.keys())


def load_scorer(scorer_name, scorer_parameters):
    """
    :param scorer_name: see `defined_scorers` in `Scorers.py`
    :param scorer_parameters: dict, kwargs passed when initializing the scorer
    :return: a Scorer object
    """
    return get_scorer[scorer_name](**scorer_parameters)
