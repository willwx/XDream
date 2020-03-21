import os
from shutil import copyfile
import numpy as np
import Generators
import utils


class Optimizer:
    """
    Base class for Optimizer, which
    - Keeps a collection of codes (samples) that are used to synthesize images
    - Implements the `step()` method which
        - Takes the scores corresponding to the images
        - Updates the samples to increase expected scores
    """

    def __init__(self, generator_name, log_dir, random_seed=None, thread=None, generator_parameters=None):
        """
        :param generator_name: str, see `all_generators` in `net_catalogue.py`
        :param log_dir: str (path), directory to which to write backup codes & other log files
        :param random_seed: int
            when set, the optimizer will have deterministic behavior,
            producing identical results given identical calls to `step()`
            default is an arbitrary integer
        :param thread: int
            uniquely identifies concurrent opitmizers to avoid sample ID and backup filename clashes
            default is to assume no concurrent threads
        :param generator_parameters: dict, kwargs passed when initializing the generator
        """
        if thread is not None:
            assert isinstance(thread, int), 'thread must be an integer'
        assert os.path.isdir(log_dir)
        assert isinstance(random_seed, int) or random_seed is None, 'random_seed must be an integer or None'
        if generator_parameters is None:
            generator_parameters = {}
        assert isinstance(generator_parameters, dict)

        self._generator = Generators.get_generator(generator_name, **generator_parameters)
        self._code_shape = self._generator.code_shape
        self._istep = 0

        self._curr_samples = None       # array of codes
        self._curr_images = None        # list of image arrays
        self._curr_sample_idc = None    # range object
        self._curr_sample_ids = None    # list
        self._next_sample_idx = 0       # scalar

        self._best_code = None
        self._best_score = None

        self._thread = thread
        if thread is not None:
            log_dir = os.path.join(log_dir, 'thread%02d' % self._thread)
            if not os.path.isdir(log_dir):
                os.mkdir(log_dir)
        self._logdir = log_dir

        self._random_generator = np.random.RandomState()
        thread_str = ''
        if self._thread is not None:
            thread_str = ' (thread %d)' % self._thread
        if random_seed is None:
            random_seed = np.random.randint(100000)
            print('%s%s: random seed not provided, using %d for reproducibility' %
                  (self.__class__.__name__, thread_str, random_seed))
        else:
            print('%s%s: random seed set to %d' % (self.__class__.__name__, thread_str, random_seed))
        self._random_seed = random_seed
        self._random_generator = np.random.RandomState(seed=self._random_seed)

    def load_generator(self):
        """
        Load the underlying generative neural network (if applicable)
        """
        self._generator.load_generator()
        self._prepare_images()

    def _prepare_images(self):
        """
        Synthesize current images from current samples
        """
        if not self._generator.loaded:
            raise RuntimeError('generator not loaded; please run optimizer.load_generator() first')

        curr_images = []
        for sample in self._curr_samples:
            im_arr = self._generator.visualize(sample)
            curr_images.append(im_arr)
        self._curr_images = curr_images

    def set_initialization(self, samples):
        if self._curr_samples is None:
            raise NotImplemented
        else:
            assert self._istep == 0, 'can only change initialization at the beginning'
            assert samples.shape == self._curr_samples.shape, \
                f'initialization should be shape {self._curr_samples.shape}; '\
                f'got {samples.shape}'
            self._curr_samples = samples

    def step(self, scores):
        """
        Updates current samples and images to increase expected future scores
        :param scores: array of scalars corresponding to current images/samples
        """
        raise NotImplementedError

    def save_current_codes(self):
        """
        Saves current samples to disk
        """
        utils.write_codes(self._curr_samples.astype(self._generator.dtype), self._curr_sample_ids, self._logdir)

    def save_current_state(self, image_size=None):
        """
        Saves current samples and images to disk
        """
        self.save_current_codes()
        utils.write_images(self._curr_images, self._curr_sample_ids, self._logdir, image_size)

    @property
    def thread(self):
        return self._thread

    @property
    def istep(self):
        return self._istep

    @property
    def current_samples_copy(self):
        if self._curr_samples is None:
            return None
        else:
            return self._curr_samples.copy()

    @property
    def current_images(self):
        if self._curr_images is None:
            raise RuntimeError('Current images have not been initialized. Is generator loaded?')
        return self._curr_images

    @property
    def current_images_copy(self):
        return list(np.array(self._curr_images).copy())

    @property
    def current_image_ids(self):
        return self._curr_sample_ids

    @property
    def nsamples(self):
        return len(self._curr_samples)

    @property
    def generator(self):
        return self._generator

    @property
    def parameters(self):
        params = {'class': self.__class__.__name__,
                  'generator_name': self._generator.name, 'optimizer_random_seed': self._random_seed}
        if self._thread is not None:
            params['thread'] = self._thread
        params.update({'generator_parameters': self._generator.parameters})
        return params


class Genetic(Optimizer):
    """
    An optimizer based on the following genetic algorithm:
    - Each sample is an array of scalars (roughly analogous to a "genome" comprising "genes")
    - Each generation consists of `population_size` samples
    - Each new generation consists of
        - A few best samples from the last generation, as defined by `n_conserve`
        - New samples each produced by recombining, element-wise, two parent samples
    - Parents are selected based on their "fitness"
        - Fitness is computed from scores as a softmax of Z-scores within each generation
        - Selection is based on fitness but stochastic, with a stringency defined by `selectivity`;
          The higher the selectivity, the less likely that a low-fitness sample will be a parent
    - The two parents may contribute unevenly, as defined by `heritability`
    - Each recombined sample is subject to random mutation that
        - Affects a portion of the "genes", as defined by `mutation_rate`
        - Is drawn, element-wise, from an unbiased gaussian with scale `mutation_size`
    """

    def __init__(self, generator_name, log_dir, population_size, mutation_rate, mutation_size, selectivity,
                 heritability=0.5, n_conserve=0, n_conserve_ratio=None, random_seed=None, thread=None,
                 initial_codes_dir=None, save_init=True, generator_parameters=None):
        """
        :param generator_name: str, see see `all_generators` in `net_catalogue.py`
        :param log_dir: str (path), directory to which to write backup codes & other log files
        :param population_size: int (>= 1), number of samples per generation
        :param mutation_rate: float (>= 0, <= 1), portion of genes to mutate
        :param mutation_size: float (> 0), scale of mutation
        :param selectivity: float (> 0), stringency of selection
        :param heritability: float (meaningful range >= 0.5, <= 1)
            portion of genes contributed by one parent with the rest contributed by the other parent
        :param n_conserve: int (>= 0, < population size), number of best samples to keep unchanged per generation
        :param n_conserve_ratio: float, portion of samples to keep unchanged per generation; superseded by n_conserve
        :param random_seed: int
            when set, the optimizer will have deterministic behavior,
            producing identical results given identical calls to `step()`
            default is an arbitrary integer
        :param thread: int
            uniquely identifies concurrent opitmizers to avoid sample ID and backup filename clashes
            default is to assume no concurrent threads
        :param initial_codes_dir: str (path)
            directory containing code (.npy) files to make up the initial population
            default is to initialize with white noise
        :param save_init: bool, whether to save a copy of the initial codes; only used if using initial_codes_dir
        :param generator_parameters: dict, kwargs passed when initializing the generator
        """
        super(Genetic, self).__init__(generator_name, log_dir, random_seed, thread,
                                      generator_parameters)
        population_size = int(population_size)
        mutation_rate = float(mutation_rate)
        assert 0 <= mutation_rate <= 1, 'mutation_rate must be in range [0, 1]'
        mutation_size = float(mutation_size)
        assert mutation_size > 0, 'mutation_size must be positive'
        assert selectivity > 0, 'selectivity must be positive'
        heritability = float(heritability)
        assert 0 <= heritability <= 1, 'heritability must be in range [0, 1]'
        if heritability < 0.5:
            heritability = 1 - heritability
            print('Heritability is symmetrical around 0.5, so set to %.2f' % heritability)
        if n_conserve is not None:
            n_conserve = int(n_conserve)
        elif n_conserve_ratio is not None:
            n_conserve_ratio = float(n_conserve_ratio)
            assert 0 <= n_conserve_ratio < 1, 'n_conserve_ratio must be in range [0, 1)'
            n_conserve = max(0, int(population_size * n_conserve_ratio))
        else:
            raise ValueError('At least one of n_conserve or n_conserve_ratio needs to be passed')
        assert n_conserve < population_size
        if initial_codes_dir is not None:
            assert os.path.isdir(initial_codes_dir), 'initial_codes_dir %s is not a valid directory' % initial_codes_dir

        # optimizer parameters
        self._popsize = population_size
        self._mut_rate = mutation_rate
        self._mut_size = mutation_size
        self._selectivity = float(selectivity)
        self._heritability = heritability
        self._n_conserve = int(n_conserve)
        self._kT = None

        # initialize samples & indices
        self._init_population_dir = initial_codes_dir
        self._init_population_fns = None
        self._genealogy = None
        if initial_codes_dir is None:
            self._init_population = \
                self._random_generator.normal(loc=0, scale=1, size=(self._popsize, *self._code_shape))
            self._genealogy = ['standard_normal'] * self._popsize
        else:
            # this will set self._init_population and self._genealogy
            self._load_init_population(initial_codes_dir, copy=save_init)
        self._curr_samples = self._init_population.copy()    # curr_samples is current population of codes
        self._curr_sample_idc = range(self._popsize)
        self._next_sample_idx = self._popsize
        if self._thread is None:
            self._curr_sample_ids = ['gen%03d_%06d' % (self._istep, idx) for idx in self._curr_sample_idc]
        else:
            self._curr_sample_ids = ['thread%02d_gen%03d_%06d' %
                                     (self._thread, self._istep, idx) for idx in self._curr_sample_idc]

    def set_initialization(self, samples):
        super().set_initialization(samples)
        self._genealogy = ['manual_init'] * self._popsize

    def _load_init_population(self, intcodedir, size=None, copy=True):
        # make sure we are at the beginning of experiment
        assert self._istep == 0, 'initialization only allowed at the beginning'
        # make sure size <= population size
        if size is not None:
            assert size <= self._popsize, 'size %d too big for population of size %d' % (size, self._popsize)
        else:
            size = self._popsize
        # load codes
        init_population, genealogy = utils.load_codes2(intcodedir, size, self._random_generator)
        init_population = init_population.reshape((len(init_population), *self._code_shape))
        # fill the rest of population if size==len(codes) < population size
        if len(init_population) < self._popsize:
            remainder_size = self._popsize - len(init_population)
            remainder_pop, remainder_genealogy = \
                self._mate(init_population, genealogy, np.ones(len(init_population)), remainder_size)
            remainder_pop, remainder_genealogy = self._mutate(remainder_pop, remainder_genealogy)
            init_population = np.concatenate((init_population, remainder_pop))
            genealogy = genealogy + remainder_genealogy
        # apply
        self._init_population = init_population
        self._init_population_fns = genealogy    # a list of '*.npy' file names
        self._curr_samples = self._init_population.copy()
        self._genealogy = ['[init]%s' % g for g in genealogy]
        # no update for idc, idx, ids because popsize unchanged
        if copy:
            self._copy_init_population()
        if self._generator.loaded:
            self._prepare_images()

    def _copy_init_population(self):
        assert (self._init_population_fns is not None) and (self._init_population_dir is not None),\
            'please load init population first by calling _load_init_population();' + \
            'if init is not loaded from file, it will be backed-up as gen000 after experiment runs'
        savedir = os.path.join(self._logdir, 'init_population')
        try:
            os.mkdir(savedir)
        except OSError as e:
            if e.errno == 17:
                raise OSError('trying to save init population but directory already exists: %s' % savedir)
            else:
                raise
        for fn in self._init_population_fns:
            copyfile(os.path.join(self._init_population_dir, fn), os.path.join(savedir, fn))

    def _mutate(self, population, genealogy):
        do_mutate = self._random_generator.random_sample(population.shape) < self._mut_rate
        population_new = population.copy()
        population_new[do_mutate] += self._random_generator.normal(loc=0, scale=self._mut_size, size=np.sum(do_mutate))
        genealogy_new = ['%s+mut' % gen for gen in genealogy]
        return population_new, genealogy_new

    def _mate(self, population, genealogy, fitness, new_size):
        # need fitness > 0
        if np.max(fitness) == 0:
            fitness[np.argmax(fitness)] = 0.001
        if np.min(fitness) <= 0:
            fitness[fitness <= 0] = np.min(fitness[fitness > 0])

        fitness_bins = np.cumsum(fitness)
        fitness_bins /= fitness_bins[-1]
        parent1s = np.digitize(self._random_generator.random_sample(new_size), fitness_bins)
        parent2s = np.digitize(self._random_generator.random_sample(new_size), fitness_bins)
        new_samples = np.empty(shape=(new_size, *self._code_shape))
        new_genealogy = []
        for i in range(new_size):
            parentage = self._random_generator.random_sample(size=self._code_shape) < self._heritability
            new_samples[i, parentage] = population[parent1s[i]][parentage]
            new_samples[i, ~parentage] = population[parent2s[i]][~parentage]
            new_genealogy.append('%s+%s' % (genealogy[parent1s[i]], genealogy[parent2s[i]]))
        return new_samples, new_genealogy

    def step(self, scores):
        # clean variables
        assert len(scores) == len(self._curr_samples), \
            'number of scores (%d) != population size (%d)' % (len(scores), len(self._curr_samples))
        new_size = self._popsize
        new_samples = np.empty(shape=(new_size, *self._code_shape))
        curr_genealogy = np.array(self._curr_sample_ids, dtype=str)    # instead of chaining genealogy, alias every step
        new_genealogy = [''] * new_size    # np array not used because str len will be limited by len at init

        # deal with nan scores:
        nan_mask = np.isnan(scores)
        n_nans = int(np.sum(nan_mask))
        valid_mask = ~nan_mask
        n_valid = int(np.sum(valid_mask))
        if n_nans > 0:
            print('optimizer: missing %d scores for samples %s' %
                  (n_nans, str(np.array(self._curr_sample_idc)[nan_mask])))
            if n_nans > new_size:
                print('Warning: n_nans > new population_size because population_size has just been changed AND ' +
                      'too many images failed to score. This will lead to arbitrary loss of some nan score images.')
            if n_nans > new_size - self._n_conserve:
                print('Warning: n_nans > new population_size - self._n_conserve. ' +
                      'IFF population_size has just been changed, ' +
                      'this will lead to aribitrary loss of some/all nan score images.')
            # carry over images with no scores
            thres_n_nans = min(n_nans, new_size)
            new_samples[-thres_n_nans:] = self._curr_samples[nan_mask][-thres_n_nans:]
            new_genealogy[-thres_n_nans:] = curr_genealogy[nan_mask][-thres_n_nans:]

        # if some images have scores
        if n_valid > 0:
            valid_scores = scores[valid_mask]
            self._kT = max((np.std(valid_scores) / self._selectivity, 1e-8))    # prevents underflow kT = 0
            print('kT: %f' % self._kT)
            sort_order = np.argsort(valid_scores)[::-1]    # sort from high to low
            valid_scores = valid_scores[sort_order]
            valid_scores = np.clip(valid_scores, a_min=None, a_max=np.percentile(valid_scores, 95))    # remove outlier
            # Note: if new_size is smalled than n_valid, low ranking images will be lost
            thres_n_valid = min(n_valid, new_size)
            new_samples[:thres_n_valid] = self._curr_samples[valid_mask][sort_order][:thres_n_valid]
            new_genealogy[:thres_n_valid] = curr_genealogy[valid_mask][sort_order][:thres_n_valid]

            # if need to generate new samples
            if n_nans < new_size - self._n_conserve:
                fitness = np.exp((valid_scores - valid_scores[0]) / self._kT)
                # skips first n_conserve samples
                n_mate = new_size - self._n_conserve - n_nans
                new_samples[self._n_conserve:thres_n_valid], new_genealogy[self._n_conserve:thres_n_valid] = \
                    self._mate(new_samples[:thres_n_valid], new_genealogy[:thres_n_valid], fitness, n_mate)
                new_samples[self._n_conserve:thres_n_valid], new_genealogy[self._n_conserve:thres_n_valid] = \
                    self._mutate(new_samples[self._n_conserve:thres_n_valid],
                                 new_genealogy[self._n_conserve:thres_n_valid])

            # if any score turned out to be best
            if self._best_score is None or self._best_score < valid_scores[0]:
                self._best_score = valid_scores[0]
                self._best_code = new_samples[0].copy()

        self._istep += 1
        self._curr_samples = new_samples
        self._genealogy = new_genealogy
        self._curr_sample_idc = range(self._next_sample_idx, self._next_sample_idx + new_size)
        self._next_sample_idx += new_size
        if self._thread is None:
            self._curr_sample_ids = ['gen%03d_%06d' % (self._istep, idx) for idx in self._curr_sample_idc]
        else:
            self._curr_sample_ids = ['thread%02d_gen%03d_%06d' %
                                     (self._thread, self._istep, idx) for idx in self._curr_sample_idc]
        self._prepare_images()

    def save_current_genealogy(self):
        savefpath = os.path.join(self._logdir, 'genealogy_gen%03d.npz' % self._istep)
        save_kwargs = {'image_ids': np.array(self._curr_sample_ids, dtype=str),
                       'genealogy': np.array(self._genealogy, dtype=str)}
        utils.savez(savefpath, save_kwargs)

    @property
    def generation(self):
        return self._istep

    @property
    def n_samples(self):
        return self._popsize

    @property
    def current_average_sample(self):
        return np.mean(self._curr_samples, axis=0)

    @property
    def parameters(self):
        params = super(Genetic, self).parameters
        params.update(
            {'population_size': self._popsize, 'mutation_rate': self._mut_rate, 'mutation_size': self._mut_size,
             'selectivity': self._selectivity, 'heritability': self._heritability, 'n_conserve': self._n_conserve})
        return params


class FDGD(Optimizer):
    """
    An optimizer that estimates gradients using finite differences
    At each step, several samples are proposed around the current center, the local gradient is estimated as dy/dx,
        and the optimizer descends the gradient
    """

    def __init__(self, generator_name, log_dir, n_samples, search_radius, learning_rate,
                 antithetic=True, random_seed=None, thread=None,
                 initial_codes_dir=None, save_init=True,
                 generator_parameters=None):
        """
        :param generator_name: str, see see `all_generators` in `net_catalogue.py`
        :param log_dir: str (path), directory to which to write backup codes & other log files
        :param n_samples: int (>= 1), number of samples per step; rounded down to an even integer if antithetic == True
        :param search_radius: float (> 0), scale of search around the current center
        :param learning_rate: float (> 0), size of gradient descent step
        :param antithetic: bool
            whether to use antithetic sampling; see Ilyas et al. 2018: https://arxiv.org/abs/1804.08598
        :param random_seed: int
            when set, the optimizer will have deterministic behavior,
            producing identical results given identical calls to `step()`
            default is an arbitrary integer
        :param thread: int
            uniquely identifies concurrent opitmizers to avoid sample ID and backup filename clashes
            default is to assume no concurrent threads
        :param initial_codes_dir: str (path)
            directory containing code (.npy) files; a random one will be drawn as the initial center
            default is to initialize with zeros
        :param save_init: bool, whether to save a copy of the initial codes; only used if using initial_codes_dir
        :param generator_parameters: dict, kwargs passed when initializing the generator
        """
        super(FDGD, self).__init__(generator_name, log_dir, random_seed, thread,
                                   generator_parameters)
        assert n_samples >= 2, 'n_samples must be at least 2'
        if initial_codes_dir is not None:
            assert os.path.isdir(initial_codes_dir)

        # optimizer parameters
        self._r = float(search_radius)
        self._lr = float(learning_rate)
        self._antithetic = bool(antithetic)
        self._nsamps = int(n_samples) - 1
        self._n_indep_samps = self._nsamps
        if self._antithetic:
            self._n_indep_samps //= 2
            if int(n_samples) % 2 == 0:
                self._n_indep_samps += 1
                self._nsamps += 1
            else:
                print('FDGD: actual n_samples will be', self._n_indep_samps * 2)
        assert self._nsamps > 0

        if initial_codes_dir is None:
            self._curr_center = np.zeros(self._code_shape)
        else:
            self._load_init_code(initial_codes_dir, save_init)
        self._pos_steps = None
        self._norm_steps = None
        self._best_code = self._curr_center.copy()
        self._best_score = None
        self._init_codefn = None

        self._prepare_next_samples()

    def _prepare_next_samples(self):
        self._pos_steps = self._random_generator.normal(loc=0, scale=self._r,
                                                        size=(self._n_indep_samps, *self._curr_center.shape))
        self._norm_steps = np.mean(np.abs(self._pos_steps.reshape(self._n_indep_samps, -1)), axis=1)

        pos_samples = self._curr_center + self._pos_steps
        if self._antithetic:
            neg_samples = self._curr_center - self._pos_steps
            self._curr_samples = np.concatenate((pos_samples, neg_samples))
        else:
            self._curr_samples = np.concatenate((pos_samples, (self._curr_center.copy(),)))

        self._curr_sample_idc = range(self._next_sample_idx, self._next_sample_idx + len(self._curr_samples))
        self._next_sample_idx += len(self._curr_samples)
        if self._thread is None:
            self._curr_sample_ids = ['gen%03d_%06d' % (self._istep, idx) for idx in self._curr_sample_idc]
        else:
            self._curr_sample_ids = ['thread%02d_gen%03d_%06d' %
                                     (self._thread, self._istep, idx) for idx in self._curr_sample_idc]
        if self._generator.loaded:
            self._prepare_images()

    def _load_init_code(self, initcodedir, copy=True):
        # make sure we are at the beginning of experiment
        assert self._istep == 0, 'initialization only allowed at the beginning'
        init_code, init_codefns = utils.load_codes2(initcodedir, 1, self._random_generator)
        self._curr_center = init_code.reshape(self._code_shape)
        self._init_codefn = init_codefns[0]
        self._best_code = self._curr_center.copy()
        if copy:
            self._copy_init_code()

    def _copy_init_code(self):
        if self._init_codefn is None:
            print('FDGD: init code is default (all zero); not saved')
        else:
            savedir = os.path.join(self._logdir, 'init_code')
            try:
                os.mkdir(savedir)
            except OSError as e:
                if e.errno == 17:
                    raise OSError('trying to save init population but directory already exists: %s' % savedir)
                else:
                    raise
            utils.write_codes([self._curr_center], [self._init_codefn], savedir)

    def step(self, scores):
        assert len(scores) == len(self._curr_samples), \
            'number of scores (%d) != number of samples (%d)' % (len(scores), len(self._curr_samples))
        scores = np.array(scores)

        pos_scores = scores[:self._n_indep_samps]
        if self._antithetic:
            neg_scores = scores[self._n_indep_samps:]
            dscore = (pos_scores - neg_scores) / 2.
        else:
            dscore = pos_scores - scores[-1]

        grad = np.mean(dscore.reshape(-1, 1) * (self._norm_steps.reshape(-1, 1) ** -2) *
                       self._pos_steps.reshape(self._n_indep_samps, -1), axis=0).reshape(self._curr_center.shape)
        self._curr_center += self._lr * grad

        score_argmax = np.argsort(scores)[-1]
        if self._best_score is None or self._best_score < scores[score_argmax]:
            self._best_score = scores[score_argmax]
            self._best_code = self._curr_samples[score_argmax]

        self._istep += 1
        self._prepare_next_samples()

    @property
    def generation(self):
        return self._istep

    @property
    def n_samples(self):
        return self._nsamps

    @property
    def current_average_sample(self):
        return self._curr_center.copy()

    @property
    def parameters(self):
        params = super(FDGD, self).parameters
        params.update({'n_samples': self._nsamps, 'search_radius': self._r,
                       'learning_rate': self._lr, 'antithetic': self._antithetic})
        return params


get_optimizer = {'genetic': Genetic, 'Genetic': Genetic, 'fdgd': FDGD, 'FDGD': FDGD}
defined_optimizers = tuple(get_optimizer.keys())


def load_optimizer(optimizer_name, thread, optimizer_parameters):
    """
    :param optimizer_name: str, see `defined_optimizers` in `Optimizers.py`
    :param thread: int or None
        uniquely identifies concurrent opitmizers to avoid sample ID and backup filename clashes
        None means to assume no concurrent threads
    :param optimizer_parameters: dict, kwargs passed when initializing the optimizer
    :return: an Optimizer object
    """
    return get_optimizer[optimizer_name](thread=thread, **optimizer_parameters)
