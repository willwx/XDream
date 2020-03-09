import os
import time
import numpy as np
from Experiments import CNNExperiment

np.set_printoptions(precision=4, suppress=True)

# some arbitrary targets
target_neurons = (('caffenet', 'fc8', 1), ('caffenet', 'fc8', 407), ('caffenet', 'fc8', 632),
                  # ('placesCNN', 'fc8', 55), ('placesCNN', 'fc8', 74), ('placesCNN', 'fc8', 162),
                  # ('googlenet', 'loss3/classifier', 1), ('googlenet', 'loss3/classifier', 407),
                  # ('googlenet', 'loss3/classifier', 632),
                  # ('resnet-152', 'fc1000', 1), ('resnet-152', 'fc1000', 407), ('resnet-152', 'fc1000', 632),
                  )

optimizer_name = 'genetic'
target_params = ['population_size', 'mutation_rate', 'mutation_size', 'selectivity', 'heritability', 'n_conserve']
# optimizer_name = 'FDGD'
# target_params = ['n_samples', 'search_radius', 'learning_rate', 'antithetic']
init_codes_dir = None
stoch = False
if stoch:
    target_params += ['reps']
generator_name = 'deepsim-fc6'
serial_number = 1
max_images = 10000
search_steps = (5, 3,)
advance_in_cycle = False

# param_vals_options is accessed as tuple = param_vals_options[optimizer_name][generator_name][stoch]
#   tuple[0] is tuple containing current best values, in the same order as target_params
#   tuple[1] is tuple containing either
#     one tuple of tuples per target_param (in the same order)
#       - tuple_[0]: value step sizes, in decreasing order
#       - tuple_[1]: value bounds (lbound, ubound), inclusive
#       - tuple_[2]: value dtype
#   or value options
param_vals_options = (    # Genetic
    (20, 0.5, 0.5, 2, 0.5, 0, None,),
    (((15, 10, 5, 2),         (2, 100), int),    # population_size
     ((0.2, 0.1, 0.05),       (0, 1), float),    # mutation_rate
     ((0.5, 0.25, 0.1, 0.05), (0, 20), float),    # mutation_size
     ((1, 0.5, 0.25),         (0.1, 5), float),    # selectivity
     ((0.25, 0.1),            (0.5, 1), float),    # heritability
     ((1,),                   (0, 10), int)),     # n_conserve
)
# param_vals_options = (    # FDGD
#     (20, 1.5, 1.5, True),
#     (((20, 10, 5, 2, 1), (2, 100), int),    # n_samples
#      ((0.5, 0.25, 0.1),  (0.01, 10), float),    # search_radius
#      ((0.5, 0.25, 0.1),  (0.01, 20), float),    # learning_rate
#      (True, False)),    # antithetic
# )

jobsuperdir = 'param_opt'    # to be changed by user
jobname = '%s-%s-%s%d' % (optimizer_name, generator_name, ('det', 'stoch')[stoch], serial_number)
jobrootdir = os.path.join(jobsuperdir, jobname)
logfn = '%s.txt' % jobname
logfpath = os.path.join(jobsuperdir, logfn)
if os.path.isfile(logfpath):
    raise IOError('log file %s exists!' % logfpath)


def flush_logtext_to_file(logtext):
    with open(logfpath, 'a') as f:
        f.write(logtext)
        return ''


# current best settings and param val options
best_exp_settings = {
    'optimizer_name': optimizer_name,
    'optimizer_parameters': {
        'generator_name': generator_name,
        'initial_codes_dir': init_codes_dir,
    },
    'with_write': False,
    'max_images': max_images,
    'random_seed': 0,
    'stochastic': stoch,
    'config_file_path': __file__,
}
param_stepsize_ptr = {param: 0 for param in target_params}
param_searchsteps_ptr = {param: 0 for param in target_params}
param_steps_unchanged = {param: 0 for param in target_params}
param_fixed_val_options = {param: not isinstance(param_vals_options[1][iparam][0], tuple)
                           for iparam, param in enumerate(target_params)}
param_to_index = {param: iparam for iparam, param in enumerate(target_params)}


def get_param_val(param, exp_settings):
    if param == 'reps':
        return exp_settings[param]
    else:
        return exp_settings['optimizer_parameters'][param]


def set_param_val(param, new_val, exp_settings):
    if param == 'reps':
        exp_settings[param] = new_val
    else:
        exp_settings['optimizer_parameters'][param] = new_val
    return exp_settings


def get_params_vals(exp_settings):
    param_vals = [None for _ in target_params]
    for param in target_params:
        param_vals[param_to_index[param]] = get_param_val(param, exp_settings)
    return param_vals


def get_param_val_options(param, param_curr_best):
    iparam = param_to_index[param]
    if param_fixed_val_options[param]:
        return param_vals_options[1][iparam]
    stepsize = param_vals_options[1][iparam][0][param_stepsize_ptr[param]]
    ub, lb = param_vals_options[1][iparam][1]
    dtype = param_vals_options[1][iparam][2]
    searchsteps = search_steps[param_searchsteps_ptr[param]]
    val_options = np.arange(param_curr_best-stepsize*searchsteps, param_curr_best+stepsize*searchsteps, stepsize)
    val_options = np.round(np.unique(np.clip(val_options, ub, lb)), 3).astype(dtype)
    val_options = val_options[np.argsort(np.abs(val_options-param_curr_best))][:searchsteps]
    val_options.sort()
    return val_options


def get_remaining_stepsizes():
    remaining_stepsizes = {}
    for param in target_params:
        iparam = param_to_index[param]
        if param_fixed_val_options[param]:
            remaining_stepsizes[param] = None
        else:
            remaining_stepsizes[param] = param_vals_options[1][iparam][0][param_stepsize_ptr[param]:]
    return remaining_stepsizes


param_val_options = {}
params_curr_best = {}
for iparam, param in enumerate(target_params):
    param_curr_best = param_vals_options[0][iparam]
    params_curr_best[param] = param_curr_best
    best_exp_settings = set_param_val(param, param_curr_best, best_exp_settings)
    param_val_options[param] = get_param_val_options(param, param_curr_best)
best_scores = np.ones(len(target_neurons))    # dummy initial score
logtext = 'Initial condition:\n\tbest scores (dummy initialized)\n\t\t%s\n\texp settings\n\t\t%s\n' %\
          (str(best_scores), str(best_exp_settings))
logtext += 'Target neurons:\n\t%s\n' % str(target_neurons)
logtext += 'Target params values:\n\t%s\n' % param_val_options
logtext += 'Target params step sizes:\n\t%s\n' % str(get_remaining_stepsizes())
logtext = flush_logtext_to_file(logtext)

finished = False
recently_advanced_params = set()
tested_params_vals_scores = {}
t0 = time.time()
iround = -1
while not finished:
    iround += 1
    # change up the order of target params
    np.random.shuffle(target_params)

    logtext += '\n\nRound %d\nTarget param order:\n\t%s\n' % (iround, target_params)
    # logtext += 'Target params values:\n\t%s\n' % param_val_options
    logtext = flush_logtext_to_file(logtext)

    none_changed = True
    for iparam, param in enumerate(target_params):
        logtext += '\nTesting param %s\n' % param
        old_param_val = get_param_val(param, best_exp_settings)
        tested_param_vals = []
        scoress = []              # list of lists (one per param val option) of scores (one per target neuron)
        score_ratioss = []        # same structure as above
        mean_score_ratios = []    # list of ratios (one per param val option), averaged over target neurons
        for new_param_val in param_val_options[param]:
            if new_param_val == old_param_val and (iround > 0 or iparam > 0):
                continue
            tested_param_vals.append(new_param_val)
            to_try_params_vals = get_params_vals(best_exp_settings)
            to_try_params_vals[param_to_index[param]] = new_param_val
            to_try_params_vals = tuple(to_try_params_vals)
            try:
                scoress.append(tested_params_vals_scores[to_try_params_vals])
                logtext += '\tvalue: %s, scores\t[' % str(new_param_val)
                for curr_score in scoress[-1]:
                    logtext += '%5.2f, ' % curr_score
            except KeyError:
                scoress.append([])
                best_exp_settings = set_param_val(param, new_param_val, best_exp_settings)
                logtext += '\tvalue: %s, scores\t[' % str(new_param_val)
                for target_neuron in target_neurons:
                    neuron = target_neuron
                    if len(neuron) == 5:
                        subdir = '%s_%s_%04d_%d,%d' % \
                                 (neuron[0], neuron[1].replace('/', '_'), neuron[2], neuron[3], neuron[4])
                    else:
                        subdir = '%s_%s_%04d' % (neuron[0], neuron[1].replace('/', '_'), neuron[2])
                    i = 0
                    while os.path.isdir(os.path.join(jobrootdir, subdir, str(i))):
                        i += 1
                    best_exp_settings['project_dir'] = os.path.join(jobrootdir, subdir, str(i))
                    best_exp_settings['target_neuron'] = target_neuron
                    os.makedirs(best_exp_settings['project_dir'])

                    experiment = CNNExperiment(**best_exp_settings)
                    experiment.run()

                    curr_score = np.max(experiment.scorer.curr_scores, axis=0)
                    scoress[-1].append(curr_score)
                    logtext += '%5.2f, ' % curr_score
                tested_params_vals_scores[to_try_params_vals] = scoress[-1].copy()

            score_ratioss.append(np.array(scoress[-1]) / best_scores)
            mean_score_ratios.append(np.mean(score_ratioss[-1]))
            if max(mean_score_ratios) == mean_score_ratios[-1] and mean_score_ratios[-1] > 1:
                sigstr = '*'
            else:
                sigstr = ''
            logtext = logtext[:-2] + ']\t%5.2f%s\t(T: +%d s)\n' % (mean_score_ratios[-1], sigstr, int(time.time() - t0))
            logtext = flush_logtext_to_file(logtext)
        if len(scoress) == 0:
            continue
        scoress = np.array(scoress)
        score_ratio = np.array(mean_score_ratios)
        imax = np.argmax(score_ratio)

        if score_ratio[imax] > 1:
            new_param_val = tested_param_vals[imax]
            params_curr_best[param] = new_param_val
            best_scores = np.clip(scoress[imax], 1, None)    # in case score is negative or underflows
            logtext += 'Selected new value %s for param %s\n' % (str(tested_param_vals[imax]), param)
            logtext += 'Current parameter setting:\n\t%s\n' % str(params_curr_best)
            if new_param_val != old_param_val:    # for edge case of first rep, first iparam
                none_changed = False
            param_steps_unchanged[param] = 0
            param_searchsteps_ptr[param] = 0
            param_val_options[param] = get_param_val_options(param, new_param_val)
            if not param_fixed_val_options[param]:
                logtext += 'New target values for param %s:\n\t%s\n' % (param, param_val_options[param])
            logtext = flush_logtext_to_file(logtext)
        else:
            new_param_val = old_param_val
            logtext += 'Selected old value %s for param %s\n' % (old_param_val, param)
            param_steps_unchanged[param] += 1
        best_exp_settings = set_param_val(param, new_param_val, best_exp_settings)

    if none_changed:
        advanceable_params = [param for param in target_params if not param_fixed_val_options[param]]
        if advance_in_cycle:
            considered_params = [param for param in advanceable_params if param not in recently_advanced_params]
        else:
            considered_params = advanceable_params
        steps_unchanged = [param_steps_unchanged[param] for param in considered_params]
        advanced = False
        advanced_param = None
        for advance_param in np.array(considered_params)[np.argsort(steps_unchanged)][::-1]:
            if param_stepsize_ptr[advance_param] == -1:
                if param_searchsteps_ptr[advance_param] == -1:
                    continue
                else:
                    param_searchsteps_ptr[advance_param] += 1
                    if param_searchsteps_ptr[advance_param] == len(search_steps):
                        param_searchsteps_ptr[advance_param] = -1
                    advanced = True
                    advanced_param = advance_param
                    break
            else:
                param_stepsize_ptr[advance_param] += 1
                iparam = param_to_index[advance_param]
                if param_stepsize_ptr[advance_param] == len(param_vals_options[1][iparam][0]):
                    param_stepsize_ptr[advance_param] = -1
                advanced = True
                advanced_param = advance_param
                break

        if not advanced:
            finished = True
        else:
            if advanced_param == 'reps':
                curr_param_val = best_exp_settings[advanced_param]
            else:
                curr_param_val = best_exp_settings['optimizer_parameters'][advanced_param]
            param_val_options[advanced_param] = get_param_val_options(advanced_param, curr_param_val)
            recently_advanced_params.add(advanced_param)
            if len(recently_advanced_params) == len(advanceable_params):
                recently_advanced_params = set()
            logtext += '\nAdvanced optimization (param: %s)\nCurrent target parameter step sizes:\n\t%s\n' %\
                       (advanced_param, str(get_remaining_stepsizes()))

flush_logtext_to_file('\n\nFinal parameter setting:\n\t%s\n' % str(best_exp_settings))
