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
nrounds = 3
max_images = 10000

# param_vals_options is accessed as tuple = param_vals_options[optimizer_name][generator_name][stoch]
#     tuple[0] is tuple containing current best values, in the same order as target_params
#     tuple[1] is tuple containing value options, one tuple of options per target_param (in the same order)
param_vals_options = {
    'genetic': {
        'deepsim-fc6': {
            False: (
                (20, 0.5, 0.5, 2, 0.5, 0, None,),
                ((10, 20, 30, 40),    # population_size
                 (0.3, 0.4, 0.5, 0.6, 0.7),    # mutation_rate
                 (0.1, 0.25, 0.5, 0.75, 1),    # mutation_size
                 (1, 2, 3, 4),    # selectivity
                 (0.5, 0.75),    # heritability
                 (0, 1, 2)),    # n_conserve
            ),
        },
    }
}

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
param_val_options = {}
for iparam, param in enumerate(target_params):
    if param == 'reps':
        best_exp_settings[param] = param_vals_options[optimizer_name][generator_name][stoch][0][iparam]
    else:
        best_exp_settings['optimizer_parameters'][param] = \
            param_vals_options[optimizer_name][generator_name][stoch][0][iparam]
    if param in target_params:
        param_val_options[param] = param_vals_options[optimizer_name][generator_name][stoch][1][iparam]
best_scores = np.ones(len(target_neurons))    # dummy initial score

logtext = 'Initial condition:\n\tbest scores (dummy initialized)\n\t\t%s\n\texp settings\n\t\t%s\n' %\
          (str(best_scores), str(best_exp_settings))
logtext += 'Target neurons:\n\t%s\n' % str(target_neurons)
logtext += 'Target params values:\n\t%s\n' % param_val_options
logtext = flush_logtext_to_file(logtext)
t0 = time.time()
for iround in range(nrounds):
    # change up the order of target params
    np.random.shuffle(target_params)
    flush_logtext_to_file('\n\nRound %d\nTarget param order:\n\t%s\n' % (iround, target_params))

    for iparam, param in enumerate(target_params):
        logtext += '\nTesting param %s\n' % param
        if param == 'reps':
            old_param_val = best_exp_settings[param]
        else:
            old_param_val = best_exp_settings['optimizer_parameters'][param]
        scoress = []              # list of lists (one per param val option) of scores (one per target neuron)
        score_ratioss = []        # same structure as above
        mean_score_ratios = []    # list of ratios (one per param val option), averaged over target neurons
        tested_param_vals = []
        for new_param_val in param_val_options[param]:
            if new_param_val == old_param_val and (iparam > 0 or iround > 0):
                continue
            tested_param_vals.append(new_param_val)
            if param == 'reps':
                best_exp_settings[param] = new_param_val
            else:
                best_exp_settings['optimizer_parameters'][param] = new_param_val
            scoress.append([])
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
            best_scores = np.clip(scoress[imax], 1, None)    # in case score is negative or underflows
            logtext += 'Selected new value %s for param %s\n' % (str(tested_param_vals[imax]), param)
        else:
            new_param_val = old_param_val
            logtext += 'Selected old value %s for param %s\n' % (old_param_val, param)
        if param == 'reps':
            best_exp_settings[param] = new_param_val
        else:
            best_exp_settings['optimizer_parameters'][param] = new_param_val
        logtext += 'Current parameter setting:\n\t%s\n' % str(best_exp_settings)
        logtext = flush_logtext_to_file(logtext)

flush_logtext_to_file('\n\nFinal parameter setting:\n\t%s\n' % str(best_exp_settings))
