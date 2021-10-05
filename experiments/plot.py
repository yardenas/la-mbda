import argparse
import itertools
import json
import os
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from tensorboard.backend.event_processing import event_accumulator

numbers = re.compile(r'(\d+)')

BENCHMARK_THRESHOLD = 25.0

SG6 = ['PointGoal1', 'PointGoal2', 'CarGoal1', 'PointButton1', 'PointPush1', 'DoggoGoal1']


def numerical_sort(value):
    value = str(value)
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts


def parse_tf_event_file(file_path):
    print('Parsing event file {}'.format(file_path))
    ea = event_accumulator.EventAccumulator(file_path)
    ea.Reload()
    if any(map(lambda metric: metric not in ea.scalars.Keys(),
               ['evaluation/average_return',
                'evaluation/average_cost_return',
                'training/episode_cost_return'])
           ):
        return [], [], [], []
    rl_objective, safety_objective, timesteps = [], [], []
    for i, (objective, cost_objective) in enumerate(zip(
            ea.Scalars('evaluation/average_return'), ea.Scalars('evaluation/average_cost_return')
    )):
        rl_objective.append(objective.value)
        safety_objective.append(cost_objective.value)
        timesteps.append(objective.step)
    sum_costs = 0.0
    sum_costs_per_step = []
    costs_iter = iter(ea.Scalars('training/episode_cost_return'))
    for step in timesteps:
        while True:
            cost = next(costs_iter)
            sum_costs += cost.value
            if cost.step >= step:
                break
        sum_costs_per_step.append(sum_costs)
    return rl_objective, safety_objective, sum_costs_per_step, timesteps


def parse(experiment_path, run, max_steps):
    run_rl_objective, run_cost_objective, run_sum_costs, run_timesteps = [], [], [], []
    files = list(Path(experiment_path).glob(os.path.join(run, 'events.out.tfevents.*')))
    last_time = -1
    all_sum_costs = 0
    for file in sorted(files, key=numerical_sort):
        objective, cost_objective, sum_costs, timestamps = parse_tf_event_file(
            str(file)
        )
        if not all([objective, cost_objective, sum_costs, timestamps]):
            print("Not all metrics are available!")
            continue
        # Filter out time overlaps, taking the first event file.
        run_rl_objective += [obj for obj, stamp in zip(objective, timestamps) if
                             last_time < stamp <= max_steps]
        run_cost_objective += [obj for obj, stamp in zip(cost_objective, timestamps) if
                               last_time < stamp <= max_steps]
        run_sum_costs += [(cost + all_sum_costs) / stamp for cost, stamp in zip(
            sum_costs, timestamps
        ) if last_time < stamp <= max_steps]
        run_timesteps += [stamp for stamp in timestamps if last_time < stamp <= max_steps]
        last_time = timestamps[-1]
        all_sum_costs = run_sum_costs[-1] * last_time
    return run_rl_objective, run_cost_objective, run_sum_costs, run_timesteps


def parse_experiment_data(experiment_path, max_steps=2e6):
    rl_objectives, cost_objectives, sum_costs, all_timesteps = [], [], [], []
    for metrics in map(
            parse, itertools.repeat(experiment_path), next(os.walk(experiment_path))[1],
            itertools.repeat(max_steps)
    ):
        run_rl_objective, run_cost_objective, run_sum_costs, run_timesteps = metrics
        rl_objectives.append(run_rl_objective)
        cost_objectives.append(run_cost_objective)
        sum_costs.append(run_sum_costs)
        all_timesteps.append(run_timesteps)
    return (
        np.asarray(rl_objectives), np.asarray(cost_objectives),
        np.asarray(sum_costs), np.asarray(all_timesteps)
    )


def median_percentiles(metric):
    median = np.median(metric, axis=0)
    upper_percentile = np.percentile(metric, 95, axis=0, interpolation='linear')
    lower_percentile = np.percentile(metric, 5, axis=0, interpolation='linear')
    return median, upper_percentile, lower_percentile


def make_statistics(eval_rl_objectives, eval_mean_sum_costs, sum_costs, timesteps):
    objectives_median, objectives_upper, objectives_lower = median_percentiles(eval_rl_objectives)
    mean_sum_costs_median, mean_sum_costs_upper, mean_sum_costs_lower = median_percentiles(
        eval_mean_sum_costs)
    average_costs_median, average_costs_upper, average_costs_lower = median_percentiles(sum_costs)
    return dict(objectives_median=objectives_median,
                objectives_upper=objectives_upper,
                objectives_lower=objectives_lower,
                mean_sum_costs_median=mean_sum_costs_median,
                mean_sum_costs_upper=mean_sum_costs_upper,
                mean_sum_costs_lower=mean_sum_costs_lower,
                average_costs_median=average_costs_median,
                average_costs_upper=average_costs_upper,
                average_costs_lower=average_costs_lower,
                timesteps=timesteps[0]
                )


def draw(ax, timesteps, median, upper, lower, label):
    ax.plot(timesteps, median, label=label)
    ax.fill_between(timesteps, lower, upper, alpha=0.2)
    ax.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
    ax.set_xlim([0, timesteps[-1]])
    ax.xaxis.set_major_locator(ticker.MaxNLocator(5, steps=[1, 2, 2.5, 5, 10]))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(5, steps=[1, 2, 2.5, 5, 10]))


def resolve_name(name):
    if 'not_safe' in name:
        return 'Unsafe LAMBDA'
    elif 'la_mbda' in name:
        return 'LAMBDA'
    elif 'greedy' in name:
        return 'Greedy LAMBDA'
    elif 'cem_mpc' in name:
        return 'CEM-MPC'
    else:
        return ""


def resolve_environment(name):
    if 'point_goal1' in name:
        return 'PointGoal1'
    elif 'point_goal2' in name:
        return 'PointGoal2'
    elif 'car_goal1' in name:
        return 'CarGoal1'
    elif 'point_button1' in name:
        return 'PointButton1'
    elif 'point_push1' in name:
        return 'PointPush1'
    elif 'doggo_goal1' in name:
        return 'DoggoGoal1'
    else:
        return ""


def draw_baseline(environments_paths, axes, baseline_path):
    with open(baseline_path) as file:
        benchmark_results = json.load(file)
    for environment, env_axes in zip(environments_paths, axes):
        env_name = resolve_environment(environment)
        ppo_lagrangian = benchmark_results[env_name]['ppo_lagrangian']
        trpo_lagrangian = benchmark_results[env_name]['trpo_lagrangian']
        cpo_lagrangian = benchmark_results[env_name]['cpo']
        for ax, value_cpo, value_ppo, value_trpo in zip(
                env_axes, cpo_lagrangian, ppo_lagrangian, trpo_lagrangian):
            lims = np.array(ax.get_xlim())
            ax.plot(lims, np.ones_like(lims) * value_cpo, ls='--',
                    label='CPO (proprio)')
            ax.plot(lims, np.ones_like(lims) * value_ppo, ls='--',
                    label='PPO Lagrangian (proprio)')
            ax.plot(lims, np.ones_like(lims) * value_trpo, ls='--',
                    label='TRPO Lagrangian (proprio)')


def draw_threshold(axes):
    for env_axes in axes:
        env_axes[1].axhline(BENCHMARK_THRESHOLD, ls='-', color='red')
        env_axes[2].axhline(BENCHMARK_THRESHOLD / 1000.0, ls='-', color='red')


def draw_experiment(env_axes, experiment_statistics, algo):
    for ax, metric_name, label in zip(
            env_axes,
            ['objectives', 'mean_sum_costs', 'average_costs'],
            ['Average reward return', 'Average cost return',
             'Cost regret']
    ):
        draw(ax, experiment_statistics['timesteps'],
             experiment_statistics[metric_name + '_median'],
             experiment_statistics[metric_name + '_upper'],
             experiment_statistics[metric_name + '_lower'],
             label=resolve_name(algo))
        ax.set_ylabel(label)


def summarize_experiments(config):
    root, algos, _ = next(os.walk(config.data_path))
    environments = list(next(os.walk(os.path.join(root, algos[0])))[1])
    fig, axes = plt.subplots(len(environments), 3, figsize=(2.4 * 4,
                                                            2.0 * len(environments)), sharex='row')
    axes = axes[None,] if len(environments) < 2 else axes
    steps = dict(PointGoal1=int(1e6), PointGoal2=int(1e6), CarGoal1=int(1e6),
                 PointButton1=int(2e6), PointPush1=int(2e6), DoggoGoal1=int(2e6))
    all_results = defaultdict(dict)
    all_errors = defaultdict(dict)
    annnotations = []
    for algo in algos:
        environments = next(os.walk(os.path.join(root, algo)))[1]
        for environment, env_axes in zip(environments, axes):
            experiment = os.path.join(root, algo, environment)
            print('Processing experiment {}...'.format(experiment))
            experiment_statistics = make_statistics(*parse_experiment_data(
                experiment,
                steps[resolve_environment(environment)]))
            all_results[resolve_environment(environment)][resolve_name(algo)] = (
                experiment_statistics['objectives_median'][-1],
                experiment_statistics['mean_sum_costs_median'][-1],
                experiment_statistics['average_costs_median'][-1]
            )
            all_errors[resolve_environment(environment)][resolve_name(algo)] = (
                ((experiment_statistics['objectives_lower'][-1],
                  experiment_statistics['objectives_upper'][-1])),
                (experiment_statistics['mean_sum_costs_lower'][-1],
                 experiment_statistics['mean_sum_costs_upper'][-1]),
                (experiment_statistics['average_costs_lower'][-1],
                 experiment_statistics['average_costs_upper'][-1])
            )
            draw_experiment(env_axes, experiment_statistics, algo)
            ymin, ymax = env_axes[1].get_ylim()
            env_axes[1].set_ylim([max(-12.5, ymin), min(300, ymax)])
            if env_axes[0].is_first_col():
                ann = env_axes[0].annotate(resolve_environment(environment), (0, 0.5),
                                           xytext=(-45, 0), ha='right', va='center',
                                           size=10, xycoords='axes fraction',
                                           textcoords='offset points')
                annnotations.append(ann)
    if not config.remove_baseline:
        draw_baseline(environments, axes, config.baseline_path)
    if not config.remove_threshold:
        draw_threshold(axes)
    for ax in axes[-1]:
        ax.set_xlabel('Training steps')
    leg = fig.legend(*axes[0, 0].get_legend_handles_labels(), loc='center',
                     bbox_to_anchor=(0.5, 0.00), ncol=5, frameon=False, fontsize=10, numpoints=1,
                     labelspacing=0.2, columnspacing=0.8, handlelength=1.2, handletextpad=0.5)
    fig.tight_layout(h_pad=0.5, w_pad=0.5)
    artist_to_keep = [leg] + annnotations
    plt.savefig(config.output_basename + '_curves.pdf',
                bbox_extra_artists=artist_to_keep, bbox_inches='tight')
    with open(config.output_basename + '_results.json', 'w') as file:
        json.dump(all_results, file)
    with open(config.output_basename + '_errors.json', 'w') as file:
        json.dump(all_errors, file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--remove_baseline', action='store_true')
    parser.add_argument('--remove_threshold', action='store_true')
    parser.add_argument('--baseline_path', type=str, default='baseline_long.json')
    parser.add_argument('--output_basename', type=str, default='output')
    args = parser.parse_args()
    import matplotlib as mpl

    mpl.rcParams["text.usetex"] = True
    mpl.rcParams["font.family"] = "serif"
    mpl.rcParams["font.serif"] = "Times New Roman"
    summarize_experiments(args)
