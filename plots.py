import matplotlib.pyplot as plt
from operator import itemgetter
import json
import numpy as np

GIST_FEATURES_STATS_FILE_NAME = "GIST_FEATURES_STATS.json"
base_GTFR = 3.36

COLORS = ['tab:blue', 'tab:red', 'tab:green', 'tab:cyan', 'tab:pink', 'tab:brown']
LINE_COLORS = ['black', 'c', 'm', 'y', 'r', 'g', 'b']

model_results = {
    'GNN_Baseline':{
        'recall': 0.74406,
        'auc': 0.81654,
        'name': 'Baseline'
    },
    'CareGNN_Baseline':{
        'recall': 0.74462,
        'auc': 0.8237
    },
    'CareGNN_with_gist':{
        'recall': 0.76444,
        'auc': 0.85008
    },
    'RandomForest_Baseline':{
        'recall': 0.6391,
        'auc': 0.85874,
        'name': 'RF Baseline'
    },
    'RandomForest_with_25_gist_sqrt':{
        'recall': 0.63266,
        'auc': 0.86476
    },
    'RandomForest_with_50_gist_sqrt':{
        'recall': 0.62796,
        'auc': 0.86332
    },
    'RandomForest_with_25_gist_075':{
        'recall': 0.6496,
        'auc': 0.86644
    },
    'RandomForest_with_50_gist_075':{
        'recall': 0.64786,
        'auc': 0.86616
    },
    'RandomForest_with_25_gist_only_sqrt':{
        'recall': 0.59934,
        'auc': 0.78842
    },
    'RandomForest_with_50_gist_only_sqrt':{
        'recall': 0.59896,
        'auc': 0.79448
    },
    'RandomForest_with_25_gist_only_075':{
        'recall': 0.60242,
        'auc': 0.78484,
        'name': 'RF_25'
    },
    'RandomForest_with_50_gist_only_075':{
        'recall': 0.60398,
        'auc': 0.79228,
        'name': 'RF_50'
    }
}

model_results_vs_gist = {
    'A':{
        'x': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100, 112, 125, 137, 150, 162, 175, 187, 200],
        'recall': [0.75242, 0.75524, 0.7532, 0.75704, 0.75572, 0.75776, 0.75614, 0.75732, 0.75702, 0.76084, 0.75928, 0.76134, 0.76104, 0.7628, 0.76224, 0.76174, 0.76702, 0.76398, 0.76672, 0.76746, 0.7711, 0.76704, 0.77204, 0.77124, 0.77206, 0.77364, 0.7723, 0.77418, 0.77532, 0.7736, 0.77366],
        'auc': [0.82982, 0.83206, 0.83114, 0.8364, 0.83606, 0.83698, 0.8357, 0.8364, 0.83822, 0.8391, 0.8399, 0.84604, 0.84642, 0.84728, 0.84736, 0.84774, 0.84814, 0.84858, 0.84854, 0.84946, 0.84988, 0.8494, 0.85042, 0.85186, 0.85222, 0.85286, 0.852, 0.85338, 0.85338, 0.8538, 0.85468]
    },
    'B1':{
        'x': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 112, 125, 137, 150, 162, 175, 187, 200],
        'recall': [0.74706, 0.74554, 0.74456, 0.7472, 0.75038, 0.75012, 0.75268, 0.7534, 0.75328, 0.75498, 0.75124, 0.75311, 0.75306, 0.75546, 0.75248, 0.75122, 0.75356, 0.75596, 0.75656, 0.75543, 0.75378, 0.75266, 0.75822, 0.75766, 0.75592, 0.75582, 0.75382],
        'auc': [0.82072, 0.82166, 0.8233, 0.8267, 0.82606, 0.82718, 0.82862, 0.82948, 0.82958, 0.82994, 0.82984, 0.82989, 0.8321, 0.83174, 0.83192, 0.83212, 0.83146, 0.83242, 0.83062, 0.83353, 0.833, 0.83488, 0.83434, 0.83428, 0.8342, 0.83272, 0.8325]
    },
    'B2':{
        'x': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 35, 40, 45, 50],
        'recall': [0.74724, 0.74524, 0.74636, 0.75224, 0.74756, 0.75384, 0.75156, 0.7496, 0.75156, 0.75444, 0.7519, 0.75418, 0.75446, 0.75582, 0.7527, 0.75292, 0.75344, 0.75576],
        'auc': [0.82068, 0.8232, 0.82532, 0.82598, 0.82662, 0.82878, 0.82926, 0.82688, 0.82898, 0.83006, 0.83046, 0.8309, 0.83114, 0.8317, 0.83136, 0.8308, 0.83214, 0.83292]
    },
    'B3':{
        'x': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 35, 40, 45, 50],
        'recall': [0.7469, 0.74648, 0.75004, 0.7479, 0.74992, 0.7497, 0.74916, 0.7518, 0.74808, 0.7536, 0.7533, 0.75686, 0.759, 0.7597, 0.75756, 0.75984, 0.75996, 0.7604],
        'auc': [0.8211, 0.82384, 0.82394, 0.82672, 0.82666, 0.82616, 0.82852, 0.829, 0.82762, 0.82724, 0.82998, 0.8326, 0.83596, 0.83744, 0.83714, 0.83766, 0.8378, 0.83728]
    },
    'C':{
        'x': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100],
        'recall': [0.7531, 0.75558, 0.75704, 0.76166, 0.76082, 0.76174, 0.76188, 0.76218, 0.76202, 0.76092, 0.76378, 0.76854, 0.76746, 0.76906, 0.76864, 0.76914, 0.76886, 0.77154, 0.76858, 0.76948, 0.76456, 0.76464, 0.76818],
        'auc': [0.8323, 0.83552, 0.83648, 0.84102, 0.8411, 0.84138, 0.84182, 0.8427, 0.8427, 0.84432, 0.84536, 0.85134, 0.85162, 0.85198, 0.85244, 0.85204, 0.85308, 0.85382, 0.85194, 0.85322, 0.8512, 0.85168, 0.85074]
    },
    'D':{
        'x': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100, 125, 150, 175, 200],
        'recall': [0.51634, 0.53974, 0.60966, 0.61218, 0.6197, 0.61784, 0.62416, 0.6239, 0.6505, 0.65612, 0.66392, 0.68588, 0.6834, 0.68124, 0.68248, 0.68596, 0.68716, 0.6829, 0.6952, 0.69216, 0.6946, 0.70242, 0.70158, 0.70409, 0.71536, 0.7101, 0.66436],
        'auc': [0.71388, 0.61046, 0.67328, 0.68192, 0.68976, 0.69452, 0.70298, 0.70512, 0.73304, 0.73906, 0.74428, 0.76218, 0.75976, 0.76114, 0.7598, 0.7614, 0.7635, 0.76786, 0.77094, 0.773, 0.77542, 0.7752, 0.78012, 0.785805, 0.79022, 0.79358, 0.79148]
    }
}

def plot_model(model, lines=['GNN_Baseline'], title=f'Recall and AUC', save=False):
    title = f'{title} (Model {model})'
    x, recall, auc = model_results_vs_gist[model]['x'], model_results_vs_gist[model]['recall'], model_results_vs_gist[model]['auc']
    label = f'Model {model}'
    plt.close('all')

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(6, 6))
    handles = []
    for idx, line in enumerate(lines):
        handle = axes[0].axhline(model_results[line]['recall'], label=model_results[line]['name'], linestyle='--', linewidth=1, c=LINE_COLORS[idx])
        if len(lines) > 1:
            handles.append(handle)
        else:
            axes[0].text(max(x), model_results[line]['recall'], model_results[line]['name'], c=LINE_COLORS[idx], ha="right", va="bottom")
    plot, = axes[0].plot(x, recall, label=label, c="tab:red")
    axes[0].scatter(x, recall, label=label, s=4, c="tab:red")

    handles = []
    for idx, line in enumerate(lines):
        handle = axes[1].axhline(model_results[line]['auc'], label=model_results[line]['name'], linestyle='--', linewidth=1, c=LINE_COLORS[idx])
        if len(lines) > 1:
            handles.append(handle)
        else:
            axes[1].text(max(x), model_results[line]['auc'], model_results[line]['name'], c=LINE_COLORS[idx], ha="right", va="bottom")
    plot, = axes[1].plot(x, auc, label=label, c="tab:red")
    axes[1].scatter(x, auc, label=label, s=4, c="tab:red")

    axes[1].set_xlabel('Number of Gist')
    axes[0].text(max(x), min(recall), 'AUC', c="black", ha="right", va="bottom")
    axes[1].text(max(x), min(auc), 'Recall', c="black", ha="right", va="bottom")
    axes[0].legend(handles=[plot, *handles], fontsize='small', loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=1 + len(lines) if len(lines) > 1 else 1)


    if save: plt.savefig(f"graph/fig_{title}")

def plot_models(models=model_results_vs_gist.keys(), lines=['GNN_Baseline'], title=f'Recall and AUC', save=False, limit_x=False, colors=COLORS):
    plt.close('all')
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(6, 6))
    handles = [[], []]
    title = f'{title} (Model {", ".join(models)})'
    if limit_x:
        min_x = min(max(result['x']) for result in itemgetter(*models)(model_results_vs_gist))
        index_limits = [np.argwhere(np.array(result['x']) <= min_x)[-1][0] + 1 for result in itemgetter(*models)(model_results_vs_gist)]
    else:
        index_limits = [None] * len(models)

    for idx, model in enumerate(models):
        model_name = 'Model ' + model
        x, recall, auc = model_results_vs_gist[model]['x'][:index_limits[idx]], model_results_vs_gist[model]['recall'][:index_limits[idx]], model_results_vs_gist[model]['auc'][:index_limits[idx]]

        handles[0].append(axes[0].plot(x, recall, label=model_name, c=colors[idx])[0])
        axes[0].scatter(x, recall, label=model_name, s=4, c=colors[idx])

        handles[1].append(axes[1].plot(x, auc, label=model_name, c=colors[idx])[0])
        axes[1].scatter(x, auc, label=model_name, s=4, c=colors[idx])

    for idx, line in enumerate(lines):
        axes[0].axhline(model_results[line]['recall'], linestyle='--', linewidth=1, c=LINE_COLORS[idx])
        axes[0].text(max([max(result['x'][:index_limits[idx]]) for idx, result in enumerate(itemgetter(*models)(model_results_vs_gist))]), model_results[line]['recall'], model_results[line]['name'], c=LINE_COLORS[idx], ha="right", va="bottom")
        axes[1].axhline(model_results[line]['auc'], linestyle='--', linewidth=1, c=LINE_COLORS[idx])
        axes[1].text(max([max(result['x'][:index_limits[idx]]) for idx, result in enumerate(itemgetter(*models)(model_results_vs_gist))]), model_results[line]['auc'], model_results[line]['name'], c=LINE_COLORS[idx], ha="right", va="bottom")

    axes[1].set_xlabel('Number of Gist')
    axes[0].text(0, max([max(result['recall'][:index_limits[idx]]) for idx, result in enumerate(itemgetter(*models)(model_results_vs_gist))]), 'Recall', c="black", ha="left", va="top")
    axes[1].text(0, max([max(result['auc'][:index_limits[idx]]) for idx, result in enumerate(itemgetter(*models)(model_results_vs_gist))]), 'AUC', c="black", ha="left", va="top")
    axes[0].legend(handles=handles[0], fontsize='small', loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=len(models))

    if save: plt.savefig(f"graph/fig_{title}{'_limited' if limit_x else ''}")

def plot_gist_stats(num_of_gist_to_plot=3, title=f'Gist Stats', plot_average=True, save=False):
    with open(GIST_FEATURES_STATS_FILE_NAME) as file:
        gist_stats = json.load(file)
    plt.close('all')
    DISTANCE_STEP = 0.1
    MIN_DISTANCE = min([min(stats['distance_list']) for stats in gist_stats[:num_of_gist_to_plot]])
    MAX_DISTANCE = max([max(stats['distance_list']) for stats in gist_stats[:num_of_gist_to_plot]])
    num_to_plot = num_of_gist_to_plot + 1 if plot_average else num_of_gist_to_plot

    fig, left_axis = plt.subplots(figsize=(10, 5))
    right_axis = left_axis.twinx()
    total_population = sum(gist_stats[0]['population_list'])
    left_axis.axhline(base_GTFR, linestyle='--', linewidth=1, c="black")
    left_axis.text(MIN_DISTANCE, base_GTFR, f'Base = {base_GTFR}', c="black", ha="center", va="bottom")

    for idx, stats in enumerate(gist_stats[:num_of_gist_to_plot]):
        GTFR_list = [round(genuine / (fraudulent or 1), 2)  for genuine, fraudulent in zip(stats['genuine_list'], stats['fraudulent_list'])]
        left_axis.plot(stats['distance_list'], GTFR_list, label=stats['gist'])
        left_axis.scatter(stats['distance_list'], GTFR_list, s=4, label=stats['gist'])
        right_axis.bar(
            np.array(stats['distance_list']) - DISTANCE_STEP/2 + DISTANCE_STEP/num_to_plot*idx,
            np.array(stats['population_list']) / total_population * 100,
            width=DISTANCE_STEP/num_to_plot,
            align='edge',
            label=stats['gist'], alpha=0.5
        )

    if plot_average:
        gist_distance_map = [
            {
                distance: (population, genuine, fraudulent)
                for distance, population, genuine, fraudulent in
                zip(*itemgetter('distance_list', 'population_list', 'genuine_list', 'fraudulent_list')(stats))
            }
            for stats in gist_stats
        ]
        avg_distance_list, avg_population_list, avg_genuine_list, avg_fraudulent_list = [], [], [], []
        for distance in np.arange(MIN_DISTANCE, MAX_DISTANCE + DISTANCE_STEP, DISTANCE_STEP):
            distance = round(distance, 1)
            avg_distance_list.append(distance)
            avg_pop, avg_g, avg_f = np.mean([map[distance] for map in gist_distance_map if distance in map], axis=0)
            avg_population_list.append(avg_pop)
            avg_genuine_list.append(avg_g)
            avg_fraudulent_list.append(avg_f)

        GTFR_list = [round(genuine / (fraudulent or 1), 2) for genuine, fraudulent in zip(avg_genuine_list, avg_fraudulent_list)]
        left_axis.plot(avg_distance_list, GTFR_list, label='Average', c='black')
        left_axis.scatter(avg_distance_list, GTFR_list, s=4, label='Average', c='black')
        right_axis.bar(
            np.array(avg_distance_list) - DISTANCE_STEP / 2 + DISTANCE_STEP / num_to_plot * num_of_gist_to_plot,
            np.array(avg_population_list) / total_population * 100,
            width=DISTANCE_STEP / num_to_plot,
            align='edge',
            label='Average', alpha=0.5, color='black'
        )

    plt.legend(fontsize='small')
    left_axis.set_xlabel('Gist Distance')
    left_axis.set_ylabel('Genuine-To-Fradulent Ratio (GTFR)')
    right_axis.set_ylabel('Population (%)')

    if save: plt.savefig(f"graph/fig_{title}")

plot_gist_stats(3, save=True)

# for model in model_results_vs_gist.keys():
#     plot_model(model, save=True)
# plot_models(save=True)
plot_models(['A', 'B1', 'C'], save=True, colors=COLORS[0:3])
plot_model('D', lines=['GNN_Baseline', 'RandomForest_Baseline', 'RandomForest_with_25_gist_only_075', 'RandomForest_with_50_gist_only_075'], save=True)
plot_models([model for model in model_results_vs_gist.keys() if model.startswith('B')], save=True, limit_x=True, colors=[COLORS[1], *COLORS[3:5]])