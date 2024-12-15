import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

crime_runtimes = pd.read_csv('../results/data/runtimes_crime.csv')
housing_runtimes = pd.read_csv('../results/data/runtimes_housing.csv')
crime_runtimes['Dataset'] = 'Crime'
housing_runtimes['Dataset'] = 'Housing'
combined_runtimes = pd.concat([crime_runtimes, housing_runtimes])
training_times = combined_runtimes[combined_runtimes['Type'] == 'training time']
prediction_times = combined_runtimes[combined_runtimes['Type'] == 'prediction time']

crime_scores = pd.read_csv('../results/data/scores_crime.csv')
housing_scores = pd.read_csv('../results/data/scores_housing.csv')
crime_scores['Dataset'] = 'Crime'
housing_scores['Dataset'] = 'Housing'
r2_scores = pd.concat([
    crime_scores[crime_scores['Type'] == 'r2'],
    housing_scores[housing_scores['Type'] == 'r2']
], ignore_index=True)

crime_grid = pd.read_csv('../results/data/grid_search_results_Custom Random Forest_crime.csv')
housing_grid = pd.read_csv('../results/data/grid_search_results_Custom Random Forest_housing.csv')


def plot_grouped_bar(data, title, value_column, file_name):
    pivot_data = data.pivot(index='Model', columns='Dataset', values=value_column)
    pivot_data.plot(kind='bar', figsize=(12, 6), width=0.8)
    plt.title(title)
    plt.ylabel(value_column)
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Dataset', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f'../results/plots/{file_name}.png')
    plt.close('all')


plot_grouped_bar(training_times, 'Training Times by Model and Dataset', 'Time', 'training_times.png')
plot_grouped_bar(prediction_times, 'Prediction Times by Model and Dataset', 'Time', 'prediction_times.png')

plot_grouped_bar(r2_scores, 'R2 Scores by Model and Dataset', 'Score', 'r2_scores.png')

crime_mse = crime_scores[crime_scores['Type'] == 'mse']
housing_mse = housing_scores[housing_scores['Type'] == 'mse']

crime_mse.plot(kind='bar', x='Model', y='Score', legend=False, figsize=(12, 6))
plt.title('MSE Scores by Model for Crime Dataset')
plt.ylabel('MSE')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f'../results/plots/mse_scores_crime.png')
plt.close('all')

housing_mse.plot(kind='bar', x='Model', y='Score', legend=False, figsize=(12, 6))
plt.title('MSE Scores by Model for Housing Dataset')
plt.ylabel('MSE')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f'../results/plots/mse_scores_housing.png')
plt.close('all')


def plot_grouped_bar_with_scores(grid_data, dataset_name, relevant_columns):
    for hyperparam in relevant_columns:
        grouped_data = grid_data.groupby(hyperparam)[['mean_test_r2', 'mean_test_mse']].mean().reset_index()
        grouped_data['mean_test_mse'] = -grouped_data['mean_test_mse']
        legend_label_mse = 'MSE Score'
        if dataset_name.lower() == 'housing':
            grouped_data['mean_test_mse'] = np.log(grouped_data['mean_test_mse'])
            legend_label_mse = 'MSE Score (Log Transformed)'
        pivot_data = grouped_data.melt(id_vars=[hyperparam], value_vars=['mean_test_r2', 'mean_test_mse'],
                                       var_name='Metric', value_name='Score')
        pivot_table = pivot_data.pivot(index=hyperparam, columns='Metric', values='Score')
        ax = pivot_table.plot(kind='bar', figsize=(12, 6))
        for container in ax.containers:
            ax.bar_label(container, fmt='%.2f')
        plt.title(f"Mean Scores for {hyperparam.replace('param_model__', '').replace('_', ' ').title()} in {dataset_name} Dataset")
        plt.ylabel('Mean Test Score')
        plt.xlabel(hyperparam.replace('param_model__', '').replace('_', ' ').title())
        plt.xticks(rotation=45)
        plt.legend(title='Metric', labels=['R2 Score', legend_label_mse], loc='right')
        plt.tight_layout()
        plt.savefig(f'../results/plots/{dataset_name.lower()}_{hyperparam.replace("param_model__", "").replace("_", "")}.png')
        plt.close('all')


relevant_columns = ['param_model__max_depth', 'param_model__n_features', 'param_model__n_trees']

plot_grouped_bar_with_scores(crime_grid, 'Crime', relevant_columns)
plot_grouped_bar_with_scores(housing_grid, 'Housing', relevant_columns)
