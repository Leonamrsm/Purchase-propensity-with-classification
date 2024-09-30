import pandas              as pd
import numpy               as np
import seaborn             as sns
import matplotlib.gridspec as gridspec
import scipy.stats         as stats

from IPython.core.display    import HTML
from matplotlib              import pyplot as plt


def get_first_order_statistics(df):
    # Central Tendency Metrics
    mean = pd.DataFrame(df.apply(np.mean)).T
    median = pd.DataFrame(df.apply(np.median)).T

    # Dispersion Metrics
    min_ = pd.DataFrame(df.apply(min)).T
    max_ = pd.DataFrame(df.apply(max)).T
    range_ = pd.DataFrame(df.apply(lambda x: x.max() - x.min())).T
    std = pd.DataFrame(df.apply(np.std)).T
    skew = pd.DataFrame(df.apply(lambda x: x.skew())).T
    kurtosis = pd.DataFrame(df.apply(lambda x: x.kurtosis())).T

    # Metrics Concatenation
    m = pd.concat([min_, max_, range_, mean, median, std, skew, kurtosis]).T.reset_index()
    m.columns = ['attributes', 'min', 'max', 'range', 'mean', 'median', 'std', 'skew', 'kurtosis']
    
    return m

def jupyter_settings():
    plt.style.use('bmh')  
    plt.rcParams['figure.figsize'] = [25, 12]  # Default figure size
    plt.rcParams['font.size'] = 24  # Font size in the plots
    display(HTML('<style>.container { width:100% !important; }</style>'))  # Change the notebook width
    pd.options.display.max_columns = None  # Show all columns
    pd.options.display.max_rows = None  # Show all rows
    pd.set_option('display.expand_frame_repr', False)  # Prevent line breaks to show more columns
    pd.set_option('display.max_colwidth', 1000)  # Increase the character limit per cell
    sns.set_theme()  # Set the default theme for Seaborn

def plot_insurance_interest_by_numerical_variable(df, dependent_var='response', independent_var='vintage', bins=30):
    """
    Gera gráficos de densidade, box plot e gráfico de barras proporcionais para visualizar
    a relação entre uma variável dependente e uma variável independente em relação ao interesse em seguros.

    Parameters:
    dependent_var (str): Nome da coluna da variável dependente (ex: 'response').
    independent_var (str): Nome da coluna da variável independente (ex: 'vintage').
    df (DataFrame): DataFrame contendo os dados.
    bins (int): Número de bins para o gráfico de barras proporcionais.
    """
    
    # Setting up the figure and subplots
    fig = plt.figure(figsize=(12, 12))
    gs = gridspec.GridSpec(2, 2)

    # Graph 1: Density Plot
    ax1 = fig.add_subplot(gs[0, 0])
    sns.kdeplot(data=df, x=independent_var, hue=dependent_var, fill=True, alpha=0.5, ax=ax1, legend=False)
    ax1.set_title(f'Density Plot of {independent_var} by {dependent_var}')

    # Graph 2: Box Plot
    ax2 = fig.add_subplot(gs[0, 1])
    sns.boxplot(data=df, x=dependent_var, y=independent_var, ax=ax2, hue=dependent_var, legend=False)
    ax2.set_title(f'Box Plot of {independent_var} by {dependent_var}')
    ax2.set_xlabel(dependent_var)
    ax2.set_ylabel(independent_var)

    # Graph 3: Proportional Bar Plot
    ax3 = fig.add_subplot(gs[1, :])
    # Create bins for the independent variable without adding a new column to df
    bins_labels = pd.cut(df[independent_var], bins=bins)
    # Create a DataFrame to plot proportions
    proportions = df.groupby([bins_labels, dependent_var]).size().unstack(fill_value=0)
    proportions = proportions.div(proportions.sum(axis=1), axis=0)  # Normalize to get proportions

    # Stacked bar plot
    proportions.plot(kind='bar', stacked=True, width=0.8, ax=ax3, legend=False)
    ax3.set_title(f'Proportional Bar Plot of {independent_var} by {dependent_var}')
    ax3.set_ylabel('Proportion')
    ax3.set_xlabel(f'{independent_var} Bins')

    # Definindo os ticks do eixo x
    ax3.set_xticks(range(len(proportions.index)))  # Define o número de ticks baseado no número de bins
    ax3.set_xticklabels(proportions.index.astype(str), rotation=45, ha='right')  # Rótulos dos bins

    # Create a custom legend
    handles = [
        plt.Line2D([0], [0], marker='o', color='w', label='NO Interest', markerfacecolor='C0', markersize=10),
        plt.Line2D([0], [0], marker='o', color='w', label='Interest', markerfacecolor='C1', markersize=10)
    ]
    # Add the custom legend to the figure
    fig.legend(handles=handles, title='Insurance Interest', loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=2)

    # Final adjustments and layout
    plt.tight_layout()
    plt.show()


def plot_insurance_interest_by_categorical_variable(df, target_variable, independent_variable, topk=None, figsize=(12, 12)):
    """
    Generates two bar plots for the relationship between a target variable and an independent categorical variable:
    - A stacked proportional bar chart showing the proportion of target categories within each independent variable.
    - A count bar plot showing the frequency of each category of the target variable for each independent variable.
    
    Parameters:
    ----------
    df : DataFrame
        The DataFrame containing the data to be plotted.
    target_variable : str
        The name of the target (dependent) variable, typically a binary or categorical variable.
    independent_variable : str
        The name of the independent (categorical) variable to be analyzed.
    topk : int, optional
        The number of most frequent values of the independent variable to be included in the plot. If not provided, all categories are used.
    figsize : tuple, optional
        The size of the figure, with a default of (16, 12).
    
    Returns:
    -------
    None
        Displays the plots.
    """
    
    # 1. If topk is specified, filter the 'topk' most frequent independent_variable values
    if topk:
        top_channels = df[independent_variable].value_counts().nlargest(topk).index
        df_filtered = df[df[independent_variable].isin(top_channels)]
    else:
        df_filtered = df

    # 2. Generate the crosstab table (proportions)
    aux_crosstab = pd.crosstab(df_filtered[independent_variable], df_filtered[target_variable], normalize='index')

    # 3. Generate the count table
    aux_counts = df_filtered[[target_variable, independent_variable]].value_counts().reset_index(name='count')

    # Map 0 and 1 to 'No' and 'Yes' (adjust this mapping based on your data)
    response_mapping = {0: 'No', 1: 'Yes'}
    aux_crosstab.columns = aux_crosstab.columns.map(response_mapping)
    aux_counts[target_variable] = aux_counts[target_variable].map(response_mapping)

    # Configure the plot to have two graphs, one below the other
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)

    # 4. Graph 1: Stacked bar chart (proportions)
    aux_crosstab.plot(kind='bar', stacked=True, ax=ax1)
    ax1.set_title(f'Proportional Stacked Bar Chart of {target_variable} by {"top " + str(topk) if topk else ""} {independent_variable}')
    ax1.set_xlabel(independent_variable)
    ax1.set_ylabel('Proportion')
    ax1.legend(title=target_variable, bbox_to_anchor=(1.05, 1), loc='upper left')

    # Set the x-axis labels to be horizontal if there are less than 10 bars
    if len(aux_crosstab) <= 10:
        ax1.set_xticklabels(aux_crosstab.index, rotation=0)  # Horizontal labels
    else:
        ax1.set_xticklabels(aux_crosstab.index, rotation=90)  # Vertical labels

    # 5. Graph 2: Bar chart with counts (value_counts)
    sns.barplot(data=aux_counts, x=independent_variable, y='count', hue=target_variable, order=aux_crosstab.index, ax=ax2)
    ax2.set_title(f'Count Bar Plot of {target_variable} by {"top " + str(topk) if topk else ""} {independent_variable}')
    ax2.set_xlabel(independent_variable)
    ax2.set_ylabel('Count')

    # Set the x-axis labels to be horizontal if there are less than 10 bars
    if len(aux_counts) > 10:
        ax2.set_xticklabels(ax2.get_xticklabels(), rotation=90)


    # Adjust the legend of the second graph
    ax2.legend(title=target_variable, bbox_to_anchor=(1.05, 1), loc='upper left')

    # Adjust layout to avoid overlap
    plt.tight_layout()
    plt.show()


def cramer_v(x, y):
    """
    Calculate Cramér's V statistic for categorical-categorical association.

    Cramér's V is a measure of association between two nominal variables, giving a value between 0 and 1.
    A value of 0 indicates no association between the variables, while a value of 1 indicates perfect association.

    Parameters:
    -----------
    x : array-like
        The first categorical variable.
    y : array-like
        The second categorical variable.

    Returns:
    --------
    float
        Cramér's V statistic.
        
    Notes:
    ------
    This function calculates Cramér's V with bias correction as described in Bergsma (2013).
    """
    cm = pd.crosstab(x, y).to_numpy()
    n = cm.sum()
    r, k = cm.shape

    chi2 = stats.chi2_contingency(cm)[0]
    chi2corr = max(0, chi2 - (k-1)*(r-1)/(n-1))
    
    kcorr = k - (k-1)**2/(n-1)
    rcorr = r - (r-1)**2/(n-1)
    
    v = np.sqrt((chi2corr/n) / (min(kcorr-1, rcorr-1)))

    return v