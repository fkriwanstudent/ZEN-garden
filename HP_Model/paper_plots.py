import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
import os
import openpyxl
from matplotlib.colors import LinearSegmentedColormap

def region_scenario_d(flow_file, capacity_file, demand_file, output_dir="paper"):
    """
    Create one plot for each region (Global, China, Europe, USA) showing production, capacity, and demand.

    Parameters:
    flow_file (str/Path): Path to flow conversion output data file
    capacity_file (str/Path): Path to capacity data file
    demand_file (str/Path): Path to demand data file
    output_dir (str/Path): Directory to save output plots (default: "paper")
    """
    # Set plot style
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({
        'font.size': 13,
        'axes.labelsize': 13,
        'axes.titlesize': 13,
        'xtick.labelsize': 13,
        'ytick.labelsize': 13,
        'legend.fontsize': 13,
        'figure.titlesize': 15
    })

    # Define color palette for different data types and scenarios
    colors = {
        'capacity_s1': "#b24020",  # Light purple
        'capacity_s2': "#004182",  # Darker purple
        'production_s1': "#f5987e",  # Light green
        'production_s2': "#4185be",  # Darker green
        'demand': "#236133"  # Reddish
    }

    # Ensure files are Path objects
    flow_file = Path(flow_file)
    capacity_file = Path(capacity_file)
    demand_file = Path(demand_file)

    # Create output directory if it doesn't exist
    output_dir = Path(output_dir)
    if not output_dir.exists():
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    # Read data files
    flow_df = pd.read_csv(flow_file)
    capacity_df = pd.read_csv(capacity_file)

    # Process demand file (special format where each country is a column)
    demand_df = pd.read_csv(demand_file)

    # Transform demand data from wide to long format
    demand_long = pd.melt(
        demand_df,
        id_vars=['year'],
        var_name='node',
        value_name='demand'
    )

    # Convert kW to GW for demand
    demand_long['demand'] = demand_long['demand'] / 1000000

    # Filter for HP assembly technology in flow data
    flow_df = flow_df[flow_df['technology'] == 'HP_assembly']

    # Filter for HP assembly in capacity data
    capacity_df = capacity_df[capacity_df['technology'] == 'HP_assembly']

    # Define European countries and mapping for regions
    european_countries = ['CZE', 'AUT', 'ITA', 'DEU', 'ROE', 'EUR']

    # Define region mapping function
    def map_to_region(node):
        if node in european_countries:
            return 'EUR'
        elif node in ['CHN', 'USA']:
            return node
        else:
            return 'Global'

    # Apply region mapping to dataframes
    flow_df['region'] = flow_df['node'].apply(map_to_region)
    if 'location' in capacity_df.columns:
        capacity_df['region'] = capacity_df['location'].apply(map_to_region)
    else:
        capacity_df['region'] = capacity_df['node'].apply(map_to_region)

    demand_long['region'] = demand_long['node'].apply(map_to_region)

    # Get scenario columns
    flow_s1_col = [col for col in flow_df.columns if col == 'value_scenario_'][0]
    flow_s2_col = [col for col in flow_df.columns if col == 'value_scenario_S1'][0]

    capacity_s1_col = [col for col in capacity_df.columns if col == 'value_scenario_'][0]
    capacity_s2_col = [col for col in capacity_df.columns if col == 'value_scenario_S1'][0]

    # Convert time indices to actual years
    if 'time_operation' in flow_df.columns:
        flow_df['year'] = flow_df['time_operation'] + 2022

    if 'year' in capacity_df.columns and capacity_df['year'].max() < 2000:
        # If year values are small, they likely need to be offset
        capacity_df['year'] = capacity_df['year'] + 2022

    # Aggregate data by region and year
    flow_by_region = flow_df.groupby(['region', 'year']).agg({
        flow_s1_col: 'sum',
        flow_s2_col: 'sum'
    }).reset_index()

    capacity_by_region = capacity_df.groupby(['region', 'year']).agg({
        capacity_s1_col: 'sum',
        capacity_s2_col: 'sum'
    }).reset_index()

    demand_by_region = demand_long.groupby(['region', 'year']).agg({
        'demand': 'sum'
    }).reset_index()

    # Filter data to only include years until 2035
    flow_by_region = flow_by_region[flow_by_region['year'] <= 2035]
    capacity_by_region = capacity_by_region[capacity_by_region['year'] <= 2035]
    demand_by_region = demand_by_region[demand_by_region['year'] <= 2035]


    # Create 4-column plot for each region
    regions = ['Global', 'CHN', 'EUR', 'USA']
    fig, axes = plt.subplots(1, 4, figsize=(20, 6))

    for i, region in enumerate(regions):
        ax = axes[i]

        # Get data for this region
        region_capacity = capacity_by_region[capacity_by_region['region'] == region]
        region_flow = flow_by_region[flow_by_region['region'] == region]
        region_demand = demand_by_region[demand_by_region['region'] == region]

        # Plot production as bars
        bar_width = 0.3
        if not region_flow.empty:
            # Plot production as bars using actual year values
            ax.bar(region_flow['year'] - bar_width / 2, region_flow[flow_s1_col],
                   width=bar_width, color=colors['production_s1'], alpha=0,
                   label='Production (BAU)')
            ax.bar(region_flow['year'] + bar_width / 2, region_flow[flow_s2_col],
                   width=bar_width, color=colors['production_s2'], alpha=0,
                   label='Production (NZE)')

        # Plot capacity as dashed lines
        if not region_capacity.empty:
            ax.scatter(region_capacity['year'], region_capacity[capacity_s1_col],
                    color=colors['capacity_s1'], linewidth=2, linestyle='--',alpha=0,
                     label='Capacity (BAU)')
            ax.scatter(region_capacity['year'], region_capacity[capacity_s2_col],
                    color=colors['capacity_s2'], linewidth=2, linestyle='--',alpha=0,
                    label='Capacity (NZE)')

        # Plot demand as scatter points
        if not region_demand.empty:
            ax.scatter(region_demand['year'], region_demand['demand'],
                       color=colors['demand'], s=80, marker='^',
                       edgecolors='white', linewidth=1, label='Demand')

        # Customize plot
        ax.set_title(f"{region}", fontsize=14, fontweight='bold')
        ax.set_ylabel('GW', fontsize=13)
        ax.set_xlabel('Year', fontsize=13)

        # Set x-axis to show years properly
        all_years = []
        if not region_capacity.empty:
            all_years.extend(region_capacity['year'].tolist())
        if not region_flow.empty:
            all_years.extend(region_flow['year'].tolist())
        if not region_demand.empty:
            all_years.extend(region_demand['year'].tolist())

        if all_years:
            unique_years = sorted(set(all_years))
            ax.set_xticks(unique_years)
            ax.tick_params(axis='x', rotation=90)

        ax.grid(True, which="both", ls="-", alpha=0)

    # Add a single legend below the plots
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels,
               loc='lower center',
               bbox_to_anchor=(0.5, -0.1),
               ncol=5, frameon=False, fontsize=13)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)  # Make room for the legend

    # Save the plot
    output_path = output_dir / 'regional_comparison_d.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    return fig

def region_scenario_dc(flow_file, capacity_file, demand_file, output_dir="paper"):
    """
    Create one plot for each region (Global, China, Europe, USA) showing production, capacity, and demand.

    Parameters:
    flow_file (str/Path): Path to flow conversion output data file
    capacity_file (str/Path): Path to capacity data file
    demand_file (str/Path): Path to demand data file
    output_dir (str/Path): Directory to save output plots (default: "paper")
    """
    # Set plot style
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({
        'font.size': 13,
        'axes.labelsize': 13,
        'axes.titlesize': 13,
        'xtick.labelsize': 13,
        'ytick.labelsize': 13,
        'legend.fontsize': 13,
        'figure.titlesize': 15
    })

    # Define color palette for different data types and scenarios
    colors = {
        'capacity_s1': "#b24020",  # Light purple
        'capacity_s2': "#004182",  # Darker purple
        'production_s1': "#f5987e",  # Light green
        'production_s2': "#4185be",  # Darker green
        'demand': "#236133"  # Reddish
    }

    # Ensure files are Path objects
    flow_file = Path(flow_file)
    capacity_file = Path(capacity_file)
    demand_file = Path(demand_file)

    # Create output directory if it doesn't exist
    output_dir = Path(output_dir)
    if not output_dir.exists():
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    # Read data files
    flow_df = pd.read_csv(flow_file)
    capacity_df = pd.read_csv(capacity_file)

    # Process demand file (special format where each country is a column)
    demand_df = pd.read_csv(demand_file)

    # Transform demand data from wide to long format
    demand_long = pd.melt(
        demand_df,
        id_vars=['year'],
        var_name='node',
        value_name='demand'
    )

    # Convert kW to GW for demand
    demand_long['demand'] = demand_long['demand'] / 1000000

    # Filter for HP assembly technology in flow data
    flow_df = flow_df[flow_df['technology'] == 'HP_assembly']

    # Filter for HP assembly in capacity data
    capacity_df = capacity_df[capacity_df['technology'] == 'HP_assembly']

    # Define European countries and mapping for regions
    european_countries = ['CZE', 'AUT', 'ITA', 'DEU', 'ROE', 'EUR']

    # Define region mapping function
    def map_to_region(node):
        if node in european_countries:
            return 'EUR'
        elif node in ['CHN', 'USA']:
            return node
        else:
            return 'Global'

    # Apply region mapping to dataframes
    flow_df['region'] = flow_df['node'].apply(map_to_region)
    if 'location' in capacity_df.columns:
        capacity_df['region'] = capacity_df['location'].apply(map_to_region)
    else:
        capacity_df['region'] = capacity_df['node'].apply(map_to_region)

    demand_long['region'] = demand_long['node'].apply(map_to_region)

    # Get scenario columns
    flow_s1_col = [col for col in flow_df.columns if col == 'value_scenario_'][0]
    flow_s2_col = [col for col in flow_df.columns if col == 'value_scenario_S1'][0]

    capacity_s1_col = [col for col in capacity_df.columns if col == 'value_scenario_'][0]
    capacity_s2_col = [col for col in capacity_df.columns if col == 'value_scenario_S1'][0]

    # Convert time indices to actual years
    if 'time_operation' in flow_df.columns:
        flow_df['year'] = flow_df['time_operation'] + 2022

    if 'year' in capacity_df.columns and capacity_df['year'].max() < 2000:
        # If year values are small, they likely need to be offset
        capacity_df['year'] = capacity_df['year'] + 2022

    # Aggregate data by region and year
    flow_by_region = flow_df.groupby(['region', 'year']).agg({
        flow_s1_col: 'sum',
        flow_s2_col: 'sum'
    }).reset_index()

    capacity_by_region = capacity_df.groupby(['region', 'year']).agg({
        capacity_s1_col: 'sum',
        capacity_s2_col: 'sum'
    }).reset_index()

    demand_by_region = demand_long.groupby(['region', 'year']).agg({
        'demand': 'sum'
    }).reset_index()

    # Filter data to only include years until 2035
    flow_by_region = flow_by_region[flow_by_region['year'] <= 2035]
    capacity_by_region = capacity_by_region[capacity_by_region['year'] <= 2035]
    demand_by_region = demand_by_region[demand_by_region['year'] <= 2035]


    # Create 4-column plot for each region
    regions = ['Global', 'CHN', 'EUR', 'USA']
    fig, axes = plt.subplots(1, 4, figsize=(20, 6))

    for i, region in enumerate(regions):
        ax = axes[i]

        # Get data for this region
        region_capacity = capacity_by_region[capacity_by_region['region'] == region]
        region_flow = flow_by_region[flow_by_region['region'] == region]
        region_demand = demand_by_region[demand_by_region['region'] == region]

        # Plot production as bars with years on x-axis
        bar_width = 0.3
        if not region_flow.empty:
            # Plot production as bars using actual year values
            ax.bar(region_flow['year'] - bar_width / 2, region_flow[flow_s1_col],
                   width=bar_width, color=colors['production_s1'], alpha=0,
                   label='Production (BAU)')
            ax.bar(region_flow['year'] + bar_width / 2, region_flow[flow_s2_col],
                   width=bar_width, color=colors['production_s2'], alpha=0,
                   label='Production (NZE)')

        # Plot capacity as dashed lines
        if not region_capacity.empty:
            ax.plot(region_capacity['year'], region_capacity[capacity_s1_col],
                    color=colors['capacity_s1'], linewidth=2, linestyle='--',
                    marker='o', markersize=6, label='Capacity (BAU)')
            ax.plot(region_capacity['year'], region_capacity[capacity_s2_col],
                    color=colors['capacity_s2'], linewidth=2, linestyle='--',
                    marker='o', markersize=6, label='Capacity (NZE)')

        # Plot demand as scatter points
        if not region_demand.empty:
            ax.scatter(region_demand['year'], region_demand['demand'],
                       color=colors['demand'], s=80, marker='^',
                       edgecolors='white', linewidth=1, label='Demand')

        # Customize plot
        ax.set_title(f"{region}", fontsize=14, fontweight='bold')
        ax.set_ylabel('GW', fontsize=13)
        ax.set_xlabel('Year', fontsize=13)

        # Set x-axis to show years properly
        all_years = []
        if not region_capacity.empty:
            all_years.extend(region_capacity['year'].tolist())
        if not region_flow.empty:
            all_years.extend(region_flow['year'].tolist())
        if not region_demand.empty:
            all_years.extend(region_demand['year'].tolist())

        if all_years:
            unique_years = sorted(set(all_years))
            ax.set_xticks(unique_years)
            ax.tick_params(axis='x', rotation=90)

        ax.grid(True, which="both", ls="-", alpha=0)

    # Add a single legend below the plots
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels,
               loc='lower center',
               bbox_to_anchor=(0.5, -0.1),
               ncol=5, frameon=False, fontsize=13)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)  # Make room for the legend

    # Save the plot
    output_path = output_dir / 'regional_comparison_dc.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    return fig

def region_scenario_dcp(flow_file, capacity_file, demand_file, output_dir="paper"):
    """
    Create one plot for each region (Global, China, Europe, USA) showing production, capacity, and demand.

    Parameters:
    flow_file (str/Path): Path to flow conversion output data file
    capacity_file (str/Path): Path to capacity data file
    demand_file (str/Path): Path to demand data file
    output_dir (str/Path): Directory to save output plots (default: "paper")
    """
    # Set plot style
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({
        'font.size': 13,
        'axes.labelsize': 13,
        'axes.titlesize': 13,
        'xtick.labelsize': 13,
        'ytick.labelsize': 13,
        'legend.fontsize': 13,
        'figure.titlesize': 15
    })

    # Define color palette for different data types and scenarios
    colors = {
        'capacity_s1': "#b24020",  # Light purple
        'capacity_s2': "#004182",  # Darker purple
        'production_s1': "#f5987e",  # Light green
        'production_s2': "#4185be",  # Darker green
        'demand': "#236133"  # Reddish
    }

    # Ensure files are Path objects
    flow_file = Path(flow_file)
    capacity_file = Path(capacity_file)
    demand_file = Path(demand_file)

    # Create output directory if it doesn't exist
    output_dir = Path(output_dir)
    if not output_dir.exists():
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    # Read data files
    flow_df = pd.read_csv(flow_file)
    capacity_df = pd.read_csv(capacity_file)

    # Process demand file (special format where each country is a column)
    demand_df = pd.read_csv(demand_file)

    # Transform demand data from wide to long format
    demand_long = pd.melt(
        demand_df,
        id_vars=['year'],
        var_name='node',
        value_name='demand'
    )

    # Convert kW to GW for demand
    demand_long['demand'] = demand_long['demand'] / 1000000

    # Filter for HP assembly technology in flow data
    flow_df = flow_df[flow_df['technology'] == 'HP_assembly']

    # Filter for HP assembly in capacity data
    capacity_df = capacity_df[capacity_df['technology'] == 'HP_assembly']

    # Define European countries and mapping for regions
    european_countries = ['CZE', 'AUT', 'ITA', 'DEU', 'ROE', 'EUR']

    # Define region mapping function
    def map_to_region(node):
        if node in european_countries:
            return 'EUR'
        elif node in ['CHN', 'USA']:
            return node
        else:
            return 'Global'

    # Apply region mapping to dataframes
    flow_df['region'] = flow_df['node'].apply(map_to_region)
    if 'location' in capacity_df.columns:
        capacity_df['region'] = capacity_df['location'].apply(map_to_region)
    else:
        capacity_df['region'] = capacity_df['node'].apply(map_to_region)

    demand_long['region'] = demand_long['node'].apply(map_to_region)

    # Get scenario columns
    flow_s1_col = [col for col in flow_df.columns if col == 'value_scenario_'][0]
    flow_s2_col = [col for col in flow_df.columns if col == 'value_scenario_S1'][0]

    capacity_s1_col = [col for col in capacity_df.columns if col == 'value_scenario_'][0]
    capacity_s2_col = [col for col in capacity_df.columns if col == 'value_scenario_S1'][0]

    # Convert time indices to actual years
    if 'time_operation' in flow_df.columns:
        flow_df['year'] = flow_df['time_operation'] + 2022

    if 'year' in capacity_df.columns and capacity_df['year'].max() < 2000:
        # If year values are small, they likely need to be offset
        capacity_df['year'] = capacity_df['year'] + 2022

    # Aggregate data by region and year
    flow_by_region = flow_df.groupby(['region', 'year']).agg({
        flow_s1_col: 'sum',
        flow_s2_col: 'sum'
    }).reset_index()

    capacity_by_region = capacity_df.groupby(['region', 'year']).agg({
        capacity_s1_col: 'sum',
        capacity_s2_col: 'sum'
    }).reset_index()

    demand_by_region = demand_long.groupby(['region', 'year']).agg({
        'demand': 'sum'
    }).reset_index()

    # Filter data to only include years until 2035
    flow_by_region = flow_by_region[flow_by_region['year'] <= 2035]
    capacity_by_region = capacity_by_region[capacity_by_region['year'] <= 2035]
    demand_by_region = demand_by_region[demand_by_region['year'] <= 2035]


    # Create 4-column plot for each region
    regions = ['Global', 'CHN', 'EUR', 'USA']
    fig, axes = plt.subplots(1, 4, figsize=(20, 6))

    for i, region in enumerate(regions):
        ax = axes[i]

        # Get data for this region
        region_capacity = capacity_by_region[capacity_by_region['region'] == region]
        region_flow = flow_by_region[flow_by_region['region'] == region]
        region_demand = demand_by_region[demand_by_region['region'] == region]

        # Plot production as bars with years on x-axis
        bar_width = 0.3
        if not region_flow.empty:
            # Plot production as bars using actual year values
            ax.bar(region_flow['year'] - bar_width / 2, region_flow[flow_s1_col],
                   width=bar_width, color=colors['production_s1'], alpha=0.8,
                   label='Production (BAU)')
            ax.bar(region_flow['year'] + bar_width / 2, region_flow[flow_s2_col],
                   width=bar_width, color=colors['production_s2'], alpha=0.8,
                   label='Production (NZE)')

        # Plot capacity as dashed lines
        if not region_capacity.empty:
            ax.plot(region_capacity['year'], region_capacity[capacity_s1_col],
                    color=colors['capacity_s1'], linewidth=2, linestyle='--',
                    marker='o', markersize=6, label='Capacity (BAU)')
            ax.plot(region_capacity['year'], region_capacity[capacity_s2_col],
                    color=colors['capacity_s2'], linewidth=2, linestyle='--',
                    marker='o', markersize=6, label='Capacity (NZE)')

        # Plot demand as scatter points
        if not region_demand.empty:
            ax.scatter(region_demand['year'], region_demand['demand'],
                       color=colors['demand'], s=80, marker='^',
                       edgecolors='white', linewidth=1, label='Demand')

        # Customize plot
        ax.set_title(f"{region}", fontsize=14, fontweight='bold')
        ax.set_ylabel('GW', fontsize=13)
        ax.set_xlabel('Year', fontsize=13)

        # Set x-axis to show years properly
        all_years = []
        if not region_capacity.empty:
            all_years.extend(region_capacity['year'].tolist())
        if not region_flow.empty:
            all_years.extend(region_flow['year'].tolist())
        if not region_demand.empty:
            all_years.extend(region_demand['year'].tolist())

        if all_years:
            unique_years = sorted(set(all_years))
            ax.set_xticks(unique_years)
            ax.tick_params(axis='x', rotation=90)

        ax.grid(True, which="both", ls="-", alpha=0)

    # Add a single legend below the plots
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels,
               loc='lower center',
               bbox_to_anchor=(0.5, -0.1),
               ncol=5, frameon=False, fontsize=13)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)  # Make room for the legend

    # Save the plot
    output_path = output_dir / 'regional_comparison_dcp.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    return fig


def self_suff_scenario(flow_file, demand_file, output_dir="paper"):
    # Set plot style
    plt.rcParams.update({
        'font.size': 9,
        'axes.labelsize': 9,
        'axes.titlesize': 10,
        'xtick.labelsize': 8,
        'ytick.labelsize': 9,
        'figure.titlesize': 12
    })

    # Ensure files are Path objects
    flow_file = Path(flow_file)
    demand_file = Path(demand_file)

    # Create output directory if it doesn't exist
    output_dir = Path(output_dir)
    if not output_dir.exists():
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    # Read data files
    flow_df = pd.read_csv(flow_file)

    # Process demand file (special format where each country is a column)
    demand_df = pd.read_csv(demand_file)

    # Transform demand data from wide to long format
    demand_long = pd.melt(
        demand_df,
        id_vars=['year'],
        var_name='node',
        value_name='demand'
    )

    # Convert kW to GW for demand
    demand_long['demand'] = demand_long['demand'] / 1000000

    # Define technologies to analyze
    technologies = ['HP_assembly', 'HEX_manufacturing', 'Compressor_manufacturing']

    # Define European countries and mapping for regions
    european_countries = ['CZE', 'AUT', 'ITA', 'DEU', 'ROE', 'EUR']

    # Define region mapping function
    def map_to_region(node):
        if node in european_countries:
            return 'EUR'
        elif node in ['CHN', 'USA']:
            return node
        else:
            return 'Global'

    # Apply region mapping to dataframes
    flow_df['region'] = flow_df['node'].apply(map_to_region)
    demand_long['region'] = demand_long['node'].apply(map_to_region)

    # Get all scenario columns
    scenario_cols = [col for col in flow_df.columns if col.startswith('value_scenario')]
    print(f"Found scenario columns: {scenario_cols}")

    # Create scenario labels
    scenario_labels = []
    for col in scenario_cols:
        if col == 'value_scenario_':
            scenario_labels.append('BAU')
        elif 'S1' in col:
            scenario_labels.append('NZE')
        elif 'S2' in col:
            scenario_labels.append('Europe 40%')
        elif 'S3' in col:
            scenario_labels.append('Recycling')
        elif 'S4' in col:
            scenario_labels.append('Diffusion')
        elif 'S5' in col:
            scenario_labels.append('Tariffs')
        else:
            scenario_labels.append(col.replace('value_scenario_', ''))

    # Convert time indices to actual years
    if 'time_operation' in flow_df.columns:
        flow_df['year'] = flow_df['time_operation'] + 2022

    # Define specific years to analyze
    target_years = [2025, 2030, 2035]
    regions = ['Global', 'CHN', 'EUR', 'USA']

    # Aggregate data by technology, region and year
    flow_by_tech_region = flow_df.groupby(['technology', 'region', 'year']).agg({
        col: 'sum' for col in scenario_cols
    }).reset_index()

    demand_by_region = demand_long.groupby(['region', 'year']).agg({
        'demand': 'sum'
    }).reset_index()

    # Create one plot for each region
    for region in regions:
        print(f"Creating heatmap for region: {region}")

        # Create subplots for each technology (1 row, 3 columns)
        fig, axes = plt.subplots(1, 3, figsize=(8, 4))

        for i, tech in enumerate(technologies):
            ax = axes[i]

            # Filter data for this technology and region
            tech_region_flow = flow_by_tech_region[
                (flow_by_tech_region['technology'] == tech) &
                (flow_by_tech_region['region'] == region)
                ]
            region_demand = demand_by_region[demand_by_region['region'] == region]

            # Create heatmap data matrix: scenarios (rows) x years (columns)
            heatmap_data = []

            for scenario_col in scenario_cols:
                row_data = []
                for year in target_years:
                    # Get production data for this year
                    year_flow = tech_region_flow[tech_region_flow['year'] == year]
                    year_demand = region_demand[region_demand['year'] == year]

                    if not year_flow.empty and not year_demand.empty:
                        production = year_flow[scenario_col].iloc[0] if len(year_flow) > 0 else 0
                        demand = year_demand['demand'].iloc[0] if len(year_demand) > 0 else 1
                        self_suff = min(production / demand, 1) if demand > 0 else 0
                    else:
                        self_suff = 0
                    row_data.append(self_suff)
                heatmap_data.append(row_data)

            # Convert to numpy array
            heatmap_array = np.array(heatmap_data)

            # Use viridis colormap
            custom_cmap = 'viridis'

            # Create heatmap
            im = ax.imshow(heatmap_array, cmap=custom_cmap, aspect='auto', vmin=0, vmax=1)

            # Set ticks and labels
            ax.set_xticks(range(len(target_years)))
            ax.set_xticklabels(target_years, rotation=0)
            ax.set_yticks(range(len(scenario_labels)))

            # Only show y-tick labels for the first subplot
            if i == 0:
                ax.set_yticklabels(scenario_labels)
            else:
                ax.set_yticklabels([])
            # Add text annotations showing the values as percentages
            for row in range(len(scenario_labels)):
                for col in range(len(target_years)):
                    value = heatmap_array[row, col]
                    percentage = value * 100  # Convert to percentage
                    color = 'white' if value < 0.5 else 'black'
                    ax.text(col, row, f'{percentage:.0f}%', ha='center', va='center',
                            color=color, fontweight='bold', fontsize=8)

            # Customize subplot
            ax.set_title(f'{tech.replace("_", " ")}', fontsize=10, fontweight='bold')
            ax.set_xlabel('Year', fontsize=9)
            if i == 0:  # Only leftmost gets y-label
                ax.set_ylabel('Scenario', fontsize=9)

            ax.grid(False)

            # Add grid lines as borders of color blocks
            for x in range(len(target_years) + 1):
                ax.axvline(x - 0.5, color='white', linewidth=1)
            for y in range(len(scenario_labels) + 1):
                ax.axhline(y - 0.5, color='white', linewidth=1)

        # Add a shared colorbar
        fig.subplots_adjust(right=0.85)
        cbar_ax = fig.add_axes([0.87, 0.15, 0.02, 0.7])
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.set_label('Self-Sufficiency Ratio\n(Production/Demand)', rotation=270, labelpad=15, fontsize=9)

        # Add main title for the region
        fig.suptitle(f'Self-Sufficiency for Region: {region}',
                     fontsize=12, fontweight='bold', y=0.95)

        #plt.tight_layout()
        plt.subplots_adjust(top=0.85, right=0.85)

        # Save the plot for this region
        output_path = output_dir / f'self_sufficiency_region_{region}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Heatmap for region {region} saved to: {output_path}")

        plt.show()  # Display the plot

    # Create and save comprehensive summary data
    print("Creating comprehensive summary data...")
    all_summary_data = {}

    for region in regions:
        region_data = {}
        for tech in technologies:
            tech_region_flow = flow_by_tech_region[
                (flow_by_tech_region['technology'] == tech) &
                (flow_by_tech_region['region'] == region)
                ]
            region_demand = demand_by_region[demand_by_region['region'] == region]

            tech_data = []
            for year in target_years:
                year_flow = tech_region_flow[tech_region_flow['year'] == year]
                year_demand = region_demand[region_demand['year'] == year]

                row_dict = {'Year': year}
                for j, scenario_col in enumerate(scenario_cols):
                    if not year_flow.empty and not year_demand.empty:
                        production = year_flow[scenario_col].iloc[0] if len(year_flow) > 0 else 0
                        demand = year_demand['demand'].iloc[0] if len(year_demand) > 0 else 1
                        self_suff = min(production / demand, 1) if demand > 0 else 0
                    else:
                        self_suff = 0
                    row_dict[scenario_labels[j]] = self_suff
                tech_data.append(row_dict)

            region_data[tech] = pd.DataFrame(tech_data)
        all_summary_data[region] = region_data

    # Save comprehensive summary data
    data_output_path = output_dir / 'self_sufficiency_comprehensive_summary.xlsx'
    with pd.ExcelWriter(data_output_path) as writer:
        for region, region_data in all_summary_data.items():
            for tech, df in region_data.items():
                sheet_name = f'{region}_{tech}'
                df.to_excel(writer, sheet_name=sheet_name, index=False)
    print(f"Comprehensive summary data saved to: {data_output_path}")

    print("All regional heatmaps completed!")
    return None


def lcohp_region_scenario(lcohp_file, output_dir='paper'):
    # Set plot style
    plt.rcParams.update({
        'font.size': 10,
        'axes.labelsize': 10,
        'axes.titlesize': 10,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'figure.titlesize': 12
    })

    # Ensure files are Path objects
    lcohp_file = Path(lcohp_file)
    output_dir = Path(output_dir)

    # Create output directory if it doesn't exist
    if not output_dir.exists():
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    # Read LCOHP data
    lcohp_df = pd.read_csv(lcohp_file, index_col=0)

    # Rename scenario columns with proper labels
    scenario_mapping = {
        '': 'BAU',  # First column (empty name) is BAU
        'Unnamed: 1': 'BAU',  # Handle pandas unnamed column
        'S1': 'NZE',
        'S2': 'Europe 40%',
        'S3': 'Recycling',
        'S4': 'Diffusion',
        'S5': 'Tariffs'
    }

    # Apply the mapping to column names
    lcohp_df.columns = [scenario_mapping.get(col, col) for col in lcohp_df.columns]

    # Debug: print original and mapped column names
    print(f"Original columns: {list(pd.read_csv(lcohp_file, index_col=0).columns)}")
    print(f"Mapped columns: {list(lcohp_df.columns)}")

    print(f"LCOHP data shape: {lcohp_df.shape}")
    print(f"Regions: {list(lcohp_df.index)}")
    print(f"Scenarios: {list(lcohp_df.columns)}")

    # Define manual colors for scenarios
    scenario_colors = {
        'BAU': '#dd1c77',  #
        'NZE': '#2b8cbe',  #
        'Europe 40%': '#fbbc05',  #
        'Recycling': '#34a853',  #
        'Diffusion': '#8762c9',  #
        'Tariffs': '#f96302'  #
    }

    # Create figure
    fig, ax = plt.subplots(figsize=(7, 5))

    # Prepare data for plotting
    for scenario in lcohp_df.columns:
        regions_numeric = range(len(lcohp_df.index))
        values = lcohp_df[scenario].values

        # Create scatter plot with hollow circles (colored borders only)
        ax.scatter(regions_numeric, values,
                   facecolors='none',  # No fill - hollow circles
                   edgecolors=scenario_colors[scenario],  # Colored borders
                   label=scenario,
                   s=70,
                   alpha=0.9,
                   marker='D',
                   linewidth=1)  # Border thickness

        # # Add value labels on each point
        # for i, (region, value) in enumerate(zip(lcohp_df.index, values)):
        #     ax.text(i, value + 5, f'{value:.0f}',
        #             ha='center', va='bottom',
        #             fontsize=8, fontweight='bold',
        #             color=scenario_colors[scenario])

    # Customize axes
    ax.set_xticks(range(len(lcohp_df.index)))
    ax.set_xticklabels(lcohp_df.index, rotation=0, ha='right')

    # Labels and title
    ax.set_xlabel('Region', fontsize=12, fontweight='bold')
    ax.set_ylabel('LCOHP (Euro)', fontsize=12, fontweight='bold')
    ax.set_title('LCOHP (Levelized Cost of Heat Pump) by Region and Scenario',
                 fontsize=14, fontweight='bold', pad=20)

    # Add legend
    ax.legend(title='Scenario',
              loc='upper right',
              frameon=True,
              fontsize=11,
              title_fontsize=12,
              fancybox=True,
              framealpha=0.9)

    # Add grid for better readability
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    # Set y-axis limits with some padding
    y_min, y_max = lcohp_df.values.min(), lcohp_df.values.max()
    y_padding = (y_max - y_min) * 0.1
    ax.set_ylim(y_min - y_padding, y_max + y_padding)

    plt.tight_layout()

    # Save the plot
    output_path = output_dir / 'lcohp_region_scenario_scatter.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"LCOHP scatter plot saved to: {output_path}")

    # Create summary statistics by scenario
    summary_stats = {}
    for scenario in lcohp_df.columns:
        values = lcohp_df[scenario].values
        summary_stats[scenario] = {
            'min': values.min(),
            'max': values.max(),
            'mean': values.mean(),
            'std': values.std(),
            'median': np.median(values)
        }

    print("\nLCOHP Summary Statistics by Scenario:")
    for scenario, stats in summary_stats.items():
        print(f"\n{scenario}:")
        for key, value in stats.items():
            print(f"  {key}: {value:.2f}")

    # Save detailed data
    # Transform data to long format for easier analysis
    data_long = []
    for region in lcohp_df.index:
        for scenario in lcohp_df.columns:
            data_long.append({
                'Region': region,
                'Scenario': scenario,
                'LCOHP_Value': lcohp_df.loc[region, scenario],
                'Color': scenario_colors[scenario]
            })

    summary_df = pd.DataFrame(data_long)
    data_output_path = output_dir / 'lcohp_scatter_data.csv'
    summary_df.to_csv(data_output_path, index=False)
    print(f"Summary data saved to: {data_output_path}")

    plt.show()

    return fig, summary_stats


if __name__ == "__main__":
    flow_file = "./parameter_results/flow_conversion_output/flow_conversion_output_scenarios.csv"
    capacity_file = "./parameter_results/capacity/capacity_scenarios.csv"
    demand_file = Path(__file__).parent.parent/'ZEN-Model_HP/set_carriers/HP/demand_yearly_variation.csv'
    lcohp_file = "./LCOHP/lcohp_scenario_comparison.csv"
    # Create the plots
    region_scenario_d(flow_file, capacity_file, demand_file)
    region_scenario_dc(flow_file, capacity_file, demand_file)
    region_scenario_dcp(flow_file, capacity_file, demand_file)
    self_suff_scenario(flow_file, demand_file)
    lcohp_region_scenario(lcohp_file)