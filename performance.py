import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path

# Set style for better visualizations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

# Read the Excel file - Summary sheet
file_path = r"C:\Users\ASUS\Desktop\research\DQN_adaptive\results\dqn_advanced_patterns_20260226_220100.xlsx"
df = pd.read_excel(file_path, sheet_name='Summary')

# Display basic info
print("Dataset shape:", df.shape)
print("\nFirst few rows:")
print(df.head())
print("\nColumn names:", df.columns.tolist())
print("\nUnique patterns:", df['Pattern'].nunique())
print("\nMetrics:", df['Metric'].unique())

# Create output directory for plots
output_dir = Path('dqn_analysis_plots')
output_dir.mkdir(exist_ok=True)

# 1. Bar Chart: Improvement % by Pattern for each Metric
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

metrics = df['Metric'].unique()
for idx, metric in enumerate(metrics):
    if idx < len(axes):
        metric_data = df[df['Metric'] == metric].sort_values('Improvement %', ascending=False)
        ax = axes[idx]
        bars = ax.bar(range(len(metric_data)), metric_data['Improvement %'])
        ax.set_xticks(range(len(metric_data)))
        ax.set_xticklabels(metric_data['Pattern'], rotation=45, ha='right')
        ax.set_title(f'Improvement % - {metric}')
        ax.set_ylabel('Improvement %')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        # Color bars based on positive/negative
        for bar, val in zip(bars, metric_data['Improvement %']):
            bar.set_color('green' if val > 0 else 'red')
        
        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars, metric_data['Improvement %'])):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                   f'{val:.1f}%', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig(output_dir / '1_improvement_by_metric.png', dpi=300, bbox_inches='tight')
plt.show()

# 2. Heatmap: Improvement % across all Patterns and Metrics
pivot_improvement = df.pivot(index='Pattern', columns='Metric', values='Improvement %')

plt.figure(figsize=(14, 10))
sns.heatmap(pivot_improvement, annot=True, fmt='.1f', cmap='RdYlGn', center=0,
            linewidths=1, cbar_kws={'label': 'Improvement %'})
plt.title('Improvement % Heatmap Across All Patterns and Metrics', fontsize=16)
plt.tight_layout()
plt.savefig(output_dir / '2_improvement_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()

# 3. Comparison plots for each metric (Baseline vs DQN)
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

for idx, metric in enumerate(metrics):
    if idx < len(axes):
        metric_data = df[df['Metric'] == metric].copy()
        metric_data = metric_data.sort_values('Pattern')
        
        ax = axes[idx]
        x = np.arange(len(metric_data))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, metric_data['Baseline'], width, label='Baseline', alpha=0.8)
        bars2 = ax.bar(x + width/2, metric_data['DQN'], width, label='DQN', alpha=0.8)
        
        ax.set_xlabel('Pattern')
        ax.set_ylabel(metric)
        ax.set_title(f'{metric}: Baseline vs DQN')
        ax.set_xticks(x)
        ax.set_xticklabels(metric_data['Pattern'], rotation=45, ha='right')
        ax.legend()
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig(output_dir / '3_baseline_vs_dqn_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# 4. Radar chart for top performing patterns
# Select top 5 patterns based on average improvement
df_avg_improvement = df.groupby('Pattern')['Improvement %'].mean().sort_values(ascending=False)
top_patterns = df_avg_improvement.head(5).index.tolist()

# Prepare data for radar chart
metrics_list = df['Metric'].unique()
n_metrics = len(metrics_list)

# Create radar chart
fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
angles += angles[:1]  # Close the loop

for pattern in top_patterns:
    values = []
    for metric in metrics_list:
        val = df[(df['Pattern'] == pattern) & (df['Metric'] == metric)]['Improvement %'].values[0]
        values.append(val)
    values += values[:1]  # Close the loop
    
    ax.plot(angles, values, 'o-', linewidth=2, label=pattern)
    ax.fill(angles, values, alpha=0.25)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(metrics_list)
ax.set_ylim(0, 100)
ax.set_title('Top 5 Patterns - Improvement % Radar Chart', size=16, pad=20)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
ax.grid(True)

plt.tight_layout()
plt.savefig(output_dir / '4_radar_chart_top_patterns.png', dpi=300, bbox_inches='tight')
plt.show()

# 5. Scatter plot matrix of metrics (Baseline values)
baseline_data = df.pivot(index='Pattern', columns='Metric', values='Baseline').reset_index()

fig, axes = plt.subplots(2, 2, figsize=(14, 12))
axes = axes.flatten()

metric_pairs = [('Queue Length', 'Delay'), ('Speed', 'Travel Time'), 
                ('Throughput', 'Delay'), ('Queue Length', 'Throughput')]

for idx, (metric1, metric2) in enumerate(metric_pairs):
    ax = axes[idx]
    scatter = ax.scatter(baseline_data[metric1], baseline_data[metric2], 
                        c=range(len(baseline_data)), cmap='viridis', s=100)
    ax.set_xlabel(metric1)
    ax.set_ylabel(metric2)
    ax.set_title(f'{metric1} vs {metric2} (Baseline)')
    
    # Add pattern labels
    for i, pattern in enumerate(baseline_data['Pattern']):
        ax.annotate(pattern[:10], (baseline_data[metric1].iloc[i], baseline_data[metric2].iloc[i]), 
                   fontsize=8, alpha=0.7)
    
    plt.colorbar(scatter, ax=ax, label='Pattern Index')

plt.tight_layout()
plt.savefig(output_dir / '5_metric_relationships.png', dpi=300, bbox_inches='tight')
plt.show()

# 6. Box plot of improvement percentages by metric
plt.figure(figsize=(12, 6))
sns.boxplot(x='Metric', y='Improvement %', data=df)
plt.title('Distribution of Improvement % by Metric', fontsize=16)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(output_dir / '6_improvement_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

# 7. Top and Bottom performers
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Top 10 improvements
top_10 = df.nlargest(10, 'Improvement %')[['Pattern', 'Metric', 'Improvement %']]
ax1 = axes[0]
bars1 = ax1.barh(range(len(top_10)), top_10['Improvement %'])
ax1.set_yticks(range(len(top_10)))
ax1.set_yticklabels([f"{p} - {m}" for p, m in zip(top_10['Pattern'], top_10['Metric'])])
ax1.set_xlabel('Improvement %')
ax1.set_title('Top 10 Improvements')
ax1.invert_yaxis()

# Bottom 10 improvements (excluding negative values from uniform pattern)
bottom_10 = df.nsmallest(10, 'Improvement %')[['Pattern', 'Metric', 'Improvement %']]
ax2 = axes[1]
bars2 = ax2.barh(range(len(bottom_10)), bottom_10['Improvement %'])
ax2.set_yticks(range(len(bottom_10)))
ax2.set_yticklabels([f"{p} - {m}" for p, m in zip(bottom_10['Pattern'], bottom_10['Metric'])])
ax2.set_xlabel('Improvement %')
ax2.set_title('Bottom 10 Improvements')
ax2.invert_yaxis()

# Color bars
for bars in [bars1, bars2]:
    for bar, val in zip(bars, [top_10['Improvement %'], bottom_10['Improvement %']][bars is bars2]):
        bar.set_color('green' if val > 0 else 'red')

plt.tight_layout()
plt.savefig(output_dir / '7_top_bottom_performers.png', dpi=300, bbox_inches='tight')
plt.show()

# 8. Statistical summary
print("\n" + "="*50)
print("STATISTICAL SUMMARY")
print("="*50)

print("\nOverall Statistics:")
print(f"Average Improvement %: {df['Improvement %'].mean():.2f}%")
print(f"Median Improvement %: {df['Improvement %'].median():.2f}%")
print(f"Std Dev Improvement %: {df['Improvement %'].std():.2f}%")
print(f"Min Improvement %: {df['Improvement %'].min():.2f}%")
print(f"Max Improvement %: {df['Improvement %'].max():.2f}%")

print("\nAverage Improvement by Metric:")
print(df.groupby('Metric')['Improvement %'].agg(['mean', 'std', 'min', 'max']).round(2))

print("\nAverage Improvement by Pattern (Top 5):")
print(df.groupby('Pattern')['Improvement %'].mean().sort_values(ascending=False).head(10).round(2))

print("\nPatterns with most consistent improvement (lowest std dev):")
pattern_stats = df.groupby('Pattern')['Improvement %'].agg(['mean', 'std']).round(2)
print(pattern_stats.nsmallest(5, 'std'))

print("\n" + "="*50)
print(f"All plots have been saved to: {output_dir}/")
print("="*50)

# 9. Create a comprehensive summary dashboard
fig = plt.figure(figsize=(20, 16))

# Grid layout
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# Main heatmap
ax1 = fig.add_subplot(gs[0, :])
sns.heatmap(pivot_improvement, annot=True, fmt='.1f', cmap='RdYlGn', center=0,
            linewidths=1, cbar_kws={'label': 'Improvement %'}, ax=ax1)
ax1.set_title('Improvement % Heatmap', fontsize=14)

# Average improvement by metric
ax2 = fig.add_subplot(gs[1, 0])
metric_avg = df.groupby('Metric')['Improvement %'].mean().sort_values()
ax2.barh(metric_avg.index, metric_avg.values, color='skyblue')
ax2.set_title('Average Improvement by Metric')
ax2.set_xlabel('Avg Improvement %')

# Average improvement by pattern (top 10)
ax3 = fig.add_subplot(gs[1, 1])
pattern_avg = df.groupby('Pattern')['Improvement %'].mean().sort_values(ascending=False).head(10)
ax3.bar(range(len(pattern_avg)), pattern_avg.values, color='lightgreen')
ax3.set_xticks(range(len(pattern_avg)))
ax3.set_xticklabels(pattern_avg.index, rotation=45, ha='right')
ax3.set_title('Top 10 Patterns by Avg Improvement')
ax3.set_ylabel('Avg Improvement %')

# Improvement distribution
ax4 = fig.add_subplot(gs[1, 2])
ax4.hist(df['Improvement %'], bins=20, edgecolor='black', alpha=0.7)
ax4.set_title('Improvement % Distribution')
ax4.set_xlabel('Improvement %')
ax4.set_ylabel('Frequency')

# Metric correlations
ax5 = fig.add_subplot(gs[2, :])
# Create correlation matrix for improvements across metrics
improvement_pivot = df.pivot(index='Pattern', columns='Metric', values='Improvement %')
corr_matrix = improvement_pivot.corr()
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=ax5)
ax5.set_title('Correlation Between Metrics (Improvement %)')

plt.suptitle('DQN Performance Analysis Dashboard', fontsize=16, y=0.98)
plt.savefig(output_dir / '8_comprehensive_dashboard.png', dpi=300, bbox_inches='tight')
plt.show()