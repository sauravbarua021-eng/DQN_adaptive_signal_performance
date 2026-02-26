#!/usr/bin/env python3
"""
Headway Pattern Visualization for Traffic Signal Control Research
Plots all 16 headway patterns with statistical distributions
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy import stats  # This is correct
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl", 16)

# ============================================================================
# HEADWAY PATTERN GENERATORS
# ============================================================================

class HeadwayVisualizer:
    """Generate and visualize all headway patterns"""
    
    def __init__(self, n_vehicles=1000, duration=3600):
        self.n_vehicles = n_vehicles
        self.duration = duration
        np.random.seed(42)  # For reproducibility
        
        # Pattern categories for organization
        self.categories = {
            'Basic': ['uniform', 'random', 'platoon'],
            'Peak Hours': ['morning_peak', 'evening_peak'],
            'Extreme': ['oversaturated', 'free_flow', 'mixed_fleet'],
            'Statistical': ['lognormal', 'gamma', 'weibull', 'erlang'],
            'Realistic': ['urban_arterial', 'highway_merge', 'incident', 'bimodal']
        }
        
        # Pattern descriptions
        self.descriptions = {
            'uniform': 'Uniform (Exponential)',
            'random': 'Random (Normal)',
            'platoon': 'Platoon (Bunched)',
            'morning_peak': 'Morning Peak (7-9 AM)',
            'evening_peak': 'Evening Peak (5-7 PM)',
            'oversaturated': 'Oversaturated (Extreme Density)',
            'free_flow': 'Free Flow (Sparse)',
            'mixed_fleet': 'Mixed Fleet (Cars + Trucks)',
            'lognormal': 'Log-normal Distribution',
            'gamma': 'Gamma Distribution',
            'weibull': 'Weibull Distribution',
            'erlang': 'Erlang Distribution',
            'urban_arterial': 'Urban Arterial (Signal Platoons)',
            'highway_merge': 'Highway Merge (Ramp Meter)',
            'incident': 'Incident (Disrupted Flow)',
            'bimodal': 'Bimodal (Two Regimes)'
        }
        
        # Flow rates for context
        self.flow_rates = {
            'oversaturated': 900,
            'free_flow': 300,
            'default': 600
        }
    
    def generate_headways(self, pattern):
        """Generate headways for specified pattern"""
        n = self.n_vehicles
        
        if pattern == 'uniform':
            # Exponential distribution (Poisson process)
            mean_headway = 3600 / self.flow_rates.get(pattern, 600)
            return np.random.exponential(mean_headway, n)
        
        elif pattern == 'random':
            # Normal distribution with bounds
            headways = np.abs(np.random.normal(2.5, 1.2, n))
            return np.clip(headways, 0.5, 10)
        
        elif pattern == 'platoon':
            # Bunched arrivals with gaps
            headways = []
            i = 0
            while i < n:
                if np.random.random() < 0.3:
                    platoon_size = np.random.randint(3, 7)
                    headways.extend([np.random.uniform(0.8, 1.5) for _ in range(platoon_size)])
                    i += platoon_size
                else:
                    headways.append(np.random.uniform(3.0, 6.0))
                    i += 1
            return np.array(headways[:n])
        
        elif pattern == 'morning_peak':
            # Heavy traffic in first third (simulating 7-9 AM)
            headways = []
            for i in range(n):
                if i < n * 0.3:
                    headways.append(np.random.uniform(1.2, 2.0))
                elif i < n * 0.6:
                    headways.append(np.random.uniform(2.0, 3.0))
                else:
                    headways.append(np.random.uniform(3.0, 5.0))
            return np.array(headways)
        
        elif pattern == 'evening_peak':
            # Heavy traffic in middle third (simulating 5-7 PM)
            headways = []
            for i in range(n):
                if i < n * 0.4:
                    headways.append(np.random.uniform(3.0, 5.0))
                elif i < n * 0.7:
                    headways.append(np.random.uniform(1.2, 2.0))
                else:
                    headways.append(np.random.uniform(3.0, 5.0))
            return np.array(headways)
        
        elif pattern == 'oversaturated':
            # Extremely dense traffic
            return np.random.exponential(1.2, n)
        
        elif pattern == 'free_flow':
            # Very sparse traffic
            return np.random.exponential(5.0, n)
        
        elif pattern == 'mixed_fleet':
            # Cars (90%) + Trucks (10%)
            headways = []
            for _ in range(n):
                if np.random.random() < 0.1:
                    headways.append(np.random.uniform(3.0, 5.0))  # Truck
                else:
                    headways.append(np.random.uniform(1.5, 3.0))  # Car
            return np.array(headways)
        
        elif pattern == 'lognormal':
            # Log-normal distribution
            return np.random.lognormal(mean=0.8, sigma=0.5, size=n)
        
        elif pattern == 'gamma':
            # Gamma distribution
            return np.random.gamma(shape=2.0, scale=1.2, size=n)
        
        elif pattern == 'weibull':
            # Weibull distribution
            return np.random.weibull(a=1.5, size=n) * 2.5
        
        elif pattern == 'erlang':
            # Erlang (Gamma with integer shape)
            return np.random.gamma(shape=3.0, scale=0.8, size=n)
        
        elif pattern == 'urban_arterial':
            # Signal-induced platoons
            headways = []
            i = 0
            while i < n:
                if np.random.random() < 0.4:
                    platoon_size = np.random.randint(4, 11)
                    base = [0.8, 1.0, 1.2, 1.4, 1.2, 1.0, 0.8]
                    for j in range(min(platoon_size, len(base))):
                        headways.append(base[j])
                    i += platoon_size
                else:
                    headways.append(np.random.uniform(4.0, 8.0))
                    i += 1
            return np.array(headways[:n])
        
        elif pattern == 'highway_merge':
            # Ramp meter releases
            headways = []
            i = 0
            while i < n:
                if np.random.random() < 0.25:
                    release_size = np.random.randint(2, 5)
                    headways.extend([np.random.uniform(1.8, 2.2) for _ in range(release_size)])
                    i += release_size
                else:
                    headways.append(np.random.uniform(4.0, 10.0))
                    i += 1
            return np.array(headways[:n])
        
        elif pattern == 'incident':
            # Disrupted flow with incident in middle
            base = np.random.exponential(2.5, n)
            start = n // 3
            end = 2 * n // 3
            base[start:end] = np.random.exponential(1.2, end - start)
            return base
        
        elif pattern == 'bimodal':
            # Two distinct regimes
            headways = []
            for _ in range(n):
                if np.random.random() < 0.6:
                    headways.append(np.random.exponential(1.8))
                else:
                    headways.append(np.random.exponential(4.5))
            return np.array(headways)
        
        else:
            return np.random.exponential(2.5, n)
    
    def compute_statistics(self, headways):
        """Compute statistics for headways - FIXED: renamed to pattern_stats to avoid conflict"""
        pattern_stats = {
            'mean': float(np.mean(headways)),
            'std': float(np.std(headways)),
            'min': float(np.min(headways)),
            'max': float(np.max(headways)),
            'p5': float(np.percentile(headways, 5)),
            'p25': float(np.percentile(headways, 25)),
            'p50': float(np.percentile(headways, 50)),
            'p75': float(np.percentile(headways, 75)),
            'p95': float(np.percentile(headways, 95))
        }
        return pattern_stats
    
    def plot_all_patterns(self):
        """Create comprehensive visualization of all patterns"""
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 24))
        
        # 1. Main grid of histograms (4x4)
        gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
        axs = gs.subplots()
        
        all_pattern_stats = []  # FIXED: renamed from all_stats
        
        # Flatten patterns list for iteration
        all_patterns = []
        for category, patterns in self.categories.items():
            all_patterns.extend(patterns)
        
        # Plot each pattern
        for idx, pattern in enumerate(all_patterns):
            row, col = idx // 4, idx % 4
            ax = axs[row, col]
            
            # Generate headways
            headways = self.generate_headways(pattern)
            pattern_stats = self.compute_statistics(headways)  # FIXED: renamed variable
            pattern_stats['pattern'] = pattern
            pattern_stats['category'] = self._get_category(pattern)
            all_pattern_stats.append(pattern_stats)
            
            # Plot histogram
            ax.hist(headways, bins=50, density=True, alpha=0.7, 
                   color=plt.cm.tab20(idx/16), edgecolor='black', linewidth=0.5)
            
            # Add KDE - FIXED: using stats.gaussian_kde correctly
            kde_x = np.linspace(0, min(15, np.percentile(headways, 95) * 1.2), 200)
            kde = stats.gaussian_kde(headways)  # This is correct - stats is the scipy module
            ax.plot(kde_x, kde(kde_x), 'r-', linewidth=2, alpha=0.8)
            
            # Add vertical lines for statistics
            ax.axvline(pattern_stats['mean'], color='green', linestyle='--', linewidth=2, 
                      label=f"Mean: {pattern_stats['mean']:.2f}s")
            ax.axvline(pattern_stats['p50'], color='blue', linestyle=':', linewidth=2, 
                      label=f"Median: {pattern_stats['p50']:.2f}s")
            
            # Customize
            ax.set_title(f"{self.descriptions[pattern]}", fontsize=10, fontweight='bold')
            ax.set_xlabel('Headway (seconds)', fontsize=8)
            ax.set_ylabel('Density', fontsize=8)
            ax.tick_params(labelsize=7)
            ax.set_xlim(0, min(15, pattern_stats['p95'] * 1.2))
            ax.legend(fontsize=6, loc='upper right')
            
            # Add stats box
            stats_text = f"σ={pattern_stats['std']:.2f}s\nmin={pattern_stats['min']:.2f}s\nmax={pattern_stats['max']:.2f}s"
            ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, 
                   fontsize=6, verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.suptitle('Traffic Headway Patterns - Distribution Comparison', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        # Save figure
        plt.savefig('headway_patterns_grid.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return all_pattern_stats
    
    def _get_category(self, pattern):
        """Get category for a pattern"""
        for category, patterns in self.categories.items():
            if pattern in patterns:
                return category
        return 'Other'
    
    def plot_comparison(self, all_pattern_stats):
        """Create comparison plots - FIXED: parameter renamed"""
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        df = pd.DataFrame(all_pattern_stats)
        
        # 1. Mean headway by pattern
        ax = axes[0, 0]
        colors = [plt.cm.Set3(i/16) for i in range(len(df))]
        bars = ax.barh(range(len(df)), df['mean'], color=colors)
        ax.set_yticks(range(len(df)))
        ax.set_yticklabels([self.descriptions[p] for p in df['pattern']], fontsize=9)
        ax.set_xlabel('Mean Headway (seconds)', fontsize=11)
        ax.set_title('Average Headway by Pattern', fontweight='bold')
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, df['mean'])):
            ax.text(val + 0.1, bar.get_y() + bar.get_height()/2, 
                   f'{val:.2f}s', va='center', fontsize=8)
        
        # 2. Box plot comparison
        ax = axes[0, 1]
        patterns_short = [self.descriptions[p][:15] + ('...' if len(self.descriptions[p]) > 15 else '') 
                         for p in df['pattern']]
        bp_data = [self.generate_headways(p) for p in df['pattern']]
        box = ax.boxplot(bp_data, labels=patterns_short, vert=False, patch_artist=True)
        
        # Color boxes
        for patch, color in zip(box['boxes'], [plt.cm.Set3(i/16) for i in range(len(df))]):
            patch.set_facecolor(color)
        
        ax.set_xlabel('Headway (seconds)', fontsize=11)
        ax.set_title('Headway Distribution Comparison', fontweight='bold')
        ax.tick_params(axis='y', labelsize=8)
        
        # 3. Coefficient of variation
        ax = axes[1, 0]
        df['cv'] = df['std'] / df['mean']
        colors = ['red' if cv > 0.8 else 'green' for cv in df['cv']]
        bars = ax.barh(range(len(df)), df['cv'], color=colors)
        ax.set_yticks(range(len(df)))
        ax.set_yticklabels([self.descriptions[p] for p in df['pattern']], fontsize=9)
        ax.set_xlabel('Coefficient of Variation (σ/μ)', fontsize=11)
        ax.set_title('Traffic Variability by Pattern', fontweight='bold')
        ax.axvline(0.8, color='red', linestyle='--', alpha=0.5, label='High variability threshold')
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, df['cv'])):
            ax.text(val + 0.02, bar.get_y() + bar.get_height()/2, 
                   f'{val:.2f}', va='center', fontsize=7)
        
        ax.legend(fontsize=8)
        
        # 4. Statistics table
        ax = axes[1, 1]
        ax.axis('tight')
        ax.axis('off')
        
        # Show top 8 patterns to avoid crowding
        table_data = []
        for i, row in df.head(8).iterrows():
            table_data.append([
                self.descriptions[row['pattern']][:12],
                f"{row['mean']:.2f}",
                f"{row['std']:.2f}",
                f"{row['min']:.2f}",
                f"{row['max']:.2f}",
                f"{row['p50']:.2f}"
            ])
        
        table = ax.table(cellText=table_data,
                        colLabels=['Pattern', 'Mean', 'Std', 'Min', 'Max', 'Median'],
                        cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1.2, 1.5)
        
        plt.suptitle('Headway Pattern Statistical Comparison', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('headway_patterns_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_time_series(self):
        """Plot time series for selected patterns"""
        
        fig, axes = plt.subplots(4, 4, figsize=(20, 16))
        fig.suptitle('Headway Time Series (First 200 Vehicles)', fontsize=16, fontweight='bold')
        
        all_patterns = []
        for category, patterns in self.categories.items():
            all_patterns.extend(patterns)
        
        for idx, pattern in enumerate(all_patterns):
            row, col = idx // 4, idx % 4
            ax = axes[row, col]
            
            headways = self.generate_headways(pattern)
            cumulative_time = np.cumsum(headways)
            
            # Plot first 200 vehicles
            n_plot = min(200, len(headways))
            ax.scatter(range(n_plot), headways[:n_plot], s=5, alpha=0.5, c='blue')
            ax.plot(range(n_plot), headways[:n_plot], 'b-', alpha=0.3, linewidth=0.5)
            
            # Add moving average
            if n_plot > 20:
                moving_avg = np.convolve(headways[:n_plot], np.ones(10)/10, mode='valid')
                ax.plot(range(9, 9 + len(moving_avg)), moving_avg, 'r-', linewidth=2, label='Moving avg (10)')
            
            ax.set_title(self.descriptions[pattern], fontsize=9)
            ax.set_xlabel('Vehicle Number', fontsize=7)
            ax.set_ylabel('Headway (s)', fontsize=7)
            ax.set_ylim(0, min(15, np.percentile(headways, 95) * 1.2))
            ax.tick_params(labelsize=6)
            ax.grid(True, alpha=0.3)
            
            if idx == 0:
                ax.legend(fontsize=6)
        
        plt.tight_layout()
        plt.savefig('headway_patterns_timeseries.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def print_summary(self, all_pattern_stats):
        """Print summary statistics - FIXED: parameter renamed"""
        
        print("\n" + "="*80)
        print("HEADWAY PATTERNS - SUMMARY STATISTICS")
        print("="*80)
        
        df = pd.DataFrame(all_pattern_stats)
        
        for category in self.categories.keys():
            print(f"\n{category} PATTERNS:")
            print("-" * 60)
            cat_df = df[df['category'] == category]
            
            for _, row in cat_df.iterrows():
                pattern = row['pattern']
                print(f"\n  {self.descriptions[pattern]}:")
                print(f"    Mean: {row['mean']:.3f}s | Std: {row['std']:.3f}s")
                print(f"    Min: {row['min']:.3f}s | Max: {row['max']:.3f}s")
                print(f"    Percentiles: 5th={row['p5']:.2f}s, 50th={row['p50']:.2f}s, 95th={row['p95']:.2f}s")
        
        print("\n" + "="*80)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main function to generate all plots"""
    
    print("\n" + "="*80)
    print("HEADWAY PATTERN VISUALIZATION")
    print("16 Different Traffic Patterns")
    print("="*80)
    
    # Create visualizer
    visualizer = HeadwayVisualizer(n_vehicles=2000)
    
    # Generate all patterns and get statistics
    print("\n📊 Generating headway patterns...")
    all_pattern_stats = visualizer.plot_all_patterns()
    
    # Print summary statistics
    visualizer.print_summary(all_pattern_stats)
    
    # Plot comparison
    print("\n📈 Creating comparison plots...")
    visualizer.plot_comparison(all_pattern_stats)
    
    # Plot time series
    print("\n⏱️  Creating time series plots...")
    visualizer.plot_time_series()
    
    print("\n✅ All plots saved!")
    print("   - headway_patterns_grid.png")
    print("   - headway_patterns_comparison.png")
    print("   - headway_patterns_timeseries.png")
    print("\n📊 Check the current directory for the generated PNG files.")

if __name__ == "__main__":
    main()