"""
Visualization tools for experimental results.
Generates phase transition heatmaps, trajectory plots, and audit statistics.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any
import os


class ExperimentVisualizer:
    """Visualize experimental results and dynamics."""
    
    def __init__(self, results_path: str):
        """
        Initialize visualizer with results file.
        
        Args:
            results_path: Path to JSON results file
        """
        with open(results_path, 'r') as f:
            self.results = json.load(f)
        
        self.output_dir = os.path.join(
            os.path.dirname(results_path),
            'visualizations'
        )
        os.makedirs(self.output_dir, exist_ok=True)
    
    def plot_phase_transition_heatmap(self, save_path: str = None):
        """
        Generate phase transition heatmap.
        X-axis: Temperature, Y-axis: Context Window, Color: Success Rate
        """
        # Extract unique temperatures and context windows
        temps = sorted(set(r['temperature'] for r in self.results))
        contexts = sorted(set(r['context_window'] for r in self.results))
        
        # Create success rate matrix
        success_matrix = np.zeros((len(contexts), len(temps)))
        
        for i, context in enumerate(contexts):
            for j, temp in enumerate(temps):
                # Get all trials for this configuration
                trials = [r for r in self.results 
                         if r['temperature'] == temp and r['context_window'] == context]
                
                if trials:
                    success_rate = sum(r.get('converged', False) for r in trials) / len(trials)
                    success_matrix[i, j] = success_rate
        
        # Plot heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            success_matrix,
            xticklabels=[f'{t:.1f}' for t in temps],
            yticklabels=[f'{c}' for c in contexts],
            annot=True,
            fmt='.2f',
            cmap='RdYlGn',
            vmin=0,
            vmax=1,
            cbar_kws={'label': 'Convergence Rate'}
        )
        
        plt.xlabel('Temperature (Learning Rate)', fontsize=12)
        plt.ylabel('Context Window (Batch Size)', fontsize=12)
        plt.title('Phase Transition Heatmap: Convergence Success Rate', fontsize=14, fontweight='bold')
        
        if save_path is None:
            save_path = os.path.join(self.output_dir, 'phase_transition_heatmap.png')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved phase transition heatmap to: {save_path}")
    
    def plot_trajectory_comparison(self, save_path: str = None):
        """
        Plot typical trajectories for different temperatures.
        Shows loss curves over optimization steps.
        """
        # Select representative trajectories
        low_temp = [r for r in self.results if r['temperature'] <= 0.3]
        mid_temp = [r for r in self.results if 0.6 <= r['temperature'] <= 0.8]
        high_temp = [r for r in self.results if r['temperature'] >= 1.2]
        
        plt.figure(figsize=(12, 6))
        
        # Plot low temperature (greedy)
        if low_temp:
            traj = low_temp[0]['trajectory']
            steps = [t['t'] for t in traj]
            losses = [t['loss'] for t in traj]
            plt.plot(steps, losses, 'b-', linewidth=2, label=f'T={low_temp[0]["temperature"]:.1f} (Greedy)', marker='o')
        
        # Plot mid temperature (sweet spot)
        if mid_temp:
            traj = mid_temp[0]['trajectory']
            steps = [t['t'] for t in traj]
            losses = [t['loss'] for t in traj]
            plt.plot(steps, losses, 'g-', linewidth=2, label=f'T={mid_temp[0]["temperature"]:.1f} (Sweet Spot)', marker='s')
        
        # Plot high temperature (exploratory)
        if high_temp:
            traj = high_temp[0]['trajectory']
            steps = [t['t'] for t in traj]
            losses = [t['loss'] for t in traj]
            plt.plot(steps, losses, 'r-', linewidth=2, label=f'T={high_temp[0]["temperature"]:.1f} (Exploratory)', marker='^')
        
        plt.xlabel('Optimization Step', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.title('Trajectory Comparison: Different Temperature Regimes', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        
        if save_path is None:
            save_path = os.path.join(self.output_dir, 'trajectory_comparison.png')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved trajectory comparison to: {save_path}")
    
    def plot_audit_statistics(self, save_path: str = None):
        """
        Plot audit log statistics.
        Shows distribution of experiment statuses.
        """
        # Collect all status counts
        all_statuses = {}
        
        for result in self.results:
            if 'statistics' in result and 'status_counts' in result['statistics']:
                for status, count in result['statistics']['status_counts'].items():
                    all_statuses[status] = all_statuses.get(status, 0) + count
        
        if not all_statuses:
            print("No audit statistics available")
            return
        
        # Plot bar chart
        plt.figure(figsize=(10, 6))
        statuses = list(all_statuses.keys())
        counts = list(all_statuses.values())
        
        colors = {
            'Normal Descent': 'green',
            'Limit Cycle Detected': 'orange',
            'Vanishing Gradient': 'red',
            'Exploding Gradient': 'darkred',
            'Converged to Solution': 'blue',
            'Warming Up': 'gray'
        }
        
        bar_colors = [colors.get(s, 'gray') for s in statuses]
        
        plt.bar(range(len(statuses)), counts, color=bar_colors, alpha=0.7)
        plt.xticks(range(len(statuses)), statuses, rotation=45, ha='right')
        plt.ylabel('Frequency', fontsize=12)
        plt.title('Dynamics Audit Statistics', fontsize=14, fontweight='bold')
        plt.grid(True, axis='y', alpha=0.3)
        
        if save_path is None:
            save_path = os.path.join(self.output_dir, 'audit_statistics.png')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved audit statistics to: {save_path}")
    
    def plot_oscillation_analysis(self, save_path: str = None):
        """
        Analyze oscillation index vs hyperparameters.
        """
        # Extract oscillation indices
        data = []
        for result in self.results:
            if 'statistics' in result and 'oscillation_index' in result['statistics']:
                data.append({
                    'temperature': result['temperature'],
                    'context_window': result['context_window'],
                    'oscillation_index': result['statistics']['oscillation_index']
                })
        
        if not data:
            print("No oscillation data available")
            return
        
        # Group by temperature
        temps = sorted(set(d['temperature'] for d in data))
        
        plt.figure(figsize=(10, 6))
        
        for temp in temps:
            temp_data = [d for d in data if d['temperature'] == temp]
            contexts = [d['context_window'] for d in temp_data]
            oscillations = [d['oscillation_index'] for d in temp_data]
            
            plt.plot(contexts, oscillations, marker='o', label=f'T={temp:.1f}', linewidth=2)
        
        plt.xlabel('Context Window Size', fontsize=12)
        plt.ylabel('Oscillation Index', fontsize=12)
        plt.title('Oscillation Index vs Context Window', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        
        if save_path is None:
            save_path = os.path.join(self.output_dir, 'oscillation_analysis.png')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved oscillation analysis to: {save_path}")
    
    def generate_all_plots(self):
        """Generate all visualization plots."""
        print("Generating visualizations...")
        
        self.plot_phase_transition_heatmap()
        self.plot_trajectory_comparison()
        self.plot_audit_statistics()
        self.plot_oscillation_analysis()
        
        print(f"\nAll visualizations saved to: {self.output_dir}")
    
    def generate_summary_report(self, save_path: str = None):
        """Generate text summary report."""
        if save_path is None:
            save_path = os.path.join(self.output_dir, 'summary_report.txt')
        
        with open(save_path, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("CODEBASE DESCENT: EXPERIMENTAL SUMMARY REPORT\n")
            f.write("=" * 60 + "\n\n")
            
            # Overall statistics
            total_trials = len(self.results)
            converged = sum(1 for r in self.results if r.get('converged', False))
            
            f.write(f"Total Trials: {total_trials}\n")
            f.write(f"Converged: {converged} ({converged/total_trials:.1%})\n")
            f.write(f"Failed: {total_trials - converged} ({(total_trials-converged)/total_trials:.1%})\n\n")
            
            # Best configuration
            converged_results = [r for r in self.results if r.get('converged', False)]
            if converged_results:
                best = min(converged_results, key=lambda r: r['total_steps'])
                f.write("Best Configuration (fastest convergence):\n")
                f.write(f"  Temperature: {best['temperature']}\n")
                f.write(f"  Context Window: {best['context_window']}\n")
                f.write(f"  Steps to convergence: {best['total_steps']}\n\n")
            
            # Temperature analysis
            f.write("Temperature Analysis:\n")
            temps = sorted(set(r['temperature'] for r in self.results))
            for temp in temps:
                temp_results = [r for r in self.results if r['temperature'] == temp]
                temp_converged = sum(1 for r in temp_results if r.get('converged', False))
                f.write(f"  T={temp:.1f}: {temp_converged}/{len(temp_results)} converged ({temp_converged/len(temp_results):.1%})\n")
            
            f.write("\n")
            
            # Context window analysis
            f.write("Context Window Analysis:\n")
            contexts = sorted(set(r['context_window'] for r in self.results))
            for ctx in contexts:
                ctx_results = [r for r in self.results if r['context_window'] == ctx]
                ctx_converged = sum(1 for r in ctx_results if r.get('converged', False))
                f.write(f"  K={ctx}: {ctx_converged}/{len(ctx_results)} converged ({ctx_converged/len(ctx_results):.1%})\n")
            
            f.write("\n" + "=" * 60 + "\n")
        
        print(f"Summary report saved to: {save_path}")


def visualize_results(results_path: str):
    """
    Main function to visualize experimental results.
    
    Args:
        results_path: Path to results JSON file
    """
    visualizer = ExperimentVisualizer(results_path)
    visualizer.generate_all_plots()
    visualizer.generate_summary_report()


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize experimental results')
    parser.add_argument('results_path', type=str, help='Path to results JSON file')
    
    args = parser.parse_args()
    visualize_results(args.results_path)
