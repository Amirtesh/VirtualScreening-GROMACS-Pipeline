#!/usr/bin/env python3
"""
GROMACS XVG File Plot Analyzer
Supports: RMSD, RMSF, SASA, Gyration, Mindist, Hydrogen Bonds
Usage: python plotXVG.py file1.xvg [file2.xvg ...]
"""

import sys
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import re

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class XVGAnalyzer:
    def __init__(self):
        # Color scheme for multiple files
        self.colors = ['#1f77b4', '#d62728', '#2ca02c', '#ff7f0e', 
                      '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', 
                      '#bcbd22', '#17becf']
        
        # Store legend labels for editing
        self.legend_labels = {}
        self.force_plot_type = None
        
        # Plot type detection patterns - IMPROVED specificity and order
        self.plot_patterns = {
            'hbond': ['hbond', 'hbnum', 'hydrogen_bond', 'hydrogen-bond', 'h-bond', 'hb-', 'bond'],
            'rmsd': ['rmsd', 'root_mean_square_deviation', 'backbone_rmsd'],
            'rmsf': ['rmsf', 'root_mean_square_fluctuation', 'residue_fluctuation', 'bfactor'],
            'sasa': ['sasa', 'solvent_accessible_surface_area', 'surface_area', 'area'],
            'gyration': ['gyrate', 'radius_of_gyration', 'rg', 'gyration_radius'],
            'mindist': ['mindist', 'minimum_distance', 'min_dist', 'distance']
        }
        
        # Units and labels
        self.plot_configs = {
            'rmsd': {
                'ylabel': 'RMSD (nm)',
                'xlabel': 'Time (ns)',
                'title': 'Root Mean Square Deviation'
            },
            'rmsf': {
                'ylabel': 'RMSF (nm)',
                'xlabel': 'Residue Number',
                'title': 'Root Mean Square Fluctuation'
            },
            'sasa': {
                'ylabel': 'SASA (nm²)',
                'xlabel': 'Time (ns)', 
                'title': 'Solvent Accessible Surface Area'
            },
            'gyration': {
                'ylabel': 'Radius of Gyration (nm)',
                'xlabel': 'Time (ns)',
                'title': 'Radius of Gyration'
            },
            'mindist': {
                'ylabel': 'Distance (nm)',
                'xlabel': 'Time (ns)',
                'title': 'Minimum Distance'
            },
            'hbond': {
                'ylabel': 'Number of Hydrogen Bonds',
                'xlabel': 'Time (ns)',
                'title': 'Hydrogen Bonds'
            }
        }

    def read_xvg(self, filename):
        """Read GROMACS XVG file and return data as pandas DataFrame"""
        data = []
        comments = []
        
        try:
            with open(filename, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('#') or line.startswith('@'):
                        comments.append(line)
                        continue
                    if line and not line.startswith('#'):
                        values = line.split()
                        if len(values) >= 2:
                            try:
                                data.append([float(x) for x in values])
                            except ValueError:
                                continue
            
            if not data:
                raise ValueError(f"No valid data found in {filename}")
                
            df = pd.DataFrame(data)
            return df, comments
            
        except Exception as e:
            print(f"Error reading {filename}: {e}")
            return None, None

    def detect_plot_type(self, filename, comments):
        """Detect the type of plot based on filename and comments - FIXED"""
        if self.force_plot_type:
            return self.force_plot_type
            
        filename_lower = filename.lower()
        comments_str = ' '.join(comments).lower()
        
        print(f"Debug - Filename: {filename_lower}")
        print(f"Debug - Comments: {comments_str}")
        
        # Check patterns in order of specificity - most specific first
        for plot_type, patterns in self.plot_patterns.items():
            for pattern in patterns:
                if pattern in filename_lower:
                    print(f"Detected '{plot_type}' from filename pattern '{pattern}'")
                    return plot_type
                elif pattern in comments_str:
                    print(f"Detected '{plot_type}' from comment pattern '{pattern}'")
                    return plot_type
        
        # Default fallback
        print(f"No specific pattern detected, using generic plot for {filename}")
        return 'generic'

    def convert_time_to_ns(self, time_data, comments):
        """Convert time units to nanoseconds based on XVG comments"""
        # Check comments for time unit
        time_unit = 'ps'  # default
        for comment in comments:
            if 'xaxis' in comment.lower() and 'time' in comment.lower():
                if '(ns)' in comment:
                    time_unit = 'ns'
                elif '(ps)' in comment:
                    time_unit = 'ps'
                elif '(fs)' in comment:
                    time_unit = 'fs'
        
        if time_unit == 'ps':
            return time_data / 1000.0  # ps to ns
        elif time_unit == 'fs':
            return time_data / 1000000.0  # fs to ns
        else:
            return time_data  # already in ns

    def on_legend_click(self, event):
        """Handle legend click for editing labels - FIXED"""
        if event.artist.get_label().startswith('_'):
            return
            
        # Get current label
        current_label = event.artist.get_label()
        
        # Simple input dialog using matplotlib
        fig = plt.figure(figsize=(4, 2))
        fig.suptitle('Edit Legend Label')
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.7, f'Current: {current_label}', ha='center', transform=ax.transAxes)
        ax.text(0.5, 0.5, 'Enter new label in console', ha='center', transform=ax.transAxes)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        plt.show(block=False)
        
        # Get input from console
        try:
            new_label = input(f"Enter new label for '{current_label}': ").strip()
            if new_label:
                event.artist.set_label(new_label)
                # Update the main plot legend
                event.artist.axes.legend()
                plt.draw()
                print(f"Label updated to: {new_label}")
            else:
                print("No change made.")
        except KeyboardInterrupt:
            print("\nLabel editing cancelled.")
        
        plt.close(fig)

    def format_axes(self, ax, plot_type='generic', x_data=None, y_data=None):
        """Format axes with proper tick spacing and limits"""
        from matplotlib.ticker import MultipleLocator, AutoLocator, MaxNLocator
        
        # Handle different plot types
        if plot_type == 'rmsf':
            # For RMSF, x-axis is residue numbers, y-axis is RMSF values
            if x_data is not None and y_data is not None:
                # X-axis: residue numbers - start from minimum residue
                x_min, x_max = x_data.min(), x_data.max()
                x_range = x_max - x_min
                
                # Set x-axis limits with small padding
                ax.set_xlim(x_min - x_range * 0.01, x_max + x_range * 0.01)
                
                # Y-axis: RMSF values - find appropriate range
                y_min, y_max = y_data.min(), y_data.max()
                y_range = y_max - y_min
                y_padding = y_range * 0.05
                
                # Start y-axis from a reasonable minimum (not necessarily 0)
                y_start = max(0, y_min - y_padding)
                ax.set_ylim(y_start, y_max + y_padding)
                
                # Set reasonable tick spacing for residues
                if x_range <= 50:
                    x_tick_spacing = 5
                elif x_range <= 200:
                    x_tick_spacing = 20
                elif x_range <= 500:
                    x_tick_spacing = 50
                else:
                    x_tick_spacing = 100
                
                ax.xaxis.set_major_locator(MultipleLocator(x_tick_spacing))
                ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
            
        elif plot_type in ['rmsd', 'sasa', 'gyration', 'mindist', 'hbond']:
            # For time-series data
            if x_data is not None and y_data is not None:
                # X-axis: time - always start from 0, end exactly at max time
                x_max = x_data.max()
                ax.set_xlim(0, x_max)
                
                # Set x-axis ticks in intervals of 20 ns (or appropriate intervals)
                if x_max <= 50:
                    x_tick_spacing = 10
                elif x_max <= 100:
                    x_tick_spacing = 20
                elif x_max <= 500:
                    x_tick_spacing = 50
                else:
                    x_tick_spacing = 100
                
                ax.xaxis.set_major_locator(MultipleLocator(x_tick_spacing))
                
                # Y-axis: find appropriate range based on data
                y_min, y_max = y_data.min(), y_data.max()
                y_range = y_max - y_min
                y_padding = y_range * 0.05
                
                # For most cases, start from 0 if y_min is close to 0, otherwise use data range
                if y_min >= 0 and y_min <= y_range * 0.1:
                    y_start = 0
                else:
                    y_start = y_min - y_padding
                
                ax.set_ylim(y_start, y_max + y_padding)
                ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
        
        else:
            # Generic case
            if x_data is not None and y_data is not None:
                x_min, x_max = x_data.min(), x_data.max()
                y_min, y_max = y_data.min(), y_data.max()
                
                x_range = x_max - x_min
                y_range = y_max - y_min
                
                x_padding = x_range * 0.02
                y_padding = y_range * 0.05
                
                ax.set_xlim(x_min - x_padding, x_max + x_padding)
                ax.set_ylim(y_min - y_padding, y_max + y_padding)
                
                ax.xaxis.set_major_locator(MaxNLocator(nbins=8))
                ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
        
        # Clean up spines
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_linewidth(1.0)
        ax.spines['bottom'].set_linewidth(1.0)
        
        # Format ticks
        ax.tick_params(axis='both', which='major', labelsize=10, direction='out')
        ax.tick_params(axis='both', which='minor', labelsize=8, direction='out')
        
        # Force a redraw
        ax.figure.canvas.draw_idle()
        
    def get_label(self, filename):
        """Get label for a filename, using custom label if available"""
        if hasattr(self, 'legend_labels') and filename in self.legend_labels:
            return self.legend_labels[filename]
        return Path(filename).stem

    def add_interactive_legend(self, ax, data_dict):
        """Add interactive legend that can be edited by clicking"""
        if len(data_dict) > 1:
            # Create legend with picker enabled
            legend = ax.legend(frameon=True, fancybox=True, shadow=True)
            
            # Make legend items clickable
            for legend_line in legend.get_lines():
                legend_line.set_picker(5)  # 5 points tolerance
            
            # Connect the click event
            ax.figure.canvas.mpl_connect('pick_event', self.on_legend_click)
            
            return legend
        return None

    def save_plot_interactive(self, output_name, plot_type):
        """Interactive save function with 600 DPI"""
        def on_key_press(event):
            if event.key == 's':
                filename = f'{output_name}_{plot_type}.png'
                plt.savefig(filename, dpi=600, bbox_inches='tight')
                print(f"Plot saved as {filename} (600 DPI)")
            elif event.key == 'q':
                plt.close('all')
        
        plt.gcf().canvas.mpl_connect('key_press_event', on_key_press)
        print("Press 's' to save (600 DPI), 'q' to quit, or close window to exit")

    def plot_rmsd(self, data_dict, output_name):
        """Plot RMSD data"""
        plt.figure(figsize=(10, 6))
        
        all_x_data = []
        all_y_data = []
        
        for i, (filename, (df, comments)) in enumerate(data_dict.items()):
            time_ns = self.convert_time_to_ns(df.iloc[:, 0], comments)
            rmsd = df.iloc[:, 1]
            
            all_x_data.extend(time_ns)
            all_y_data.extend(rmsd)
            
            label = self.get_label(filename)
            plt.plot(time_ns, rmsd, color=self.colors[i % len(self.colors)], 
                    linewidth=1.5, label=label, alpha=0.8)
        
        config = self.plot_configs['rmsd']
        plt.xlabel(config['xlabel'], fontsize=12)
        plt.ylabel(config['ylabel'], fontsize=12)
        plt.title(config['title'], fontsize=14, fontweight='bold')
        
        ax = plt.gca()
        self.format_axes(ax, 'rmsd', np.array(all_x_data), np.array(all_y_data))
        self.add_interactive_legend(ax, data_dict)
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        self.save_plot_interactive(output_name, 'rmsd')
        plt.show()

    def plot_rmsf(self, data_dict, output_name):
        """Plot RMSF data"""
        plt.figure(figsize=(12, 6))
        
        all_x_data = []
        all_y_data = []
        
        for i, (filename, (df, comments)) in enumerate(data_dict.items()):
            residue = df.iloc[:, 0]
            rmsf = df.iloc[:, 1]
            
            all_x_data.extend(residue)
            all_y_data.extend(rmsf)
            
            label = self.get_label(filename)
            plt.plot(residue, rmsf, color=self.colors[i % len(self.colors)], 
                    linewidth=1.5, label=label, alpha=0.8)
        
        config = self.plot_configs['rmsf']
        plt.xlabel(config['xlabel'], fontsize=12)
        plt.ylabel(config['ylabel'], fontsize=12)
        plt.title(config['title'], fontsize=14, fontweight='bold')
        
        ax = plt.gca()
        self.format_axes(ax, 'rmsf', np.array(all_x_data), np.array(all_y_data))
        self.add_interactive_legend(ax, data_dict)
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        self.save_plot_interactive(output_name, 'rmsf')
        plt.show()

    def plot_sasa(self, data_dict, output_name):
        """Plot SASA data"""
        plt.figure(figsize=(10, 6))
        
        all_x_data = []
        all_y_data = []
        
        for i, (filename, (df, comments)) in enumerate(data_dict.items()):
            time_ns = self.convert_time_to_ns(df.iloc[:, 0], comments)
            sasa = df.iloc[:, 1]
            
            all_x_data.extend(time_ns)
            all_y_data.extend(sasa)
            
            label = self.get_label(filename)
            plt.plot(time_ns, sasa, color=self.colors[i % len(self.colors)], 
                    linewidth=1.5, label=label, alpha=0.8)
        
        config = self.plot_configs['sasa']
        plt.xlabel(config['xlabel'], fontsize=12)
        plt.ylabel(config['ylabel'], fontsize=12)
        plt.title(config['title'], fontsize=14, fontweight='bold')
        
        ax = plt.gca()
        self.format_axes(ax, 'sasa', np.array(all_x_data), np.array(all_y_data))
        self.add_interactive_legend(ax, data_dict)
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        self.save_plot_interactive(output_name, 'sasa')
        plt.show()

    def plot_gyration(self, data_dict, output_name):
        """Plot Radius of Gyration data"""
        plt.figure(figsize=(10, 6))
        
        all_x_data = []
        all_y_data = []
        
        for i, (filename, (df, comments)) in enumerate(data_dict.items()):
            time_ns = self.convert_time_to_ns(df.iloc[:, 0], comments)
            rg = df.iloc[:, 1]
            
            all_x_data.extend(time_ns)
            all_y_data.extend(rg)
            
            label = self.get_label(filename)
            plt.plot(time_ns, rg, color=self.colors[i % len(self.colors)], 
                    linewidth=1.5, label=label, alpha=0.8)
        
        config = self.plot_configs['gyration']
        plt.xlabel(config['xlabel'], fontsize=12)
        plt.ylabel(config['ylabel'], fontsize=12)
        plt.title(config['title'], fontsize=14, fontweight='bold')
        
        ax = plt.gca()
        self.format_axes(ax, 'gyration', np.array(all_x_data), np.array(all_y_data))
        self.add_interactive_legend(ax, data_dict)
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        self.save_plot_interactive(output_name, 'gyration')
        plt.show()

    def plot_mindist(self, data_dict, output_name):
        """Plot Minimum Distance data"""
        plt.figure(figsize=(10, 6))
        
        all_x_data = []
        all_y_data = []
        
        for i, (filename, (df, comments)) in enumerate(data_dict.items()):
            time_ns = self.convert_time_to_ns(df.iloc[:, 0], comments)
            dist = df.iloc[:, 1]
            
            all_x_data.extend(time_ns)
            all_y_data.extend(dist)
            
            label = self.get_label(filename)
            plt.plot(time_ns, dist, color=self.colors[i % len(self.colors)], 
                    linewidth=1.5, label=label, alpha=0.8)
        
        config = self.plot_configs['mindist']
        plt.xlabel(config['xlabel'], fontsize=12)
        plt.ylabel(config['ylabel'], fontsize=12)
        plt.title(config['title'], fontsize=14, fontweight='bold')
        
        ax = plt.gca()
        self.format_axes(ax, 'mindist', np.array(all_x_data), np.array(all_y_data))
        self.add_interactive_legend(ax, data_dict)
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        self.save_plot_interactive(output_name, 'mindist')
        plt.show()

    def plot_hbond(self, data_dict, output_name):
        """Plot Hydrogen Bonds data"""
        plt.figure(figsize=(10, 6))
        
        all_x_data = []
        all_y_data = []
        
        for i, (filename, (df, comments)) in enumerate(data_dict.items()):
            time_ns = self.convert_time_to_ns(df.iloc[:, 0], comments)
            hbonds = df.iloc[:, 1]
            
            all_x_data.extend(time_ns)
            all_y_data.extend(hbonds)
            
            label = self.get_label(filename)
            color = self.colors[i % len(self.colors)]
            
            # Main time series
            plt.plot(time_ns, hbonds, color=color, linewidth=1.5, 
                    label=label, alpha=0.8)
            
            # Running average (optional, but lighter)
            if len(hbonds) > 100:
                window = max(50, len(hbonds) // 100)
                hbonds_smooth = pd.Series(hbonds).rolling(window=window, center=True).mean()
                plt.plot(time_ns, hbonds_smooth, color=color, linewidth=2.5, 
                        linestyle='--', alpha=0.6)
        
        config = self.plot_configs['hbond']
        plt.xlabel(config['xlabel'], fontsize=12)
        plt.ylabel(config['ylabel'], fontsize=12)
        plt.title(config['title'], fontsize=14, fontweight='bold')
        
        ax = plt.gca()
        self.format_axes(ax, 'hbond', np.array(all_x_data), np.array(all_y_data))
        self.add_interactive_legend(ax, data_dict)
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        self.save_plot_interactive(output_name, 'hbonds')
        plt.show()
        
        # Print statistics
        print("\nHydrogen Bond Statistics:")
        print("-" * 50)
        for filename, (df, comments) in data_dict.items():
            hbonds = df.iloc[:, 1]
            print(f"{Path(filename).stem}:")
            print(f"  Mean: {hbonds.mean():.2f} ± {hbonds.std():.2f}")
            print(f"  Min:  {hbonds.min():.0f}")
            print(f"  Max:  {hbonds.max():.0f}")
            print(f"  Median: {hbonds.median():.2f}")

    def plot_generic(self, data_dict, output_name):
        """Generic plot for unrecognized file types"""
        plt.figure(figsize=(10, 6))
        
        all_x_data = []
        all_y_data = []
        
        for i, (filename, (df, comments)) in enumerate(data_dict.items()):
            x_data = df.iloc[:, 0]
            y_data = df.iloc[:, 1]
            
            all_x_data.extend(x_data)
            all_y_data.extend(y_data)
            
            label = self.get_label(filename)
            plt.plot(x_data, y_data, color=self.colors[i % len(self.colors)], 
                    linewidth=1.5, label=label, alpha=0.8)
        
        plt.xlabel('X Data', fontsize=12)
        plt.ylabel('Y Data', fontsize=12)
        plt.title('XVG Data Plot', fontsize=14, fontweight='bold')
        
        ax = plt.gca()
        self.format_axes(ax, 'generic', np.array(all_x_data), np.array(all_y_data))
        self.add_interactive_legend(ax, data_dict)
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        self.save_plot_interactive(output_name, 'generic')
        plt.show()

    def analyze_files(self, filenames, custom_labels=None):
        """Main analysis function"""
        if not filenames:
            print("No files provided!")
            return
        
        # Read all files
        data_dict = {}
        plot_type = None
        
        for filename in filenames:
            if not os.path.exists(filename):
                print(f"Warning: File {filename} not found, skipping...")
                continue
                
            df, comments = self.read_xvg(filename)
            if df is not None:
                data_dict[filename] = (df, comments)
                if plot_type is None:
                    plot_type = self.detect_plot_type(filename, comments)
        
        if not data_dict:
            print("No valid data files found!")
            return
        
        # Apply custom labels if provided
        if custom_labels:
            self.legend_labels = dict(zip(filenames, custom_labels))
        
        # Generate output name
        if len(filenames) == 1:
            output_name = Path(filenames[0]).stem
        else:
            output_name = "combined_analysis"
        
        print(f"Final plot type: {plot_type}")
        print(f"Processing {len(data_dict)} files...")
        
        # Plot based on detected type
        plot_functions = {
            'rmsd': self.plot_rmsd,
            'rmsf': self.plot_rmsf,
            'sasa': self.plot_sasa,
            'gyration': self.plot_gyration,
            'mindist': self.plot_mindist,
            'hbond': self.plot_hbond,
            'generic': self.plot_generic
        }
        
        plot_func = plot_functions.get(plot_type, self.plot_generic)
        plot_func(data_dict, output_name)
        
        print("\nInteractive features:")
        print("- Click on legend entries to edit labels")
        print("- Press 's' to save plot (600 DPI)")
        print("- Press 'q' to quit or close window")

def main():
    parser = argparse.ArgumentParser(
        description='GROMACS XVG File Plot Analyzer',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python plotXVG.py rmsd.xvg
  python plotXVG.py rmsd1.xvg rmsd2.xvg rmsd3.xvg
  python plotXVG.py hbond.xvg --type hbond
  python plotXVG.py rmsd1.xvg rmsd2.xvg --labels "System A" "System B"
        '''
    )
    
    parser.add_argument('files', nargs='+', help='XVG files to analyze')
    parser.add_argument('--type', help='Force plot type (rmsd, rmsf, sasa, gyration, mindist, hbond)')
    parser.add_argument('--output', help='Output filename prefix')
    parser.add_argument('--labels', nargs='+', help='Custom legend labels for each file')
    
    args = parser.parse_args()
    
    analyzer = XVGAnalyzer()
    
    # Override plot type if specified
    if args.type:
        analyzer.force_plot_type = args.type
    
    # Check if custom labels match number of files
    if args.labels and len(args.labels) != len(args.files):
        print(f"Warning: Number of labels ({len(args.labels)}) doesn't match number of files ({len(args.files)})")
        print("Using default labels...")
        args.labels = None
    
    analyzer.analyze_files(args.files, args.labels)

if __name__ == "__main__":
    main()
