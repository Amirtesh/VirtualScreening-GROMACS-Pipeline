#!/usr/bin/env python3

import sys
import os
import subprocess
import tempfile
import time
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, 
                            QPushButton, QFileDialog, QListWidget, QTextEdit, QLabel, QSpinBox, 
                            QGroupBox, QGridLayout, QMessageBox, QProgressBar, QSplitter, 
                            QCheckBox, QListWidgetItem, QMenu, QAction, QLineEdit, QTabWidget,
                            QComboBox, QScrollArea, QDialog, QFormLayout, QDialogButtonBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QProcess
from PyQt5.QtGui import QFont, QTextCursor

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import seaborn as sns
import pandas as pd
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage import gaussian_filter
from pathlib import Path
from matplotlib.ticker import MultipleLocator, AutoLocator, MaxNLocator

class LegendEditDialog(QDialog):
    def __init__(self, current_labels, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Edit Legend Names")
        self.setModal(True)
        self.resize(400, 300)
        
        layout = QVBoxLayout()
        
        # Instructions
        instruction_label = QLabel("Edit the legend names for each file:")
        instruction_label.setFont(QFont("Arial", 10, QFont.Bold))
        layout.addWidget(instruction_label)
        
        # Form layout for legend entries
        form_layout = QFormLayout()
        self.legend_inputs = []
        
        for i, label in enumerate(current_labels):
            line_edit = QLineEdit(label)
            line_edit.setPlaceholderText(f"Legend name for file {i+1}")
            self.legend_inputs.append(line_edit)
            form_layout.addRow(f"File {i+1}:", line_edit)
        
        form_widget = QWidget()
        form_widget.setLayout(form_layout)
        
        # Scroll area in case of many files
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(form_widget)
        layout.addWidget(scroll_area)
        
        # Buttons
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)
        
        # Additional buttons
        button_layout = QHBoxLayout()
        
        reset_btn = QPushButton("Reset to Filenames")
        reset_btn.clicked.connect(self.reset_to_filenames)
        button_layout.addWidget(reset_btn)
        
        auto_name_btn = QPushButton("Auto Name (File 1, File 2, etc.)")
        auto_name_btn.clicked.connect(self.auto_name)
        button_layout.addWidget(auto_name_btn)
        
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
        self.original_labels = current_labels.copy()
    
    def reset_to_filenames(self):
        for i, line_edit in enumerate(self.legend_inputs):
            line_edit.setText(self.original_labels[i])
    
    def auto_name(self):
        for i, line_edit in enumerate(self.legend_inputs):
            line_edit.setText(f"File {i+1}")
    
    def get_legend_names(self):
        return [line_edit.text().strip() or f"File {i+1}" 
                for i, line_edit in enumerate(self.legend_inputs)]

class InteractiveTerminal(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.process = None
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        
        self.terminal_output = QTextEdit()
        self.terminal_output.setFont(QFont("Courier", 9))
        self.terminal_output.setStyleSheet("QTextEdit { background-color: #2b2b2b; color: #ffffff; border: 1px solid #555555; }")
        self.terminal_output.setReadOnly(True)
        layout.addWidget(self.terminal_output)
        
        input_layout = QHBoxLayout()
        self.command_input = QLineEdit()
        self.command_input.setFont(QFont("Courier", 9))
        self.command_input.setPlaceholderText("Type your selection or command here...")
        self.command_input.returnPressed.connect(self.send_input)
        
        input_layout.addWidget(QLabel("Input:"))
        input_layout.addWidget(self.command_input)
        
        self.send_button = QPushButton("Send")
        self.send_button.clicked.connect(self.send_input)
        input_layout.addWidget(self.send_button)
        
        layout.addLayout(input_layout)
        
        control_layout = QHBoxLayout()
        self.stop_button = QPushButton("Stop Process")
        self.stop_button.clicked.connect(self.stop_process)
        self.stop_button.setEnabled(False)
        control_layout.addWidget(self.stop_button)
        
        self.clear_button = QPushButton("Clear Terminal")
        self.clear_button.clicked.connect(self.clear_terminal)
        control_layout.addWidget(self.clear_button)
        
        layout.addLayout(control_layout)
        self.setLayout(layout)

    def start_process(self, command, work_dir):
        if self.process and self.process.state() != QProcess.NotRunning:
            self.stop_process()
        
        self.process = QProcess(self)
        self.process.setWorkingDirectory(work_dir)
        self.process.readyReadStandardOutput.connect(self.read_stdout)
        self.process.readyReadStandardError.connect(self.read_stderr)
        self.process.finished.connect(self.process_finished)
        self.process.started.connect(self.process_started)
        
        self.terminal_output.append(f"Starting: {command}")
        self.terminal_output.append(f"Working directory: {work_dir}")
        self.terminal_output.append("-" * 50)
        
        self.process.start("bash", ["-c", command])

    def process_started(self):
        self.stop_button.setEnabled(True)
        self.command_input.setEnabled(True)

    def process_finished(self, exit_code, exit_status):
        self.stop_button.setEnabled(False)
        if exit_code == 0:
            self.terminal_output.append(f"Process completed successfully (exit code: {exit_code})")
        else:
            self.terminal_output.append(f"Process failed (exit code: {exit_code})")
        self.terminal_output.append("=" * 50)

    def read_stdout(self):
        if self.process:
            data = self.process.readAllStandardOutput().data().decode('utf-8', errors='ignore')
            lines = data.splitlines()
            for line in lines:
                if line.strip():
                    self.terminal_output.append(line)
            self.scroll_to_bottom()

    def read_stderr(self):
        if self.process:
            data = self.process.readAllStandardError().data().decode('utf-8', errors='ignore')
            lines = data.splitlines()
            for line in lines:
                if line.strip():
                    self.terminal_output.append(line)
            self.scroll_to_bottom()

    def send_input(self):
        if self.process and self.process.state() == QProcess.Running:
            input_text = self.command_input.text()
            if input_text:
                self.terminal_output.append(f">>> {input_text}")
                self.process.write((input_text + '\n').encode())
                self.command_input.clear()
                self.scroll_to_bottom()

    def stop_process(self):
        if self.process and self.process.state() != QProcess.NotRunning:
            self.process.kill()
            self.process.waitForFinished(3000)
            self.stop_button.setEnabled(False)

    def clear_terminal(self):
        self.terminal_output.clear()

    def scroll_to_bottom(self):
        cursor = self.terminal_output.textCursor()
        cursor.movePosition(QTextCursor.End)
        self.terminal_output.setTextCursor(cursor)

class MultiPlotter(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.selected_files = []
        self.custom_legend_names = {}  # Store custom legend names
        self.current_ax = None  # Store current axis for legend updates
        self.current_plot_type = None  # Store current plot type
        
        # Using your plotXVG.py configurations
        self.colors = ['#1f77b4', '#d62728', '#2ca02c', '#ff7f0e',
                      '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
                      '#bcbd22', '#17becf']
        
        # Plot type detection patterns from your script
        self.plot_patterns = {
            'hbond': ['hbond', 'hbnum', 'hydrogen_bond', 'hydrogen-bond', 'h-bond', 'hb-', 'bond'],
            'rmsd': ['rmsd', 'root_mean_square_deviation', 'backbone_rmsd'],
            'rmsf': ['rmsf', 'root_mean_square_fluctuation', 'residue_fluctuation', 'bfactor'],
            'sasa': ['sasa', 'solvent_accessible_surface_area', 'surface_area', 'area'],
            'gyration': ['gyrate', 'radius_of_gyration', 'rg', 'gyration_radius'],
            'mindist': ['mindist', 'minimum_distance', 'min_dist', 'distance']
        }
        
        # Plot configurations from your script
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
        
    def init_ui(self):
        layout = QVBoxLayout()
        
        # File selection section
        file_group = QGroupBox("File Selection")
        file_layout = QVBoxLayout()
        
        file_btn_layout = QHBoxLayout()
        self.add_files_btn = QPushButton("Add XVG Files")
        self.add_files_btn.clicked.connect(self.add_files)
        file_btn_layout.addWidget(self.add_files_btn)
        
        self.clear_files_btn = QPushButton("Clear Files")
        self.clear_files_btn.clicked.connect(self.clear_files)
        file_btn_layout.addWidget(self.clear_files_btn)
        
        file_layout.addLayout(file_btn_layout)
        
        self.file_list = QListWidget()
        self.file_list.setMaximumHeight(150)
        file_layout.addWidget(self.file_list)
        
        file_group.setLayout(file_layout)
        layout.addWidget(file_group)
        
        # Analysis type selection
        analysis_group = QGroupBox("Analysis Type")
        analysis_layout = QHBoxLayout()
        
        analysis_layout.addWidget(QLabel("Plot Type:"))
        self.analysis_combo = QComboBox()
        self.analysis_combo.addItems([
            "Auto-detect", "RMSD", "RMSF", "SASA", "Gyration", 
            "Hydrogen Bonds", "Minimum Distance", "PCA 2D Projection", "Generic"
        ])
        analysis_layout.addWidget(self.analysis_combo)
        
        self.plot_btn = QPushButton("Generate Plot")
        self.plot_btn.clicked.connect(self.generate_plot)
        analysis_layout.addWidget(self.plot_btn)
        
        analysis_group.setLayout(analysis_layout)
        layout.addWidget(analysis_group)
        
        # Plot display
        self.figure = Figure(figsize=(12, 8))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        
        # Control buttons
        control_layout = QHBoxLayout()
        
        self.save_plot_btn = QPushButton("Save Plot")
        self.save_plot_btn.clicked.connect(self.save_plot)
        control_layout.addWidget(self.save_plot_btn)
        
        # Legend editing button (initially hidden)
        self.edit_legend_btn = QPushButton("Edit Legend")
        self.edit_legend_btn.clicked.connect(self.edit_legend)
        self.edit_legend_btn.setVisible(False)
        control_layout.addWidget(self.edit_legend_btn)
        
        control_layout.addStretch()
        layout.addLayout(control_layout)
        
        self.setLayout(layout)

    def add_files(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select XVG Files", "", "XVG Files (*.xvg);;All Files (*)"
        )
        for file in files:
            if file not in self.selected_files:
                self.selected_files.append(file)
                self.file_list.addItem(os.path.basename(file))
                # Initialize custom legend name with filename stem
                self.custom_legend_names[file] = Path(file).stem

    def clear_files(self):
        self.selected_files.clear()
        self.file_list.clear()
        self.custom_legend_names.clear()
        self.edit_legend_btn.setVisible(False)

    def edit_legend(self):
        if not self.selected_files or len(self.selected_files) < 2:
            QMessageBox.information(self, "Info", "Legend editing is only available for multiple files")
            return
        
        # Get current legend names
        current_labels = [self.custom_legend_names.get(file, Path(file).stem) 
                         for file in self.selected_files]
        
        # Open legend edit dialog
        dialog = LegendEditDialog(current_labels, self)
        if dialog.exec_() == QDialog.Accepted:
            # Update custom legend names
            new_names = dialog.get_legend_names()
            for file, new_name in zip(self.selected_files, new_names):
                self.custom_legend_names[file] = new_name
            
            # Regenerate plot with new legend
            self.update_legend()

    def update_legend(self):
        """Update only the legend without regenerating the entire plot"""
        if self.current_ax and len(self.selected_files) > 1:
            # Get current legend names
            legend_labels = [self.custom_legend_names.get(file, Path(file).stem) 
                           for file in self.selected_files]
            
            # Update legend
            handles = self.current_ax.lines if self.current_plot_type != 'pca' else []
            if handles:
                self.current_ax.legend(handles, legend_labels)
            
            self.canvas.draw()

    def detect_plot_type(self, filename, comments):
        """Detect plot type using your plotXVG.py logic"""
        filename_lower = filename.lower()
        comments_str = ' '.join(comments).lower()
        
        print(f"Debug - Detecting plot type for: {filename_lower}")
        print(f"Debug - Comments: {comments_str}")
        
        # Check patterns in order of specificity
        for plot_type, patterns in self.plot_patterns.items():
            for pattern in patterns:
                if pattern in filename_lower:
                    print(f"Detected '{plot_type}' from filename pattern '{pattern}'")
                    return plot_type
                elif pattern in comments_str:
                    print(f"Detected '{plot_type}' from comment pattern '{pattern}'")
                    return plot_type
        
        print(f"No specific pattern detected, using generic plot for {filename}")
        return 'generic'

    def read_xvg(self, filename):
        """Read XVG file using your plotXVG.py logic"""
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

    def convert_time_to_ns(self, time_data, comments):
        """Convert time units to nanoseconds based on XVG comments"""
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

    def format_axes(self, ax, plot_type='generic', x_data=None, y_data=None):
        """Format axes using your plotXVG.py logic with RMSF fix"""
        if plot_type == 'rmsf':
            if x_data is not None and y_data is not None:
                # Ensure x_data and y_data are numpy arrays
                x_data = np.array(x_data)
                y_data = np.array(y_data)
                
                # X-axis formatting
                x_min, x_max = x_data.min(), x_data.max()
                x_range = x_max - x_min
                ax.set_xlim(x_min - x_range * 0.01, x_max + x_range * 0.01)
                
                # Y-axis formatting
                y_min, y_max = y_data.min(), y_data.max()
                y_range = y_max - y_min
                y_padding = y_range * 0.05
                y_start = max(0, y_min - y_padding)
                ax.set_ylim(y_start, y_max + y_padding)
                
                # X-axis tick spacing - FIXED
                if x_range <= 50:
                    x_tick_spacing = 5
                elif x_range <= 200:
                    x_tick_spacing = 20
                elif x_range <= 500:
                    x_tick_spacing = 50
                else:
                    x_tick_spacing = 100
                
                # Apply the locator with explicit force
                ax.xaxis.set_major_locator(MultipleLocator(x_tick_spacing))
                ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
                
                # Force tick update
                ax.xaxis.get_major_locator().set_params(base=x_tick_spacing)
                
                print(f"RMSF formatting: x_range={x_range}, x_tick_spacing={x_tick_spacing}")
                print(f"X limits: {ax.get_xlim()}")
                
        elif plot_type in ['rmsd', 'sasa', 'gyration', 'mindist', 'hbond']:
            if x_data is not None and y_data is not None:
                # Ensure data are numpy arrays
                x_data = np.array(x_data)
                y_data = np.array(y_data)
                
                x_max = x_data.max()
                ax.set_xlim(0, x_max)
                
                if x_max <= 50:
                    x_tick_spacing = 10
                elif x_max <= 100:
                    x_tick_spacing = 20
                elif x_max <= 500:
                    x_tick_spacing = 50
                else:
                    x_tick_spacing = 100
                
                ax.xaxis.set_major_locator(MultipleLocator(x_tick_spacing))
                
                y_min, y_max = y_data.min(), y_data.max()
                y_range = y_max - y_min
                y_padding = y_range * 0.05
                
                if y_min >= 0 and y_min <= y_range * 0.1:
                    y_start = 0
                else:
                    y_start = y_min - y_padding
                
                ax.set_ylim(y_start, y_max + y_padding)
                ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
        else:
            if x_data is not None and y_data is not None:
                # Ensure data are numpy arrays
                x_data = np.array(x_data)
                y_data = np.array(y_data)
                
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

    def generate_plot(self):
        if not self.selected_files:
            QMessageBox.warning(self, "Warning", "Please select at least one XVG file")
            return
            
        self.figure.clear()
        
        plot_type_selection = self.analysis_combo.currentText().lower().replace(" ", "_")
        if plot_type_selection == "auto-detect":
            plot_type_selection = None
        
        # Read first file to determine plot type
        plot_type = None
        if plot_type_selection is None:
            for filename in self.selected_files:
                df, comments = self.read_xvg(filename)
                if df is not None:
                    plot_type = self.detect_plot_type(filename, comments)
                    break
        else:
            # Map combo box selection to internal types
            combo_to_type = {
                'rmsd': 'rmsd',
                'rmsf': 'rmsf', 
                'sasa': 'sasa',
                'gyration': 'gyration',
                'hydrogen_bonds': 'hbond',
                'minimum_distance': 'mindist',
                'pca_2d_projection': 'pca',
                'generic': 'generic'
            }
            plot_type = combo_to_type.get(plot_type_selection, 'generic')
        
        self.current_plot_type = plot_type
        print(f"Final plot type: {plot_type}")
        
        # Handle PCA scatter plot with your exact code
        if plot_type == 'pca' or any("pca" in filename.lower() or "2d" in filename.lower() for filename in self.selected_files):
            ax = self.figure.add_subplot(111)
            
            # Process first PCA file only for scatter plot
            for filename in self.selected_files:
                if "pca" in filename.lower() or "2d" in filename.lower():
                    # Use your EXACT scatter plot code
                    data = []
                    with open(filename) as file:
                        for line in file:
                            if line.startswith(('@', '#')):
                                continue
                            parts = line.split()
                            if len(parts) >= 2:
                                try:
                                    data.append([float(parts[0]), float(parts[1])])
                                except ValueError:
                                    pass
                    
                    if data:
                        df = pd.DataFrame(data, columns=["PC1", "PC2"])
                        sns.scatterplot(x="PC1", y="PC2", data=df, ax=ax)
                        ax.set_xlabel("Projection on eigenvector 1 (nm)", fontsize=16)
                        ax.set_ylabel("Projection on eigenvector 2 (nm)", fontsize=16)
                        ax.set_title("2D projection of trajectory", fontsize=16)
                    break
            
            self.current_ax = ax
            # No legend for PCA scatter plots
            self.edit_legend_btn.setVisible(False)
            self.canvas.draw()
            return
        
        # For line plots
        ax = self.figure.add_subplot(111)
        all_x_data = []
        all_y_data = []
        
        for i, filename in enumerate(self.selected_files):
            df, comments = self.read_xvg(filename)
            if df is None:
                continue
            
            # Convert time if needed
            if plot_type in ['rmsd', 'sasa', 'gyration', 'mindist', 'hbond']:
                x_data = self.convert_time_to_ns(df.iloc[:, 0], comments)
            else:
                x_data = df.iloc[:, 0]
            
            y_data = df.iloc[:, 1]
            
            all_x_data.extend(x_data)
            all_y_data.extend(y_data)
            
            # Use custom legend name if available, otherwise use filename stem
            label = self.custom_legend_names.get(filename, Path(filename).stem)
            color = self.colors[i % len(self.colors)]
            
            ax.plot(x_data, y_data, color=color, linewidth=1.5, label=label, alpha=0.8)
        
        # Set labels based on plot type using your configurations
        if plot_type in self.plot_configs:
            config = self.plot_configs[plot_type]
            ax.set_xlabel(config['xlabel'], fontsize=12)
            ax.set_ylabel(config['ylabel'], fontsize=12)
            ax.set_title(config['title'], fontsize=14, fontweight='bold')
        else:
            ax.set_xlabel('X Data', fontsize=12)
            ax.set_ylabel('Y Data', fontsize=12)
            ax.set_title('XVG Data Plot', fontsize=14, fontweight='bold')
        
        # Format axes using your logic with RMSF fix
        if all_x_data and all_y_data:
            self.format_axes(ax, plot_type, np.array(all_x_data), np.array(all_y_data))
        
        # Add legend and grid
        if len(self.selected_files) > 1:
            ax.legend()
            self.edit_legend_btn.setVisible(True)
        else:
            self.edit_legend_btn.setVisible(False)
            
        ax.grid(True, alpha=0.3)
        
        # Store current axis for legend updates
        self.current_ax = ax
        
        plt.tight_layout()
        self.canvas.draw()

    def save_plot(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Plot", "plot.png",
            "PNG Files (*.png);;PDF Files (*.pdf);;SVG Files (*.svg)",
            options=options
        )
        
        if file_path:
            self.figure.savefig(file_path, dpi=600, bbox_inches='tight', facecolor='white')
            QMessageBox.information(self, "Success", f"Plot saved as {file_path} (600 DPI)")

class FELPlotter(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Control buttons
        btn_layout = QHBoxLayout()
        self.load_btn = QPushButton("Load XPM File")
        self.load_btn.clicked.connect(self.load_xpm_file)
        btn_layout.addWidget(self.load_btn)
        
        self.save_btn = QPushButton("Save Plot")
        self.save_btn.clicked.connect(self.save_plot)
        btn_layout.addWidget(self.save_btn)
        btn_layout.addStretch()
        
        layout.addLayout(btn_layout)
        
        # Plot display
        self.figure = Figure(figsize=(16, 8))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        
        self.setLayout(layout)

    def unquote(self, s):
        """Remove quotes from XPM strings"""
        return s[s.find('"')+1:s.rfind('"')]

    def parse_xpm(self, xpm_file):
        """Parse GROMACS XPM file and return energy matrix"""
        with open(xpm_file) as f:
            lines = f.readlines()

        # Find matrix dimensions and color legend
        for i, line in enumerate(lines):
            if line.strip().startswith('"') and line.count('"') >= 2:
                header = self.unquote(line)
                break

        nx, ny, ncolors, nchar = [int(x) for x in header.split()]
        
        color_lines = lines[i+1:i+1+ncolors]
        
        # Build color symbol -> value mapping
        color_map = {}
        for cl in color_lines:
            parts = cl.strip().split('c')
            symbol = parts[0].replace('"','').strip()
            if '/*' in cl:
                value = cl.split('/*')[-1].split('"')[1]
                color_map[symbol] = float(value)
            else:
                color_map[symbol] = np.nan

        # Extract the matrix
        matrix_lines = []
        for line in lines[i+1+ncolors:]:
            if line.strip().startswith('"'):
                matrix_lines.append(self.unquote(line).replace(',', ''))

        matrix = np.zeros((ny, nx))
        for y, row in enumerate(matrix_lines):
            for x in range(nx):
                symbol = row[x*nchar:(x+1)*nchar]
                matrix[y, x] = color_map[symbol]

        return matrix, nx, ny

    def create_smooth_surface(self, matrix, nx, ny):
        """Create smooth high-resolution surface"""
        pc1_orig = np.linspace(-3, 3, nx)
        pc2_orig = np.linspace(-3, 3, ny)
        
        # Handle NaN values
        mask = ~np.isnan(matrix)
        if not np.all(mask):
            from scipy.interpolate import griddata
            points = np.array([[pc1_orig[j], pc2_orig[i]] for i in range(ny) for j in range(nx) if mask[i,j]])
            values = np.array([matrix[i,j] for i in range(ny) for j in range(nx) if mask[i,j]])
            xi, yi = np.meshgrid(pc1_orig, pc2_orig)
            matrix = griddata(points, values, (xi, yi), method='cubic', fill_value=np.nanmax(values))

        # Smoothing
        matrix_smooth = gaussian_filter(matrix, sigma=0.5)
        
        # High resolution for smooth surface
        pc1_hires = np.linspace(-3, 3, nx*2)
        pc2_hires = np.linspace(-3, 3, ny*2)
        
        spline = RectBivariateSpline(pc2_orig, pc1_orig, matrix_smooth, kx=2, ky=2)
        matrix_hires = spline(pc2_hires, pc1_hires)
        
        return matrix_smooth, matrix_hires, pc1_orig, pc2_orig, pc1_hires, pc2_hires

    def plot_2d_3d_fel(self, matrix_smooth, matrix_hires, pc1_orig, pc2_orig, pc1_hires, pc2_hires):
        """Create side-by-side 2D and 3D FEL plots"""
        self.figure.clear()
        
        # Create meshgrids
        X_hires, Y_hires = np.meshgrid(pc1_hires, pc2_hires)
        X_orig, Y_orig = np.meshgrid(pc1_orig, pc2_orig)
        
        # 2D plot on the left
        ax1 = self.figure.add_subplot(121)
        matrix_flipped = np.flipud(matrix_smooth)
        
        im = ax1.imshow(matrix_flipped, extent=[-3, 3, -3, 3],
                       origin='lower', cmap='jet', aspect='equal')
        
        ax1.set_xlabel('PC1', fontsize=14)
        ax1.set_ylabel('PC2', fontsize=14)
        ax1.grid(True, alpha=0.3)
        
        cbar1 = plt.colorbar(im, ax=ax1, orientation='horizontal',
                            shrink=0.8, aspect=30, pad=0.15)
        cbar1.set_label('Gibbs Free Energy (kJ/mol)', fontsize=12)
        
        # 3D plot on the right
        ax2 = self.figure.add_subplot(122, projection='3d')
        
        # Base heatmap
        matrix_flipped_3d = np.flipud(matrix_smooth)
        ax2.contourf(X_orig, Y_orig, matrix_flipped_3d,
                    zdir='z', offset=0, cmap='jet', alpha=0.8, levels=50)
        
        # 3D surface
        surf = ax2.plot_surface(X_hires, Y_hires, matrix_hires,
                               cmap='jet', alpha=0.85, linewidth=0.1,
                               antialiased=True, shade=True,
                               rstride=2, cstride=2, edgecolors='none')
        
        ax2.view_init(elev=30, azim=45)
        ax2.set_xlabel('PC1', fontsize=14, labelpad=10)
        ax2.set_ylabel('PC2', fontsize=14, labelpad=10)
        ax2.set_zlabel('Gibbs Free Energy (kJ/mol)', fontsize=12, labelpad=10)
        ax2.set_xlim(-3, 3)
        ax2.set_ylim(-3, 3)
        ax2.set_zlim(0, np.max(matrix_hires) * 1.1)
        
        # Styling
        ax2.grid(True, alpha=0.3, linewidth=0.5, color='gray')
        
        cbar2 = plt.colorbar(surf, ax=ax2, shrink=0.6, aspect=20, pad=0.1)
        cbar2.set_label('Gibbs Free Energy (kJ/mol)', fontsize=12)
        
        plt.tight_layout()
        self.canvas.draw()

    def load_xpm_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select XPM File", "", "XPM Files (*.xpm);;All Files (*)"
        )
        
        if file_path:
            try:
                matrix, nx, ny = self.parse_xpm(file_path)
                matrix_smooth, matrix_hires, pc1_orig, pc2_orig, pc1_hires, pc2_hires = self.create_smooth_surface(matrix, nx, ny)
                self.plot_2d_3d_fel(matrix_smooth, matrix_hires, pc1_orig, pc2_orig, pc1_hires, pc2_hires)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load XPM file: {str(e)}")

    def save_plot(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Plot", "fel_plot.png",
            "PNG Files (*.png);;PDF Files (*.pdf);;SVG Files (*.svg)",
            options=options
        )
        
        if file_path:
            self.figure.savefig(file_path, dpi=300, bbox_inches='tight', facecolor='white')
            QMessageBox.information(self, "Success", f"Plot saved as {file_path}")

class XVGPlotter(QWidget):
    def __init__(self):
        super().__init__()
        self.figure = Figure(figsize=(8, 6))
        self.canvas = FigureCanvas(self.figure)
        
        main_layout = QVBoxLayout()
        
        plot_layout = QHBoxLayout()
        plot_layout.addWidget(self.canvas, stretch=3)
        
        self.avg_group = QGroupBox("Average Value")
        avg_layout = QVBoxLayout()
        self.avg_label = QLabel("No data")
        self.avg_label.setFont(QFont("Arial", 12, QFont.Bold))
        self.avg_label.setAlignment(Qt.AlignCenter)
        self.avg_label.setStyleSheet("""
            QLabel {
                background-color: #f0f0f0;
                border: 2px solid #cccccc;
                border-radius: 5px;
                padding: 10px;
                color: #333333;
            }
        """)
        avg_layout.addWidget(self.avg_label)
        self.avg_group.setLayout(avg_layout)
        self.avg_group.setFixedWidth(150)
        
        plot_layout.addWidget(self.avg_group, stretch=0)
        main_layout.addLayout(plot_layout)
        
        self.save_button = QPushButton("Save Plot as PNG")
        self.save_button.clicked.connect(self.save_plot)
        main_layout.addWidget(self.save_button)
        
        self.setLayout(main_layout)

    def calculate_average_value(self, filename):
        try:
            awk_command = "awk '{ if ($1 != \"#\" && $1 != \"@\") total += $2; count++ } END { if (count > 0) print total/count }'"
            full_command = f"{awk_command} '{filename}'"
            result = subprocess.run(full_command, shell=True, capture_output=True, text=True)
            if result.returncode == 0 and result.stdout.strip():
                avg_value = float(result.stdout.strip())
                return avg_value
            else:
                return None
        except Exception:
            return None

    def get_analysis_title(self, filename: str) -> str:
        fname = filename.lower()
        if "rmsd" in fname:
            return "RMSD"
        elif "rmsf" in fname:
            return "RMSF"
        elif "gyrate" in fname:
            return "Radius of Gyration"
        elif "sasa" in fname or "area" in fname:
            return "SASA"
        elif "hbond" in fname:
            return "Hydrogen Bonds"
        elif "mindist" in fname:
            return "Minimum Distance"
        elif "pca" in fname and "2d" in fname:
            return "PCA 2D Projection"
        elif "pca" in fname:
            return "PCA"
        else:
            return "Analysis Result"

    def plot_xvg(self, filename):
        self.figure.clear()
        self.current_title = self.get_analysis_title(filename)
        ax = self.figure.add_subplot(111)
        
        avg_value = self.calculate_average_value(filename)
        if avg_value is not None:
            self.avg_label.setText(f"{avg_value:.4f}")
        else:
            self.avg_label.setText("N/A")

        try:
            data = []
            with open(filename, 'r') as f:
                for line in f:
                    if line.startswith('@') or line.startswith('#'):
                        continue
                    parts = line.split()
                    if len(parts) >= 2:
                        try:
                            data.append([float(parts[0]), float(parts[1])])
                        except ValueError:
                            continue

            if data:
                data = np.array(data)
                ax.plot(data[:, 0], data[:, 1], 'b-', linewidth=1)
                ax.set_xlabel("Time (ns)")
                ax.set_ylabel("Value")
                ax.set_title(self.current_title)
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, 'No valid data found in file',
                       transform=ax.transAxes, ha='center', va='center',
                       fontsize=12, color='red')

        except Exception as e:
            ax.text(0.5, 0.5, f'Error loading file:\n{str(e)}',
                   transform=ax.transAxes, ha='center', va='center',
                   fontsize=10, color='red')
            ax.set_title(f"Error - {self.current_title}")

        self.canvas.draw()

    def plot_scatter(self, filename, title=None):
        """Plot scatter plot using your exact scatter_plot_xvg.py script logic"""
        self.figure.clear()
        self.current_title = title or "2D projection of trajectory"
        self.avg_label.setText("N/A")

        try:
            # EXACT same parsing logic as your script
            data = []
            with open(filename) as file:
                for line in file:
                    if line.startswith(('@', '#')):
                        continue
                    parts = line.split()
                    if len(parts) >= 2:
                        try:
                            data.append([float(parts[0]), float(parts[1])])
                        except ValueError:
                            pass

            if data:
                # Convert to DataFrame with exact column names from your script
                df = pd.DataFrame(data, columns=["PC1", "PC2"])
                
                # EXACT plotting as your script
                ax = self.figure.add_subplot(111)
                sns.scatterplot(x="PC1", y="PC2", data=df, ax=ax)
                ax.set_xlabel("Projection on eigenvector 1 (nm)", fontsize=16)
                ax.set_ylabel("Projection on eigenvector 2 (nm)", fontsize=16)
                ax.set_title("2D projection of trajectory", fontsize=16)
                
            else:
                ax = self.figure.add_subplot(111)
                ax.text(0.5, 0.5, 'No valid data found in file',
                       transform=ax.transAxes, ha='center', va='center',
                       fontsize=12, color='red')

        except Exception as e:
            ax = self.figure.add_subplot(111)
            ax.text(0.5, 0.5, f'Error loading file:\n{str(e)}',
                   transform=ax.transAxes, ha='center', va='center',
                   fontsize=10, color='red')
            ax.set_title(f"Error - {self.current_title}")

        self.canvas.draw()

    def save_plot(self):
        if not hasattr(self, 'current_title'):
            self.current_title = "plot"
        
        options = QFileDialog.Options()
        default_name = f"{self.current_title.replace(' ', '_').replace('/', '_')}.png"
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Plot As PNG", default_name,
            "PNG Files (*.png);;PDF Files (*.pdf);;SVG Files (*.svg)",
            options=options
        )
        
        if file_path:
            self.figure.savefig(file_path, dpi=300, bbox_inches='tight', facecolor='white')
            QMessageBox.information(self, "Success", f"Plot saved as {file_path}")

class AnalysisWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.work_dir = ""
        self.selected_files = {}
        self.index_file = ""
        self.last_output_file = ""
        self.init_ui()

    def init_ui(self):
        main_layout = QHBoxLayout()
        
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)
        
        # Left panel
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        # Directory selection
        dir_group = QGroupBox("Working Directory")
        dir_layout = QVBoxLayout(dir_group)
        
        self.dir_label = QLabel("No directory selected")
        self.dir_label.setWordWrap(True)
        dir_layout.addWidget(self.dir_label)
        
        select_dir_btn = QPushButton("Select Directory")
        select_dir_btn.clicked.connect(self.select_directory)
        dir_layout.addWidget(select_dir_btn)
        
        left_layout.addWidget(dir_group)
        
        # File selection
        file_group = QGroupBox("MD Files")
        file_layout = QVBoxLayout(file_group)
        
        self.file_list = QListWidget()
        self.file_list.itemClicked.connect(self.select_file)
        file_layout.addWidget(self.file_list)
        
        self.selected_label = QLabel("Selected files will appear here")
        self.selected_label.setWordWrap(True)
        file_layout.addWidget(self.selected_label)
        
        left_layout.addWidget(file_group)
        
        # Analysis options
        analysis_group = QGroupBox("Analysis Options")
        analysis_layout = QGridLayout(analysis_group)
        self.create_analysis_buttons(analysis_layout)
        left_layout.addWidget(analysis_group)
        
        # Time range
        time_range_group = QGroupBox("Time Range (ns)")
        time_range_layout = QGridLayout(time_range_group)
        
        time_range_layout.addWidget(QLabel("Start time (ns):"), 0, 0)
        self.start_time = QSpinBox()
        self.start_time.setMaximum(10000)
        time_range_layout.addWidget(self.start_time, 0, 1)
        
        time_range_layout.addWidget(QLabel("End time (ns):"), 1, 0)
        self.end_time = QSpinBox()
        self.end_time.setMaximum(10000)
        self.end_time.setValue(100)
        time_range_layout.addWidget(self.end_time, 1, 1)
        
        left_layout.addWidget(time_range_group)
        
        splitter.addWidget(left_panel)
        
        # Right panel with tabs
        right_panel = QTabWidget()
        
        # Terminal tab
        self.terminal = InteractiveTerminal()
        self.terminal.process_finished = self.on_terminal_process_finished
        right_panel.addTab(self.terminal, "Interactive Terminal")
        
        # Results tab
        results_widget = QWidget()
        results_layout = QVBoxLayout(results_widget)
        
        # Plot widget
        self.plotter = XVGPlotter()
        results_layout.addWidget(self.plotter)
        
        # Plot controls
        plot_controls = QHBoxLayout()
        
        self.plot_btn = QPushButton("Plot Last XVG")
        self.plot_btn.clicked.connect(self.plot_last_xvg)
        plot_controls.addWidget(self.plot_btn)
        
        self.clear_plot_btn = QPushButton("Clear Plot")
        self.clear_plot_btn.clicked.connect(self.clear_plot)
        plot_controls.addWidget(self.clear_plot_btn)
        
        self.auto_plot_cb = QCheckBox("Auto-plot results")
        self.auto_plot_cb.setChecked(True)
        plot_controls.addWidget(self.auto_plot_cb)
        
        results_layout.addLayout(plot_controls)
        
        right_panel.addTab(results_widget, "Results & Plots")
        
        splitter.addWidget(right_panel)
        splitter.setSizes([350, 1250])
        
        self.setLayout(main_layout)

    def create_analysis_buttons(self, layout):
        buttons = [
            ("Center Trajectory", self.run_center),
            ("RMSD Analysis", self.run_rmsd),
            ("RMSF Analysis", self.run_rmsf),
            ("Radius of Gyration", self.run_gyration),
            ("SASA Analysis", self.run_sasa),
            ("H-Bond Analysis", self.run_hbond),
            ("Min Distance", self.run_mindist),
            ("PCA/FEL Analysis", self.run_pca)
        ]
        
        row, col = 0, 0
        for name, func in buttons:
            btn = QPushButton(name)
            btn.clicked.connect(func)
            btn.setMinimumHeight(30)
            layout.addWidget(btn, row, col)
            col += 1
            if col > 1:
                col = 0
                row += 1

    def select_directory(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select MD Files Directory")
        if dir_path:
            self.work_dir = dir_path
            self.dir_label.setText(f"Directory: {dir_path}")
            self.load_file_list()

    def load_file_list(self):
        self.file_list.clear()
        if not self.work_dir:
            return
            
        try:
            files = os.listdir(self.work_dir)
            md_extensions = ['.tpr', '.xtc', '.trr', '.gro', '.pdb', '.ndx', '.xvg', '.edr']
            for file in sorted(files):
                if any(file.endswith(ext) for ext in md_extensions):
                    self.file_list.addItem(file)
        except Exception as e:
            print(f"Error loading directory: {str(e)}")

    def select_file(self, item):
        filename = item.text()
        if filename.endswith('.tpr'):
            self.selected_files['tpr'] = filename
        elif filename.endswith(('.xtc', '.trr')):
            self.selected_files['traj'] = filename
        elif filename.endswith('.ndx'):
            self.selected_files['index'] = filename
        elif filename.endswith('.edr'):
            self.selected_files['energy'] = filename
        self.update_selected_display()

    def update_selected_display(self):
        display_text = "Selected files:\n"
        for file_type, filename in self.selected_files.items():
            display_text += f"• {file_type.upper()}: {filename}\n"
        self.selected_label.setText(display_text)

    def get_base_command_files(self):
        if 'tpr' not in self.selected_files:
            raise ValueError("No TPR file selected")
        if 'traj' not in self.selected_files:
            raise ValueError("No trajectory file selected")
        return self.selected_files['tpr'], self.selected_files['traj']

    def get_time_range_args(self):
        start = self.start_time.value()
        end = self.end_time.value()
        return f"-b {start} -e {end}"

    def run_interactive_command(self, command, description):
        if not self.work_dir:
            QMessageBox.warning(self, "Warning", "Please select a working directory first")
            return
            
        try:
            self.get_base_command_files()
        except ValueError as e:
            QMessageBox.warning(self, "Warning", str(e))
            return
            
        self.terminal.start_process(command, self.work_dir)

    def run_center(self):
        try:
            tpr_file, traj_file = self.get_base_command_files()
            base_name = traj_file.rsplit('.', 1)[0]
            output_file = f"{base_name}_center.xtc"
            command = f"gmx trjconv -s {tpr_file} -f {traj_file} -o {output_file} -center -pbc mol -ur compact"
            self.run_interactive_command(command, "Trajectory Centering")
        except ValueError as e:
            QMessageBox.warning(self, "Warning", str(e))

    def run_rmsd(self):
        try:
            tpr_file, traj_file = self.get_base_command_files()
            time_args = self.get_time_range_args()
            command = f"gmx rms -s {tpr_file} -f {traj_file} -o rmsd.xvg -tu ns {time_args}"
            if 'index' in self.selected_files:
                command += f" -n {self.selected_files['index']}"
            self.run_interactive_command(command, "RMSD Analysis")
        except ValueError as e:
            QMessageBox.warning(self, "Warning", str(e))

    def run_rmsf(self):
        try:
            tpr_file, traj_file = self.get_base_command_files()
            time_args = self.get_time_range_args()
            command = f"gmx rmsf -s {tpr_file} -f {traj_file} -o rmsf.xvg -res {time_args}"
            if 'index' in self.selected_files:
                command += f" -n {self.selected_files['index']}"
            self.run_interactive_command(command, "RMSF Analysis")
        except ValueError as e:
            QMessageBox.warning(self, "Warning", str(e))

    def run_gyration(self):
        try:
            tpr_file, traj_file = self.get_base_command_files()
            time_args = self.get_time_range_args()
            command = f"gmx gyrate -s {tpr_file} -f {traj_file} -o gyrate.xvg {time_args}"
            if 'index' in self.selected_files:
                command += f" -n {self.selected_files['index']}"
            self.run_interactive_command(command, "Radius of Gyration")
        except ValueError as e:
            QMessageBox.warning(self, "Warning", str(e))

    def run_sasa(self):
        try:
            tpr_file, traj_file = self.get_base_command_files()
            time_args = self.get_time_range_args()
            command = f"gmx sasa -s {tpr_file} -f {traj_file} -o area.xvg -tu ns {time_args}"
            if 'index' in self.selected_files:
                command += f" -n {self.selected_files['index']}"
            self.run_interactive_command(command, "SASA Analysis")
        except ValueError as e:
            QMessageBox.warning(self, "Warning", str(e))

    def run_hbond(self):
        try:
            tpr_file, traj_file = self.get_base_command_files()
            time_args = self.get_time_range_args()
            command = f"gmx hbond -s {tpr_file} -f {traj_file} -num hbond.xvg -tu ns {time_args}"
            if 'index' in self.selected_files:
                command += f" -n {self.selected_files['index']}"
            self.run_interactive_command(command, "Hydrogen Bond Analysis")
        except ValueError as e:
            QMessageBox.warning(self, "Warning", str(e))

    def run_mindist(self):
        try:
            tpr_file, traj_file = self.get_base_command_files()
            time_args = self.get_time_range_args()
            command = f"gmx mindist -f {traj_file} -s {tpr_file} -od mindist.xvg -tu ns {time_args}"
            if 'index' in self.selected_files:
                command += f" -n {self.selected_files['index']}"
            self.run_interactive_command(command, "Minimum Distance Analysis")
        except ValueError as e:
            QMessageBox.warning(self, "Warning", str(e))

    def run_pca(self):
        try:
            tpr_file, traj_file = self.get_base_command_files()
            start_time = self.start_time.value()
            end_time = self.end_time.value()
            index_part = f" -n {self.selected_files['index']}" if 'index' in self.selected_files else ""
            
            pca_script = f"""#!/bin/bash
gmx covar -f {traj_file} -s {tpr_file} -o eigenval.xvg -v eigenvec.trr -b {start_time} -e {end_time} -tu ns{index_part}
if [ $? -eq 0 ]; then
    gmx anaeig -v eigenvec.trr -f {traj_file} -eig eigenval.xvg -s {tpr_file} -first 1 -last 2 -2d PCA_2d.xvg{index_part}
    if [ $? -eq 0 ]; then
        gmx sham -f PCA_2d.xvg -ls PCA_sham.xpm -notime
        gmx xpm2ps -f PCA_sham.xpm -o PCA_sham.eps -rainbow red
    fi
fi
"""
            
            script_path = os.path.join(self.work_dir, "pca_workflow.sh")
            with open(script_path, 'w') as f:
                f.write(pca_script)
            os.chmod(script_path, 0o755)
            
            self.run_interactive_command("bash pca_workflow.sh", "PCA/FEL Analysis Workflow")
            
        except ValueError as e:
            QMessageBox.warning(self, "Warning", str(e))
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to create PCA workflow: {str(e)}")

    def plot_last_xvg(self):
        if not self.last_output_file or not os.path.exists(self.last_output_file):
            if self.work_dir:
                xvg_files = [f for f in os.listdir(self.work_dir) if f.endswith('.xvg')]
                if xvg_files:
                    xvg_files.sort(key=lambda x: os.path.getmtime(os.path.join(self.work_dir, x)), reverse=True)
                    self.last_output_file = os.path.join(self.work_dir, xvg_files[0])
                else:
                    QMessageBox.information(self, "Info", "No XVG file found in working directory")
                    return
            else:
                QMessageBox.information(self, "Info", "No XVG file available to plot")
                return
                
        filename = os.path.basename(self.last_output_file)
        if "PCA_2d" in filename or "2d" in filename.lower():
            self.plotter.plot_scatter(self.last_output_file)
        else:
            self.plotter.plot_xvg(self.last_output_file)

    def clear_plot(self):
        self.plotter.figure.clear()
        self.plotter.canvas.draw()

    def on_terminal_process_finished(self, exit_code, exit_status):
        InteractiveTerminal.process_finished(self.terminal, exit_code, exit_status)
        
        if exit_code == 0:
            # Check if PCA analysis completed and show FEL plot
            if self.work_dir and os.path.exists(os.path.join(self.work_dir, "PCA_sham.xpm")):
                self.show_fel_window()
                
            # Auto-plot if enabled
            if self.auto_plot_cb.isChecked():
                self.find_and_plot_recent_xvg()

    def show_fel_window(self):
        """Show FEL plotting window when PCA analysis is complete"""
        try:
            fel_window = QMainWindow()
            fel_window.setWindowTitle("Free Energy Landscape - PCA Results")
            fel_window.setGeometry(200, 200, 1200, 700)
            
            fel_plotter = FELPlotter()
            fel_window.setCentralWidget(fel_plotter)
            
            # Auto-load the PCA_sham.xpm file
            xpm_file = os.path.join(self.work_dir, "PCA_sham.xpm")
            if os.path.exists(xpm_file):
                matrix, nx, ny = fel_plotter.parse_xpm(xpm_file)
                matrix_smooth, matrix_hires, pc1_orig, pc2_orig, pc1_hires, pc2_hires = fel_plotter.create_smooth_surface(matrix, nx, ny)
                fel_plotter.plot_2d_3d_fel(matrix_smooth, matrix_hires, pc1_orig, pc2_orig, pc1_hires, pc2_hires)
            
            fel_window.show()
            
            # Keep reference to prevent garbage collection
            if not hasattr(self, 'fel_windows'):
                self.fel_windows = []
            self.fel_windows.append(fel_window)
            
        except Exception as e:
            QMessageBox.warning(self, "Warning", f"Could not open FEL plot: {str(e)}")

    def find_and_plot_recent_xvg(self):
        if not self.work_dir:
            return
            
        try:
            xvg_files = []
            for file in os.listdir(self.work_dir):
                if file.endswith('.xvg'):
                    file_path = os.path.join(self.work_dir, file)
                    mtime = os.path.getmtime(file_path)
                    xvg_files.append((file_path, mtime))
                    
            if xvg_files:
                xvg_files.sort(key=lambda x: x[1], reverse=True)
                latest_file = xvg_files[0][0]
                self.last_output_file = latest_file
                
                if "PCA_2d" in latest_file or "2d" in latest_file.lower():
                    self.plotter.plot_scatter(latest_file)
                else:
                    self.plotter.plot_xvg(latest_file)
                    
        except Exception as e:
            print(f"Auto-plot error: {str(e)}")

class GromacsAnalysisGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("GROMACS MD Analysis Tool - Enhanced")
        self.setGeometry(100, 100, 1600, 1000)
        
        # Create main tab widget
        self.tab_widget = QTabWidget()
        self.setCentralWidget(self.tab_widget)
        
        # Analysis tab
        self.analysis_widget = AnalysisWidget()
        self.tab_widget.addTab(self.analysis_widget, "Analyze")
        
        # Multi-plotting tab
        self.multi_plotter = MultiPlotter()
        self.tab_widget.addTab(self.multi_plotter, "Plot Multiple Files")
        
        # FEL plotting tab
        self.fel_plotter = FELPlotter()
        self.tab_widget.addTab(self.fel_plotter, "3D Free Energy Landscape")
        
        self.statusBar().showMessage("Ready - Select 'Analyze' tab to begin MD analysis or 'Plot' tab for custom plotting")

    def closeEvent(self, event):
        # Check if any processes are running
        if (hasattr(self.analysis_widget, 'terminal') and 
            self.analysis_widget.terminal.process and 
            self.analysis_widget.terminal.process.state() != QProcess.NotRunning):
            
            reply = QMessageBox.question(self, 'Close Application',
                                       'A process is still running. Do you want to stop it and exit?',
                                       QMessageBox.Yes | QMessageBox.No,
                                       QMessageBox.No)
            if reply == QMessageBox.Yes:
                self.analysis_widget.terminal.stop_process()
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()

def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    app.setApplicationName("GROMACS Analysis Tool Enhanced")
    app.setApplicationVersion("3.0")
    app.setOrganizationName("MD Analysis")
    
    window = GromacsAnalysisGUI()
    window.show()
    
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()

