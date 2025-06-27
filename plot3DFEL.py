#!/usr/bin/env python3
"""
3D Free Energy Landscape Plotter for GROMACS XPM files with 2D side-by-side view
Usage: python3 plot3dFEL.py PCA_sham.xpm
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage import gaussian_filter

def unquote(s):
    """Remove quotes from XPM strings"""
    return s[s.find('"')+1:s.rfind('"')]

def parse_xpm(xpm_file):
    """Parse GROMACS XPM file and return energy matrix"""
    print(f"Reading XPM file: {xpm_file}")
    
    with open(xpm_file) as f:
        lines = f.readlines()

    # Find matrix dimensions and color legend
    for i, line in enumerate(lines):
        if line.strip().startswith('"') and line.count('"') >= 2:
            header = unquote(line)
            break

    nx, ny, ncolors, nchar = [int(x) for x in header.split()]
    print(f"Matrix dimensions: {nx} x {ny}, Colors: {ncolors}")
    
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
            matrix_lines.append(unquote(line).replace(',', ''))

    matrix = np.zeros((ny, nx))
    for y, row in enumerate(matrix_lines):
        for x in range(nx):
            symbol = row[x*nchar:(x+1)*nchar]
            matrix[y, x] = color_map[symbol]
    
    print(f"Energy range: {np.nanmin(matrix):.2f} to {np.nanmax(matrix):.2f} kJ/mol")
    return matrix, nx, ny

def create_smooth_surface(matrix, nx, ny):
    """Create smooth high-resolution surface"""
    print("Creating smooth surface...")
    
    # Original coordinate grids
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

    # REDUCED smoothing to preserve more structure and discontinuities
    matrix_smooth = gaussian_filter(matrix, sigma=0.5)  # Reduced from 1.0 to 0.5

    # High resolution for smooth surface
    pc1_hires = np.linspace(-3, 3, nx*2)  # Reduced from nx*3 to nx*2
    pc2_hires = np.linspace(-3, 3, ny*2)  # Reduced from ny*3 to ny*2

    # Use more conservative interpolation to preserve discontinuities
    spline = RectBivariateSpline(pc2_orig, pc1_orig, matrix_smooth, kx=2, ky=2)  # Reduced from kx=3, ky=3
    matrix_hires = spline(pc2_hires, pc1_hires)

    return matrix_smooth, matrix_hires, pc1_orig, pc2_orig, pc1_hires, pc2_hires

def plot_2d_3d_fel(matrix_smooth, matrix_hires, pc1_orig, pc2_orig, pc1_hires, pc2_hires, output_name=None):
    """Create side-by-side 2D and 3D FEL plots"""
    print("Creating 2D and 3D plots...")
    
    # Create meshgrids
    X_hires, Y_hires = np.meshgrid(pc1_hires, pc2_hires)
    X_orig, Y_orig = np.meshgrid(pc1_orig, pc2_orig)

    # Create figure with two subplots side by side
    fig = plt.figure(figsize=(20, 8))
    
    # 2D plot on the left
    ax1 = fig.add_subplot(121)
    
    # FIXED: Flip the matrix vertically to match reference orientation
    matrix_flipped = np.flipud(matrix_smooth)
    
    # Create 2D contour plot with correct orientation
    im = ax1.imshow(matrix_flipped, extent=[-3, 3, -3, 3], 
                    origin='lower', cmap='jet', aspect='equal')
    
    # Labels and styling for 2D plot
    ax1.set_xlabel('PC1', fontsize=14)
    ax1.set_ylabel('PC2', fontsize=14)
    #ax1.set_title('2D Free Energy Landscape', fontsize=16, pad=20)
    ax1.grid(True, alpha=0.3)
    
    # FIXED: Colorbar at the bottom with min/max labels
    cbar1 = plt.colorbar(im, ax=ax1, orientation='horizontal', 
                        shrink=0.8, aspect=30, pad=0.15)
    cbar1.set_label('Gibbs Free Energy (kJ/mol)', fontsize=12)
    
    # Add min/max labels on colorbar
    vmin, vmax = np.nanmin(matrix_smooth), np.nanmax(matrix_smooth)
    cbar1.set_ticks([vmin, vmax])
    cbar1.set_ticklabels([f'{vmin:.1f}', f'{vmax:.1f}'])

    # 3D plot on the right
    ax2 = fig.add_subplot(122, projection='3d')

    # 1. First plot the 2D heatmap as the base/shadow at z=0
    # Use the original resolution for the base
    matrix_flipped_3d = np.flipud(matrix_smooth)
    ax2.contourf(X_orig, Y_orig, matrix_flipped_3d, 
                zdir='z', offset=0, cmap='jet', alpha=0.8, levels=50)

    # 2. Plot the 3D surface rising from the base
    surf = ax2.plot_surface(X_hires, Y_hires, matrix_hires,
                           cmap='jet',
                           alpha=0.85,
                           linewidth=0.1,
                           antialiased=True,
                           shade=True,
                           rstride=2,
                           cstride=2,
                           edgecolors='none')

    # Set the viewing angle to match reference
    ax2.view_init(elev=30, azim=45)

    # Labels and limits for 3D plot
    ax2.set_xlabel('PC1', fontsize=14, labelpad=10)
    ax2.set_ylabel('PC2', fontsize=14, labelpad=10)
    ax2.set_zlabel('Gibbs Free Energy (kJ/mol)', fontsize=12, labelpad=10)

    ax2.set_xlim(-3, 3)
    ax2.set_ylim(-3, 3)
    ax2.set_zlim(0, np.max(matrix_hires) * 1.1)

    # Enhanced styling for 3D plot to match reference
    ax2.xaxis.pane.fill = True
    ax2.yaxis.pane.fill = True  
    ax2.zaxis.pane.fill = True
    ax2.xaxis.pane.set_facecolor('#f0f0f0')
    ax2.yaxis.pane.set_facecolor('#f0f0f0')
    ax2.zaxis.pane.set_facecolor('#f0f0f0')
    ax2.xaxis.pane.set_alpha(0.3)
    ax2.yaxis.pane.set_alpha(0.3)
    ax2.zaxis.pane.set_alpha(0.3)
    
    # Grid styling
    ax2.grid(True, alpha=0.3, linewidth=0.5, color='gray')
    
    # Make the axes lines more prominent
    ax2.xaxis._axinfo['tick']['inward_factor'] = 0
    ax2.yaxis._axinfo['tick']['inward_factor'] = 0
    ax2.zaxis._axinfo['tick']['inward_factor'] = 0

    # Title for 3D plot
    #ax2.set_title('3D Free Energy Landscape', fontsize=16, pad=20)

    # Colorbar for 3D plot
    cbar2 = plt.colorbar(surf, ax=ax2, shrink=0.6, aspect=20, pad=0.1)
    cbar2.set_label('Gibbs Free Energy (kJ/mol)', fontsize=12)

    # Add main title
    #fig.suptitle('Free Energy Landscape Analysis', fontsize=18, y=0.95)

    plt.tight_layout()
    
    # Save or show
    if output_name:
        plt.savefig(output_name, dpi=300, bbox_inches='tight')
        print(f"Plot saved as: {output_name}")
    else:
        plt.show()

def main():
    """Main function"""
    if len(sys.argv) < 2:
        print("Usage: python3 plot3dFEL.py <XPM_file> [output_image]")
        print("Example: python3 plot3dFEL.py PCA_sham.xpm")
        print("         python3 plot3dFEL.py PCA_sham.xpm 2d_3d_landscape.png")
        sys.exit(1)
    
    xpm_file = sys.argv[1]
    output_name = sys.argv[2] if len(sys.argv) > 2 else None
    
    try:
        # Parse XPM file
        matrix, nx, ny = parse_xpm(xpm_file)
        
        # Create smooth surface
        matrix_smooth, matrix_hires, pc1_orig, pc2_orig, pc1_hires, pc2_hires = create_smooth_surface(matrix, nx, ny)
        
        # Create 2D and 3D plots side by side
        plot_2d_3d_fel(matrix_smooth, matrix_hires, pc1_orig, pc2_orig, pc1_hires, pc2_hires, output_name)
        
        print("Done!")
        
    except FileNotFoundError:
        print(f"Error: File '{xpm_file}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
