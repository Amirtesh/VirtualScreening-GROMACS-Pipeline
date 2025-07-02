#!/usr/bin/env python3
"""
PNG DPI Converter Script
Converts all PNG images in the current directory to a specified DPI.

Usage: python3 convertDPI.py <dpi_value>
Example: python3 convertDPI.py 600
"""

import sys
import os
import glob
from PIL import Image

def convert_png_dpi(dpi):
    """
    Convert all PNG files in current directory to specified DPI.
    
    Args:
        dpi (int): Target DPI value
    """
    # Find all PNG files in current directory
    png_files = glob.glob("*.png")
    
    if not png_files:
        print("No PNG files found in current directory.")
        return
    
    print(f"Found {len(png_files)} PNG file(s)")
    print(f"Converting to {dpi} DPI...")
    
    converted_count = 0
    
    for filename in png_files:
        try:
            # Open the image
            with Image.open(filename) as img:
                # Check current DPI
                current_dpi = img.info.get('dpi', (72, 72))
                print(f"Processing {filename} (current DPI: {current_dpi[0]}x{current_dpi[1]})")
                
                # Save with new DPI
                img.save(filename, dpi=(dpi, dpi))
                converted_count += 1
                print(f"  ✓ Converted to {dpi}x{dpi} DPI")
                
        except Exception as e:
            print(f"  ✗ Error processing {filename}: {str(e)}")
    
    print(f"\nConversion complete! {converted_count}/{len(png_files)} files processed successfully.")

def main():
    # Check command line arguments
    if len(sys.argv) != 2:
        print("Usage: python3 convertDPI.py <dpi_value>")
        print("Example: python3 convertDPI.py 600")
        sys.exit(1)
    
    try:
        dpi = int(sys.argv[1])
        if dpi <= 0:
            raise ValueError("DPI must be a positive number")
    except ValueError as e:
        print(f"Error: Invalid DPI value '{sys.argv[1]}'. Please provide a positive integer.")
        sys.exit(1)
    
    # Check if Pillow is available
    try:
        from PIL import Image
    except ImportError:
        print("Error: PIL (Pillow) library is required.")
        print("Install it using: pip install Pillow")
        sys.exit(1)
    
    # Convert images
    convert_png_dpi(dpi)

if __name__ == "__main__":
    main()
