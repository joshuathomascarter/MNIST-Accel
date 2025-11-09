#!/usr/bin/env python3
"""
Create animated systolic array dataflow visualization for ACCEL-v1 video.
Generates frame-by-frame PNG sequence showing data movement through 2x2 PE array.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os

# Configuration
OUTPUT_DIR = 'output/animation_frames'
FPS = 2  # Frames per second for final video
NUM_CYCLES = 8  # Show 8 computation cycles

# Colors
PE_COLOR = '#2E86AB'
DATA_COLOR = '#A23B72'
RESULT_COLOR = '#F18F01'
ARROW_COLOR = '#C73E1D'

def create_frame(cycle, activations, weights, results, output_path):
    """Create a single frame of the systolic array animation."""
    
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_xlim(-1, 5)
    ax.set_ylim(-1, 5)
    ax.axis('off')
    
    # Title
    ax.text(2, 4.5, f'Systolic Array Cycle {cycle}', 
            ha='center', va='center', fontsize=20, fontweight='bold')
    
    # Draw 2x2 PE array
    # Correct indexing: PE(row, col) where top row is row 0
    pe_positions = [(1, 3), (3, 3), (1, 1), (3, 1)]
    pe_labels = ['PE(0,0)', 'PE(0,1)', 'PE(1,0)', 'PE(1,1)']
    
    for (x, y), label in zip(pe_positions, pe_labels):
        # PE square
        rect = patches.Rectangle((x-0.4, y-0.4), 0.8, 0.8,
                                 linewidth=2, edgecolor=PE_COLOR,
                                 facecolor=PE_COLOR, alpha=0.3)
        ax.add_patch(rect)
        ax.text(x, y, label, ha='center', va='center',
               fontsize=10, fontweight='bold')
    
    # Draw activation inputs (flowing left to right)
    for i, (x, y) in enumerate([(1, 3), (1, 1)]):  # Row 0 and Row 1
        if cycle < len(activations[i]):
            val = activations[i][cycle]
            # Activation value box (left side)
            rect = patches.FancyBboxPatch((x-1.8, y-0.2), 0.8, 0.4,
                                         boxstyle="round,pad=0.05",
                                         edgecolor=DATA_COLOR,
                                         facecolor=DATA_COLOR, alpha=0.5)
            ax.add_patch(rect)
            ax.text(x-1.4, y, f'A[{i}]={val}', ha='center', va='center',
                   fontsize=9, fontweight='bold', color='white')
            
            # Arrow showing movement into PE
            ax.annotate('', xy=(x-0.5, y), xytext=(x-0.9, y),
                       arrowprops=dict(arrowstyle='->', lw=2, color=ARROW_COLOR))
            
            # If data propagates right, show it flowing to next PE
            if cycle > 0 and x == 1:  # PE(0,0) and PE(1,0) pass right
                rect_pass = patches.FancyBboxPatch((x+0.6, y-0.2), 0.8, 0.4,
                                                   boxstyle="round,pad=0.05",
                                                   edgecolor=DATA_COLOR,
                                                   facecolor=DATA_COLOR, alpha=0.3)
                ax.add_patch(rect_pass)
                prev_val = activations[i][cycle-1]
                ax.text(x+1.0, y, f'{prev_val}', ha='center', va='center',
                       fontsize=9, color='gray')
                ax.annotate('', xy=(x+1.5, y), xytext=(x+0.5, y),
                           arrowprops=dict(arrowstyle='->', lw=1.5, color=DATA_COLOR, alpha=0.5))
    
    # Draw weight inputs (flowing top to bottom)
    for j, (x, y) in enumerate([(1, 3), (3, 3)]):  # Col 0 and Col 1 (top row)
        if cycle < len(weights[j]):
            val = weights[j][cycle]
            # Weight value box (top)
            rect = patches.FancyBboxPatch((x-0.2, y+0.6), 0.4, 0.8,
                                         boxstyle="round,pad=0.05",
                                         edgecolor=DATA_COLOR,
                                         facecolor=DATA_COLOR, alpha=0.5)
            ax.add_patch(rect)
            ax.text(x, y+1.0, f'B[{j}]={val}', ha='center', va='center',
                   fontsize=9, fontweight='bold', color='white')
            
            # Arrow showing movement into PE
            ax.annotate('', xy=(x, y+0.5), xytext=(x, y+0.9),
                       arrowprops=dict(arrowstyle='->', lw=2, color=ARROW_COLOR))
            
            # If data propagates down, show it flowing to next PE
            if cycle > 0:  # PE(0,j) passes down to PE(1,j)
                rect_pass = patches.FancyBboxPatch((x-0.2, y-1.4), 0.4, 0.8,
                                                   boxstyle="round,pad=0.05",
                                                   edgecolor=DATA_COLOR,
                                                   facecolor=DATA_COLOR, alpha=0.3)
                ax.add_patch(rect_pass)
                prev_val = weights[j][cycle-1]
                ax.text(x, y-1.0, f'{prev_val}', ha='center', va='center',
                       fontsize=9, color='gray')
                ax.annotate('', xy=(x, y-0.5), xytext=(x, y+0.5),
                           arrowprops=dict(arrowstyle='->', lw=1.5, color=DATA_COLOR, alpha=0.5))
    
    # Draw partial sums accumulating at bottom of each column (C[0] and C[1])
    for j, (x, y) in enumerate([(1, 1), (3, 1)]):  # Bottom PEs: PE(1,0) and PE(1,1)
        if cycle > 0:
            # Accumulated result flowing down from bottom PE
            result_idx = min(cycle-1, len(results[j])-1)
            val = results[j][result_idx]
            
            # Result box below bottom PE
            rect = patches.FancyBboxPatch((x-0.2, y-1.0), 0.4, 0.6,
                                         boxstyle="round,pad=0.05",
                                         edgecolor=RESULT_COLOR,
                                         facecolor=RESULT_COLOR, alpha=0.5)
            ax.add_patch(rect)
            ax.text(x, y-0.7, f'C[{j}]={val}', ha='center', va='center',
                   fontsize=9, fontweight='bold', color='white')
            
            # Arrow showing accumulation flowing down
            ax.annotate('', xy=(x, y-1.3), xytext=(x, y-0.5),
                       arrowprops=dict(arrowstyle='->', lw=2, color=ARROW_COLOR))
    
    # Add legend
    legend_y = -0.5
    ax.text(0, legend_y, 'Data Flow:', fontsize=11, fontweight='bold')
    
    # Activation legend
    rect = patches.Rectangle((1.2, legend_y-0.15), 0.3, 0.3,
                             facecolor=DATA_COLOR, alpha=0.5)
    ax.add_patch(rect)
    ax.text(1.6, legend_y, 'Activations', fontsize=10, va='center')
    
    # Weight legend
    rect = patches.Rectangle((2.5, legend_y-0.15), 0.3, 0.3,
                             facecolor=DATA_COLOR, alpha=0.5)
    ax.add_patch(rect)
    ax.text(2.9, legend_y, 'Weights', fontsize=10, va='center')
    
    # Result legend
    rect = patches.Rectangle((3.7, legend_y-0.15), 0.3, 0.3,
                             facecolor=RESULT_COLOR, alpha=0.5)
    ax.add_patch(rect)
    ax.text(4.1, legend_y, 'Results', fontsize=10, va='center')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

def main():
    """Generate all animation frames."""
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"\nGenerating systolic array animation...")
    print("=" * 60)
    
    # Example: 2x2 matrix multiply C = A @ B
    # A = [[1,2],[3,4]]  B = [[5,6],[7,8]]
    # C[0] = A[0,:]·B[:,0] = 1*5 + 2*7 = 19
    # C[1] = A[0,:]·B[:,1] = 1*6 + 2*8 = 22
    # C[2] = A[1,:]·B[:,0] = 3*5 + 4*7 = 43
    # C[3] = A[1,:]·B[:,1] = 3*6 + 4*8 = 50
    
    # Activation inputs (rows flow left to right)
    activations = [
        [1, 2, 0, 0],  # A[0]: row 0 flows right
        [3, 4, 0, 0],  # A[1]: row 1 flows right
    ]
    
    # Weight inputs (columns flow top to bottom)
    weights = [
        [5, 7, 0, 0],  # B[0]: col 0 flows down
        [6, 8, 0, 0],  # B[1]: col 1 flows down
    ]
    
    # Accumulated results at bottom of each column
    # C[0] accumulates in column 0, C[1] accumulates in column 1
    results = [
        [5, 19, 19, 19],   # C[0]: 1*5=5, then 1*5+2*7=19
        [6, 22, 22, 22],   # C[1]: 1*6=6, then 1*6+2*8=22
    ]
    
    # Generate frames
    for cycle in range(NUM_CYCLES):
        output_path = os.path.join(OUTPUT_DIR, f'frame_{cycle:03d}.png')
        create_frame(cycle, activations, weights, results, output_path)
        print(f"✓ Generated: frame_{cycle:03d}.png")
    
    print("=" * 60)
    print(f"\n{NUM_CYCLES} frames saved to: {OUTPUT_DIR}/")
    print(f"\nCorrect dataflow:")
    print(f"  - Activations flow LEFT → RIGHT (horizontal)")
    print(f"  - Weights flow TOP → BOTTOM (vertical)")
    print(f"  - Results accumulate at BOTTOM of each column")
    print(f"  - C[0] = column 0 output, C[1] = column 1 output")
    print(f"\nTo create video from frames:")
    print(f"  ffmpeg -framerate {FPS} -i {OUTPUT_DIR}/frame_%03d.png \\")
    print(f"         -c:v libx264 -pix_fmt yuv420p \\")
    print(f"         output/systolic_array_animation.mp4")
    print()

if __name__ == '__main__':
    main()
