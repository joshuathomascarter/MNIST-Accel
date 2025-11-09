#!/usr/bin/env python3
"""
Generate all visual assets for ACCEL-v1 video
Creates charts, graphs, and comparison plots
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import os

# Create output directory
os.makedirs('output', exist_ok=True)

def create_accuracy_comparison():
    """Create FP32 vs INT8 accuracy comparison chart"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    categories = ['FP32\nBaseline', 'INT8\nQuantized']
    accuracies = [98.9, 98.7]
    colors = ['#2ecc71', '#3498db']
    
    bars = ax.bar(categories, accuracies, color=colors, width=0.6, edgecolor='black', linewidth=2)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc}%',
                ha='center', va='bottom', fontsize=16, fontweight='bold')
    
    ax.set_ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
    ax.set_title('MNIST CNN: FP32 vs INT8 Quantization', fontsize=16, fontweight='bold')
    ax.set_ylim(95, 100)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add accuracy loss annotation
    ax.annotate('', xy=(1, 98.7), xytext=(0, 98.9),
                arrowprops=dict(arrowstyle='<->', color='red', lw=2))
    ax.text(0.5, 98.8, '0.2% loss', ha='center', fontsize=12, color='red', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('output/accuracy_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Created: accuracy_comparison.png")
    plt.close()


def create_architecture_metrics():
    """Create key metrics visualization"""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('off')
    
    metrics = [
        ('Systolic Array', '2×2 PEs', '#3498db'),
        ('Data Type', 'INT8×INT8→INT32', '#e74c3c'),
        ('UART Protocol', '7-byte packets', '#2ecc71'),
        ('Buffers', 'Dual-bank ping-pong', '#f39c12'),
        ('Quantization Loss', '0.2% (MNIST)', '#9b59b6'),
        ('Clock Rate', '50 MHz (simulation)', '#1abc9c'),
    ]
    
    y_pos = 0.85
    for i, (label, value, color) in enumerate(metrics):
        # Background box
        rect = patches.Rectangle((0.1, y_pos - 0.08), 0.8, 0.12, 
                                 linewidth=2, edgecolor=color, 
                                 facecolor=color, alpha=0.2)
        ax.add_patch(rect)
        
        # Text
        ax.text(0.15, y_pos, label, fontsize=16, fontweight='bold', va='center')
        ax.text(0.85, y_pos, value, fontsize=14, ha='right', va='center', 
                fontweight='bold', color=color)
        
        y_pos -= 0.15
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title('ACCEL-v1 Key Specifications', fontsize=20, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig('output/key_metrics.png', dpi=300, bbox_inches='tight')
    print("✓ Created: key_metrics.png")
    plt.close()


def create_systolic_array_diagram():
    """Create static systolic array diagram showing dataflow"""
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Draw 2x2 PE grid
    pe_size = 1.5
    start_x, start_y = 2, 2
    
    for i in range(2):
        for j in range(2):
            x = start_x + j * 2.5
            y = start_y + i * 2.5
            
            # PE box
            rect = patches.Rectangle((x, y), pe_size, pe_size, 
                                     linewidth=3, edgecolor='#2c3e50', 
                                     facecolor='#3498db', alpha=0.3)
            ax.add_patch(rect)
            
            # PE label
            ax.text(x + pe_size/2, y + pe_size/2, f'PE\n({i},{j})', 
                   ha='center', va='center', fontsize=14, fontweight='bold')
            
            # Horizontal arrows (activations)
            if j < 1:
                ax.arrow(x + pe_size, y + pe_size/2, 0.8, 0,
                        head_width=0.2, head_length=0.15, fc='#e74c3c', ec='#e74c3c', lw=2)
                ax.text(x + pe_size + 0.4, y + pe_size/2 + 0.3, 'a', 
                       fontsize=12, color='#e74c3c', fontweight='bold')
            
            # Vertical arrows (weights)
            if i < 1:
                ax.arrow(x + pe_size/2, y + pe_size, 0, 0.8,
                        head_width=0.2, head_length=0.15, fc='#2ecc71', ec='#2ecc71', lw=2)
                ax.text(x + pe_size/2 + 0.3, y + pe_size + 0.4, 'b', 
                       fontsize=12, color='#2ecc71', fontweight='bold')
    
    # Input labels
    ax.text(start_x - 0.8, start_y + pe_size/2, 'A[0]', fontsize=12, fontweight='bold', color='#e74c3c')
    ax.text(start_x - 0.8, start_y + 2.5 + pe_size/2, 'A[1]', fontsize=12, fontweight='bold', color='#e74c3c')
    ax.text(start_x + pe_size/2, start_y - 0.8, 'B[0]', fontsize=12, fontweight='bold', color='#2ecc71')
    ax.text(start_x + 2.5 + pe_size/2, start_y - 0.8, 'B[1]', fontsize=12, fontweight='bold', color='#2ecc71')
    
    # Output arrows
    for j in range(2):
        x = start_x + j * 2.5
        y = start_y + 2 * 2.5
        ax.arrow(x + pe_size/2, y + pe_size, 0, 0.5,
                head_width=0.2, head_length=0.15, fc='#9b59b6', ec='#9b59b6', lw=2)
        ax.text(x + pe_size/2 + 0.3, y + pe_size + 0.6, f'C[{j}]', 
               fontsize=12, color='#9b59b6', fontweight='bold')
    
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 9)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('2×2 Systolic Array Dataflow', fontsize=18, fontweight='bold', pad=20)
    
    # Legend
    legend_elements = [
        patches.Patch(color='#e74c3c', label='Activations (horizontal)'),
        patches.Patch(color='#2ecc71', label='Weights (vertical)'),
        patches.Patch(color='#9b59b6', label='Outputs (accumulate)')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=11)
    
    plt.tight_layout()
    plt.savefig('output/systolic_array_static.png', dpi=300, bbox_inches='tight')
    print("✓ Created: systolic_array_static.png")
    plt.close()


def create_quantization_process():
    """Create quantization process visualization"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # FP32 distribution
    fp32_weights = np.random.normal(0, 0.5, 10000)
    ax1.hist(fp32_weights, bins=50, color='#3498db', alpha=0.7, edgecolor='black')
    ax1.set_title('FP32 Weight Distribution', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Weight Value', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.grid(alpha=0.3)
    ax1.axvline(0, color='red', linestyle='--', linewidth=2)
    
    # INT8 distribution
    scale = np.max(np.abs(fp32_weights)) / 127
    int8_weights = np.clip(np.round(fp32_weights / scale), -128, 127)
    ax2.hist(int8_weights, bins=50, color='#e74c3c', alpha=0.7, edgecolor='black')
    ax2.set_title('INT8 Quantized Distribution', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Quantized Value', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.grid(alpha=0.3)
    ax2.axvline(0, color='red', linestyle='--', linewidth=2)
    
    fig.suptitle('Post-Training Quantization: FP32 → INT8', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('output/quantization_process.png', dpi=300, bbox_inches='tight')
    print("✓ Created: quantization_process.png")
    plt.close()


def create_uart_protocol_diagram():
    """Create UART packet structure diagram"""
    fig, ax = plt.subplots(figsize=(14, 6))
    
    packet_fields = [
        ('CMD', 1, '#e74c3c'),
        ('ADDR_L', 1, '#3498db'),
        ('ADDR_H', 1, '#3498db'),
        ('DATA_0', 1, '#2ecc71'),
        ('DATA_1', 1, '#2ecc71'),
        ('DATA_2', 1, '#2ecc71'),
        ('DATA_3', 1, '#2ecc71'),
    ]
    
    x_pos = 0.5
    y_pos = 0.5
    box_width = 1.5
    box_height = 0.8
    
    for i, (label, size, color) in enumerate(packet_fields):
        # Draw box
        rect = patches.Rectangle((x_pos, y_pos), box_width, box_height,
                                 linewidth=3, edgecolor='black',
                                 facecolor=color, alpha=0.3)
        ax.add_patch(rect)
        
        # Label
        ax.text(x_pos + box_width/2, y_pos + box_height/2, label,
               ha='center', va='center', fontsize=14, fontweight='bold')
        
        # Byte number
        ax.text(x_pos + box_width/2, y_pos - 0.15, f'Byte {i}',
               ha='center', fontsize=10, style='italic')
        
        x_pos += box_width + 0.1
    
    # Add bracket showing total
    ax.plot([0.5, x_pos - 0.1], [y_pos + box_height + 0.2, y_pos + box_height + 0.2],
           'k-', linewidth=2)
    ax.text((0.5 + x_pos - 0.1)/2, y_pos + box_height + 0.35, '7 bytes total',
           ha='center', fontsize=12, fontweight='bold')
    
    # Command examples
    cmd_examples = [
        '0x0X: CSR Write',
        '0x2X: Buffer Write (Act)',
        '0x3X: Buffer Write (Wgt)',
        '0x5X: Start Compute',
    ]
    
    y_text = 0.15
    ax.text(0.5, y_text, 'Command Examples:', fontsize=12, fontweight='bold')
    for i, cmd in enumerate(cmd_examples):
        ax.text(1.5, y_text - 0.08*(i+1), f'• {cmd}', fontsize=10)
    
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 2)
    ax.axis('off')
    ax.set_title('UART Packet Protocol (115200 baud)', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('output/uart_protocol.png', dpi=300, bbox_inches='tight')
    print("✓ Created: uart_protocol.png")
    plt.close()


def create_module_breakdown():
    """Create hardware module breakdown pie chart"""
    fig, ax = plt.subplots(figsize=(10, 10))
    
    modules = ['Systolic Array\n(4 PEs)', 'Buffers\n(Act + Wgt)', 'Scheduler\nFSM', 
               'CSR Block', 'UART\n(RX + TX)', 'Glue Logic']
    sizes = [30, 25, 15, 12, 13, 5]
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#95a5a6']
    explode = (0.1, 0.05, 0, 0, 0, 0)
    
    wedges, texts, autotexts = ax.pie(sizes, explode=explode, labels=modules, colors=colors,
                                       autopct='%1.1f%%', startangle=90,
                                       textprops={'fontsize': 12, 'fontweight': 'bold'},
                                       wedgeprops={'edgecolor': 'black', 'linewidth': 2})
    
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(11)
    
    ax.set_title('Hardware Module Composition', fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig('output/module_breakdown.png', dpi=300, bbox_inches='tight')
    print("✓ Created: module_breakdown.png")
    plt.close()


def create_video_timeline():
    """Create video structure timeline"""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    sections = [
        ('Introduction\nHook', 0, 15, '#e74c3c'),
        ('Problem\nStatement', 15, 30, '#3498db'),
        ('Architecture\nOverview', 45, 45, '#2ecc71'),
        ('Quantization', 90, 45, '#f39c12'),
        ('Software\nStack', 135, 45, '#9b59b6'),
        ('Verification', 180, 30, '#1abc9c'),
        ('Technical\nDetails', 210, 30, '#34495e'),
        ('Status &\nLimits', 240, 30, '#e67e22'),
        ('Conclusion', 270, 30, '#16a085'),
    ]
    
    y_pos = 0.8
    for label, start, duration, color in sections:
        # Timeline bar
        rect = patches.Rectangle((start/300, y_pos), duration/300, 0.08,
                                 linewidth=2, edgecolor='black',
                                 facecolor=color, alpha=0.7)
        ax.add_patch(rect)
        
        # Label
        ax.text(start/300 + duration/600, y_pos + 0.04, label,
               ha='center', va='center', fontsize=10, fontweight='bold')
        
        # Time markers
        ax.text(start/300, y_pos - 0.02, f'{start}s',
               ha='center', fontsize=8, style='italic')
    
    # End time
    ax.text(1.0, y_pos - 0.02, '300s (5min)',
           ha='center', fontsize=8, style='italic')
    
    # Timeline axis
    ax.plot([0, 1], [y_pos, y_pos], 'k-', linewidth=2)
    
    # Minute markers
    for minute in range(6):
        x = minute / 5
        ax.plot([x, x], [y_pos - 0.01, y_pos + 0.01], 'k-', linewidth=2)
        ax.text(x, y_pos - 0.05, f'{minute}m', ha='center', fontsize=9, fontweight='bold')
    
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(0.6, 1.0)
    ax.axis('off')
    ax.set_title('Video Script Timeline (5 minutes)', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('output/video_timeline.png', dpi=300, bbox_inches='tight')
    print("✓ Created: video_timeline.png")
    plt.close()


if __name__ == '__main__':
    print("\nGenerating visual assets for ACCEL-v1 video...")
    print("=" * 60)
    
    create_accuracy_comparison()
    create_architecture_metrics()
    create_systolic_array_diagram()
    create_quantization_process()
    create_uart_protocol_diagram()
    create_module_breakdown()
    create_video_timeline()
    
    print("=" * 60)
    print(f"\nAll visuals saved to: video_assets/output/")
    print("Ready for video production!\n")
