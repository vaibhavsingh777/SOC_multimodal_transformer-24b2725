import json
import matplotlib.pyplot as plt

try:
    with open('cnn_val_acc.json', 'r') as f:
        cnn_acc = json.load(f)
    with open('vit_val_acc.json', 'r') as f:
        vit_acc = json.load(f)
except FileNotFoundError:
    print("Error: Could not find JSON files. Please ensure you have run both cnn_baseline.py and vit.py completely first!")
    exit()

plt.figure(figsize=(10, 6))

# The CNN trains for 10 epochs, ViT trains for 30
plt.plot(range(1, len(cnn_acc) + 1), cnn_acc, label='CNN Baseline (10 Epochs)', color='red', marker='o')
plt.plot(range(1, len(vit_acc) + 1), vit_acc, label='ViT (30 Epochs)', color='blue', marker='x')

plt.title('CIFAR-10 Validation Accuracy: Tiny CNN vs. ViT', fontsize=14, fontweight='bold')
plt.xlabel('Training Epochs', fontsize=12)
plt.ylabel('Validation Accuracy (%)', fontsize=12)
plt.legend(fontsize=12)
plt.grid(True, linestyle=':', alpha=0.7)

plt.savefig('comparison_plot.png', dpi=300, bbox_inches='tight')
print("Comparison plot saved successfully as 'comparison_plot.png'!")