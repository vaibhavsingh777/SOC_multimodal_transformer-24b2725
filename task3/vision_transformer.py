import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import json

# --- 1. HYPERPARAMETERS & DEVICE ---
img_size = 32
patch_size = 4
in_chans = 3
num_classes = 10
n_embd = 192
n_head = 6
n_layer = 6
dropout = 0.1
batch_size = 128
learning_rate = 3e-4
epochs = 30
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- 2. DATA PREPARATION ---
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

# --- 3. ViT ARCHITECTURE (No Causal Masks!) ---
class HeadBidirectional(nn.Module):
    def __init__(self, n_embd, head_size, dropout):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)
        wei = q @ k.transpose(-2, -1) * (C ** -0.5)
        # CRITICAL CHANGE: No causal mask (tril) here! Every patch sees every patch.
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        return wei @ v

class MultiHeadAttentionBidirectional(nn.Module):
    def __init__(self, n_embd, n_head, head_size, dropout):
        super().__init__()
        self.heads = nn.ModuleList([HeadBidirectional(n_embd, head_size, dropout) for _ in range(n_head)])
        self.proj = nn.Linear(n_head * head_size, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))

class FeedForward(nn.Module):
    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(), # ViTs usually use GELU
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class BlockBidirectional(nn.Module):
    def __init__(self, n_embd, n_head, dropout):
        super().__init__()
        head_size = n_embd // n_head
        self.attn = MultiHeadAttentionBidirectional(n_embd, n_head, head_size, dropout)
        self.ffn = FeedForward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x

class ViT(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_chans=3, num_classes=10, 
                 n_embd=192, n_head=6, n_layer=6, dropout=0.1):
        super().__init__()
        num_patches = (img_size // patch_size) ** 2
        
        # Patch embedding (conv trick)
        self.patch_embed = nn.Conv2d(in_chans, n_embd, kernel_size=patch_size, stride=patch_size)
        
        # CLS token and position embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, n_embd))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, n_embd))
        self.dropout = nn.Dropout(dropout)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([BlockBidirectional(n_embd, n_head, dropout) for _ in range(n_layer)])
        self.norm = nn.LayerNorm(n_embd)
        
        # Classification head
        self.head = nn.Linear(n_embd, num_classes)
        
        # Initialization
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        B = x.size(0)
        
        # Patchify
        x = self.patch_embed(x)                  # (B, C, H/p, W/p)
        x = x.flatten(2).transpose(1, 2)         # (B, num_patches, C)
        
        # Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)           # (B, 1+num_patches, C)
        
        # Add position embedding
        x = x + self.pos_embed
        x = self.dropout(x)
        
        # Pass through Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        
        # Classify from CLS token
        cls_final = x[:, 0]                      # (B, C)
        return self.head(cls_final)

# --- 4. TRAINING LOOP ---
model = ViT().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

train_acc_history, val_acc_history = [], []

print("Starting ViT Training...")
for epoch in range(epochs):
    model.train()
    correct_train, total_train = 0, 0
    for inputs, labels in trainloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        _, predicted = outputs.max(1)
        total_train += labels.size(0)
        correct_train += predicted.eq(labels).sum().item()
        
    train_acc = 100. * correct_train / total_train
    train_acc_history.append(train_acc)
    
    # Validation
    model.eval()
    correct_val, total_val = 0, 0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total_val += labels.size(0)
            correct_val += predicted.eq(labels).sum().item()
            
    val_acc = 100. * correct_val / total_val
    val_acc_history.append(val_acc)
    print(f"Epoch [{epoch+1}/{epochs}] | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

# --- 5. SAVE RESULTS & PLOT ---
# Save accuracies for the comparison plot
with open('vit_val_acc.json', 'w') as f:
    json.dump(val_acc_history, f)

plt.figure(figsize=(8, 5))
plt.plot(range(1, epochs+1), train_acc_history, label='Train Accuracy')
plt.plot(range(1, epochs+1), val_acc_history, label='Validation Accuracy')
plt.title('Vision Transformer (ViT) on CIFAR-10')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.grid(True)
plt.savefig('vit_curves.png')
print("Saved ViT plots to 'vit_curves.png' and metrics to 'vit_val_acc.json'")