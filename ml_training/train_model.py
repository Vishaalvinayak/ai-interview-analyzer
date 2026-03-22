import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import json
import pandas as pd
import os

print("=" * 60)
print("  AI INTERVIEW ANSWER ANALYZER - MODEL TRAINING")
print("=" * 60)

# ============================================
# STEP 1: LOAD DATASET
# ============================================
print("\n📂 [STEP 1/9] Loading dataset...")
dataset_path = '../dataset/dataset.json'

if not os.path.exists(dataset_path):
    print(f"❌ ERROR: {dataset_path} not found!")
    print("Please create your dataset.json file first.")
    exit()

with open(dataset_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f"✅ Loaded {len(data)} samples")

if len(data) < 50:
    print("⚠️  WARNING: Dataset is very small. Minimum 200 samples recommended.")
    print("Current samples:", len(data))

# ============================================
# STEP 2: PREPARE DATA
# ============================================
print("\n🔧 [STEP 2/9] Preparing data...")
df = pd.DataFrame(data)

# Check required columns
required_cols = ['question', 'answer', 'label', 'score']
missing_cols = [col for col in required_cols if col not in df.columns]
if missing_cols:
    print(f"❌ ERROR: Missing columns: {missing_cols}")
    exit()

# Check label distribution
print("\n📊 Label distribution:")
label_counts = df['label'].value_counts()
print(label_counts)

# Check if labels are balanced
min_count = label_counts.min()
max_count = label_counts.max()
if max_count > min_count * 2:
    print("⚠️  WARNING: Imbalanced dataset detected")

# Map labels to numbers
label_map = {'weak': 0, 'average': 1, 'strong': 2}
df['label_id'] = df['label'].map(label_map)

# Check for invalid labels
if df['label_id'].isna().any():
    print("❌ ERROR: Found invalid labels. Use only: weak, average, strong")
    print("Invalid entries:")
    print(df[df['label_id'].isna()][['id', 'label']])
    exit()

# Split data
train_texts, test_texts, train_labels, test_labels = train_test_split(
    df['answer'].tolist(), 
    df['label_id'].tolist(), 
    test_size=0.2, 
    random_state=42,
    stratify=df['label_id']
)

print(f"\n✅ Training samples: {len(train_texts)}")
print(f"✅ Test samples: {len(test_texts)}")

# ============================================
# STEP 3: CREATE DATASET CLASS
# ============================================
print("\n🏗️  [STEP 3/9] Creating dataset class...")

class InterviewDataset(Dataset):
    """Custom Dataset for interview answers"""
    
    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

print("✅ Dataset class ready")

# ============================================
# STEP 4: LOAD MODEL
# ============================================
print("\n🤖 [STEP 4/9] Loading DistilBERT model...")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"💻 Using device: {device}")

if device.type == 'cuda':
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
else:
    print("   Running on CPU (training will be slower)")

print("\n⏳ Downloading DistilBERT (this may take 1-2 minutes)...")

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertForSequenceClassification.from_pretrained(
    'distilbert-base-uncased', 
    num_labels=3
).to(device)

print("✅ Model loaded successfully")
print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# ============================================
# STEP 5: CREATE DATALOADERS
# ============================================
print("\n📦 [STEP 5/9] Creating data loaders...")

BATCH_SIZE = 8  # Small batch size for limited RAM

train_dataset = InterviewDataset(train_texts, train_labels, tokenizer)
test_dataset = InterviewDataset(test_texts, test_labels, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

print(f"✅ Train batches: {len(train_loader)}")
print(f"✅ Test batches: {len(test_loader)}")

# ============================================
# STEP 6: TRAINING SETUP
# ============================================
print("\n⚙️  [STEP 6/9] Setting up training...")

optimizer = AdamW(model.parameters(), lr=2e-5)
EPOCHS = 3

print(f"✅ Optimizer: AdamW (lr=2e-5)")
print(f"✅ Epochs: {EPOCHS}")
print(f"✅ Batch size: {BATCH_SIZE}")

# ============================================
# STEP 7: TRAINING LOOP
# ============================================
print("\n🚀 [STEP 7/9] Starting training...")
print("=" * 60)

training_stats = []

for epoch in range(EPOCHS):
    print(f"\n📍 EPOCH {epoch + 1}/{EPOCHS}")
    print("-" * 60)
    
    # Training phase
    model.train()
    total_loss = 0
    batch_count = 0
    
    for batch_idx, batch in enumerate(train_loader):
        optimizer.zero_grad()
        
        # Move to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        # Forward pass
        outputs = model(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            labels=labels
        )
        
        loss = outputs.loss
        total_loss += loss.item()
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        batch_count += 1
        
        # Print progress every 5 batches
        if (batch_idx + 1) % 5 == 0:
            print(f"   Batch {batch_idx + 1}/{len(train_loader)} | Loss: {loss.item():.4f}")
    
    avg_train_loss = total_loss / len(train_loader)
    
    # Validation phase
    model.eval()
    val_loss = 0
    val_preds = []
    val_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids, 
                attention_mask=attention_mask, 
                labels=labels
            )
            
            val_loss += outputs.loss.item()
            preds = torch.argmax(outputs.logits, dim=1)
            
            val_preds.extend(preds.cpu().numpy())
            val_labels.extend(labels.cpu().numpy())
    
    avg_val_loss = val_loss / len(test_loader)
    val_accuracy = accuracy_score(val_labels, val_preds)
    
    # Print epoch summary
    print(f"\n   📊 Epoch {epoch + 1} Summary:")
    print(f"      Train Loss: {avg_train_loss:.4f}")
    print(f"      Val Loss:   {avg_val_loss:.4f}")
    print(f"      Val Accuracy: {val_accuracy * 100:.2f}%")
    
    training_stats.append({
        'epoch': epoch + 1,
        'train_loss': avg_train_loss,
        'val_loss': avg_val_loss,
        'val_accuracy': val_accuracy
    })

print("\n" + "=" * 60)

# ============================================
# STEP 8: FINAL EVALUATION
# ============================================
print("\n📊 [STEP 8/9] Final evaluation on test set...")

model.eval()
final_predictions = []
final_true_labels = []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels']
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        preds = torch.argmax(outputs.logits, dim=1)
        
        final_predictions.extend(preds.cpu().numpy())
        final_true_labels.extend(labels.numpy())

# Calculate metrics
final_accuracy = accuracy_score(final_true_labels, final_predictions)

print("\n" + "=" * 60)
print("  FINAL RESULTS")
print("=" * 60)
print(f"\n🎯 Test Accuracy: {final_accuracy * 100:.2f}%")

print("\n📈 Classification Report:")
print("-" * 60)
report = classification_report(
    final_true_labels, 
    final_predictions, 
    target_names=['Weak', 'Average', 'Strong'],
    digits=4
)
print(report)

print("\n🔢 Confusion Matrix:")
print("-" * 60)
cm = confusion_matrix(final_true_labels, final_predictions)
print("              Predicted")
print("           Weak  Avg  Strong")
print(f"Weak       {cm[0][0]:4d}  {cm[0][1]:3d}  {cm[0][2]:6d}")
print(f"Average    {cm[1][0]:4d}  {cm[1][1]:3d}  {cm[1][2]:6d}")
print(f"Strong     {cm[2][0]:4d}  {cm[2][1]:3d}  {cm[2][2]:6d}")

# ============================================
# STEP 9: SAVE MODEL
# ============================================
print("\n💾 [STEP 9/9] Saving model...")

save_dir = '../ml_service/model_saved'
os.makedirs(save_dir, exist_ok=True)

model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)

# Save metadata
metadata = {
    'accuracy': float(final_accuracy),
    'num_labels': 3,
    'labels': ['weak', 'average', 'strong'],
    'training_samples': len(train_texts),
    'test_samples': len(test_texts),
    'epochs': EPOCHS,
    'batch_size': BATCH_SIZE,
    'learning_rate': 2e-5
}

with open(f'{save_dir}/metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"✅ Model saved to: {save_dir}")
print(f"✅ Files created:")
print(f"   - config.json")
print(f"   - pytorch_model.bin")
print(f"   - vocab.txt")
print(f"   - tokenizer_config.json")
print(f"   - metadata.json")

print("\n" + "=" * 60)
print("  🎉 TRAINING COMPLETE!")
print("=" * 60)
print("\n📝 Next steps:")
print("1. Check the results above")
print("2. If accuracy < 60%, add more data to dataset.json")
print("3. If accuracy > 65%, proceed to build the API")
print("\n✨ Your model is ready to use!\n")