import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from collections import Counter
import re
import pickle
import random
import nltk
from nltk.corpus import wordnet
import matplotlib.pyplot as plt
nltk.download('wordnet')

# 超参数设置
BATCH_SIZE = 32
EMBEDDING_DIM = 100
HIDDEN_DIM = 128
EPOCHS = 10
DROPOUT_RATE = 0.3
LEARNING_RATE = 0.002
PATIENCE = 3
MIN_DELTA = 0.001 
MAX_LEN = 64
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 关键优化之一，定义同义词替换函数，用于数据增强
def synonym_replace(text, prob=0.2):
    words = tokenize(text)
    new_words = words.copy()
    for i, word in enumerate(words):
        if random.random() < prob:
            # 获取同义词
            syns = wordnet.synsets(word)
            if syns:
                new_word = syns[0].lemmas()[0].name()
                # 替换
                new_words[i] = new_word
    return ' '.join(new_words)

# 数据增强函数
def augment_data(df):
    augmented_texts = []
    for text in df['text']:
        augmented_texts.append(synonym_replace(text))
    df_aug = df.copy()
    df_aug['text'] = augmented_texts
    return pd.concat([df, df_aug], ignore_index=True)

# 预处理函数
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9!?\s]", "", text)
    return text

# 分词器
def tokenize(text):
    return re.findall(r'\w+|"!?"', text)
    
# 构建词表
def build_vocab(texts, min_freq=2):
    counter = Counter()
    for text in texts:
        counter.update(tokenize(text))
    vocab = {'<PAD>': 0, '<UNK>': 1}
    idx = 2
    for w, c in counter.items():
        if c >= min_freq:
            vocab[w] = idx
            idx += 1
    return vocab

# 编码函数
def encode(text, vocab):
    tokens = tokenize(text)
    ids = [vocab.get(t, vocab['<UNK>']) for t in tokens]
    if len(ids) < MAX_LEN:
        ids += [vocab['<PAD>']] * (MAX_LEN - len(ids))
    else:
        ids = ids[:MAX_LEN]
    return ids


class RumorDataset(Dataset):
    def __init__(self, df, vocab):
        self.texts = df['text'].apply(preprocess_text).tolist()
        self.labels = df['label'].tolist()
        self.vocab = vocab

    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        x = torch.tensor(encode(self.texts[idx], self.vocab), dtype=torch.long).clone().detach()
        y = torch.tensor(self.labels[idx], dtype=torch.float)
        return x, y

# 定义 BiGRU 模型类
class BiGRU(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, dropout_rate):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.dropout1 = nn.Dropout(dropout_rate)
        # 双向GRU层
        self.bigru = nn.GRU(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(hidden_dim*2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        emb = self.embedding(x)
        emb = self.dropout1(emb)
        _, h = self.bigru(emb)
        h = torch.cat([h[0], h[1]], dim=1)
        h = self.dropout2(h)
        h = self.relu(self.fc1(h))
        out = self.fc2(h)
        return out.squeeze(1)

def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            preds = (torch.sigmoid(logits) > 0.5).float()
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / total

def main():
    # 读取数据集
    train_df = pd.read_csv('./train.csv')
    val_df = pd.read_csv('./val.csv')
    
    # 改进的预处理
    train_df['text'] = train_df['text'].apply(preprocess_text)
    val_df['text'] = val_df['text'].apply(preprocess_text)

    train_df = augment_data(train_df)

    # 计算正样本权重
    train_labels = train_df['label'].values
    # 负样本数/正样本数
    pos_weight = torch.tensor([(len(train_labels) - sum(train_labels)) / sum(train_labels)]).to(DEVICE)  
    

    # 构建词表
    vocab = build_vocab(train_df['text'])
    
    with open('vocab.pkl', 'wb') as f:
        pickle.dump(vocab, f)
    
    # 构建数据集
    train_set = RumorDataset(train_df, vocab)
    val_set = RumorDataset(val_df, vocab)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)
    
    # 初始化模型，优化器和损失函数
    model = BiGRU(len(vocab), EMBEDDING_DIM, HIDDEN_DIM, DROPOUT_RATE).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    # 早停机制
    best_val_acc = 0.0
    patience_counter = 0
    
    # 记录准确率
    train_accs = []
    val_accs = []
    
    # 训练模型
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_acc = evaluate(model, train_loader)
        val_acc = evaluate(model, val_loader)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        scheduler.step(val_acc)
        
        print(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}')
        
        # 早停检查
        if val_acc - best_val_acc > MIN_DELTA:
            best_val_acc = val_acc
            patience_counter = 0
            # 保存最佳模型
            torch.save({
                'model_state_dict': model.state_dict(),
                'vocab_size': len(vocab),
                'embedding_dim': EMBEDDING_DIM,
                'hidden_dim': HIDDEN_DIM,
                'dropout_rate': DROPOUT_RATE
            }, 'bigru_model.pt')
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f'Early stopping at epoch {epoch+1}')
                break
    
    print(f'Best Val Acc: {best_val_acc:.4f}')
    print('Model saved as bigru_model.pt')

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(val_accs) + 1), val_accs, label='Validation Accuracy', color='blue', marker='o')
    plt.title('Validation Accuracy per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim(0.75,0.9)
    plt.xticks(range(1, len(train_accs) + 1))
    plt.legend()
    plt.grid(True)
    plt.savefig('accuracy_curve.png')
    plt.show()

if __name__ == '__main__':
    main()