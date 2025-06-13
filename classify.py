import torch
import re
import pickle

class RumourDetectClass:
    def __init__(self):

        # 加载模型和词汇表
        checkpoint = torch.load('bigru_model.pt', map_location='cuda:0' if torch.cuda.is_available() else 'cpu')
        with open('vocab.pkl', 'rb') as f:
            self.vocab = pickle.load(f)
        

        self.max_len = 64
        # 获取嵌入维度和隐藏层维度
        self.embedding_dim = checkpoint['embedding_dim']
        self.hidden_dim = checkpoint['hidden_dim']

        # Dropout率
        self.dropout_rate = checkpoint.get('dropout_rate', 0.3)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        # 初始化模型
        self.model = BiGRU(checkpoint['vocab_size'], self.embedding_dim, self.hidden_dim, self.dropout_rate).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

    def preprocess(self, text):
        text = text.lower()
        text = re.sub(r"[^a-zA-Z0-9.,!?\'\"\s]", "", text)
        return text

    def tokenize(self, text):
        return re.findall(r'\w+|[.,!?\'"]', text)

    def encode(self, text):
        tokens = self.tokenize(text)
        # 将文本转换为词汇表中的索引
        ids = [self.vocab.get(t, self.vocab['<UNK>']) for t in tokens]
        if len(ids) < self.max_len:
            ids += [self.vocab['<PAD>']] * (self.max_len - len(ids))
        else:
            ids = ids[:self.max_len]
        return torch.tensor([ids], dtype=torch.long).to(self.device)

    def classify(self, text: str, threshold = 0.5) -> int:
        """
        Perform rumor detection on input text.
        Args:
            text: Input text string
        Returns:
            int: Predicted class (0 for non-rumor, 1 for rumor)
        """
        text = self.preprocess(text)
        x = self.encode(text)
        with torch.no_grad():
            logit = self.model(x)
            # 使用sigmoid函数将logit转换为概率
            pred = (torch.sigmoid(logit) > threshold).float().item()
        return int(pred)


# BiGRU 模型类，与train_bigru_advance.py中的定义相同
class BiGRU(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, dropout_rate):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.dropout1 = torch.nn.Dropout(dropout_rate)
        self.bigru = torch.nn.GRU(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout2 = torch.nn.Dropout(dropout_rate)
        self.fc1 = torch.nn.Linear(hidden_dim*2, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 1)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        emb = self.embedding(x)
        emb = self.dropout1(emb)
        _, h = self.bigru(emb)
        h = torch.cat([h[0], h[1]], dim=1)
        h = self.dropout2(h)
        h = self.relu(self.fc1(h))
        out = self.fc2(h)
        return out.squeeze(1)
