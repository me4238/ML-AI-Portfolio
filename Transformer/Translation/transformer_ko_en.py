# 1. Imports
import torch
import torch.nn as nn
import math
import sentencepiece as spm
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

# 2. Example Korean-English Data (can expand later)
data_pairs = [
    ("안녕하세요", "Hello"),
    ("저는 학생입니다", "I am a student"),
    ("이것은 책입니다", "This is a book"),
    ("당신을 만나서 반갑습니다", "Nice to meet you"),
    ("오늘 날씨 어때요?", "How's the weather today?")
]

# 3. Save to files for SentencePiece
with open("train.ko", "w", encoding="utf-8") as f:
    for ko, _ in data_pairs:
        f.write(ko + "\n")

with open("train.en", "w", encoding="utf-8") as f:
    for _, en in data_pairs:
        f.write(en + "\n")

# 4. Train SentencePiece Tokenizers
spm.SentencePieceTrainer.Train('--input=train.ko --model_prefix=spm_ko --vocab_size=800 --pad_id=0 --unk_id=1 --bos_id=2 --eos_id=3')
spm.SentencePieceTrainer.Train('--input=train.en --model_prefix=spm_en --vocab_size=800 --pad_id=0 --unk_id=1 --bos_id=2 --eos_id=3')

# 5. Load Tokenizers
sp_ko = spm.SentencePieceProcessor()
sp_en = spm.SentencePieceProcessor()
sp_ko.load("spm_ko.model")
sp_en.load("spm_en.model")

# 6. Encode Function
def encode_sentence(sp, sentence):
    return [sp.bos_id()] + sp.encode(sentence, out_type=int) + [sp.eos_id()]

# 7. Dataset Encoding
src_vocab_size = sp_ko.get_piece_size()
tgt_vocab_size = sp_en.get_piece_size()

pairs_encoded = [
    (
        torch.tensor(encode_sentence(sp_ko, ko), dtype=torch.long),
        torch.tensor(encode_sentence(sp_en, en), dtype=torch.long)
    )
    for ko, en in data_pairs
]

# 8. Dataloader Setup
def collate_fn(batch):
    src_batch, tgt_batch = zip(*batch)
    src_batch = pad_sequence(src_batch, batch_first=True, padding_value=0)
    tgt_batch = pad_sequence(tgt_batch, batch_first=True, padding_value=0)
    return src_batch, tgt_batch

loader = DataLoader(pairs_encoded, batch_size=2, shuffle=True, collate_fn=collate_fn)

# 9. Transformer Components
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -math.log(10000.0) / d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)].to(x.device)
        return x

class Transformer(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, d_model=256, nhead=4, num_layers=3):
        super().__init__()
        self.src_tok_emb = nn.Embedding(src_vocab, d_model)
        self.tgt_tok_emb = nn.Embedding(tgt_vocab, d_model)
        self.pos_enc = PositionalEncoding(d_model)
        self.transformer = nn.Transformer(d_model, nhead, num_layers, num_layers)
        self.fc_out = nn.Linear(d_model, tgt_vocab)

    def forward(self, src, tgt):
        src = self.pos_enc(self.src_tok_emb(src))
        tgt = self.pos_enc(self.tgt_tok_emb(tgt))
        output = self.transformer(src.transpose(0, 1), tgt.transpose(0, 1))
        return self.fc_out(output.transpose(0, 1))

# 10. Training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Transformer(src_vocab_size, tgt_vocab_size).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss(ignore_index=0)

print("🚀 Starting Training...\n")
for epoch in range(10):
    model.train()
    total_loss = 0
    for src, tgt in loader:
        src, tgt = src.to(device), tgt.to(device)
        tgt_input = tgt[:, :-1]
        tgt_out = tgt[:, 1:]

        logits = model(src, tgt_input)
        loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1} Loss: {total_loss:.4f}")
print("\n✅ Training Finished\n")

# 11. Translation Function
def translate(model, sentence, max_len=20):
    model.eval()
    src_tensor = torch.tensor([encode_sentence(sp_ko, sentence)], dtype=torch.long).to(device)
    tgt_tensor = torch.tensor([[sp_en.bos_id()]], dtype=torch.long).to(device)

    for _ in range(max_len):
        output = model(src_tensor, tgt_tensor)
        next_token = output[:, -1, :].argmax(-1).unsqueeze(1)
        tgt_tensor = torch.cat([tgt_tensor, next_token], dim=1)
        if next_token.item() == sp_en.eos_id():
            break

    translated = sp_en.decode(tgt_tensor.squeeze().tolist()[1:-1])
    return translated

# 12. Test Translations
print("🧪 Sample Translations:")
test_sentences = ["안녕하세요", "오늘 날씨 어때요?", "이것은 책입니다", "당신을 만나서 반갑습니다"]
for sentence in test_sentences:
    print(f"Korean: {sentence} --> English: {translate(model, sentence)}")
