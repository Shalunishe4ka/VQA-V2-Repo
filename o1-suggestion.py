import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from torch.utils.data import Dataset, DataLoader
from transformers import BartTokenizer
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

from torchvision import models, transforms
from PIL import Image

import requests
import zipfile

# -----------------------------------------------------------------------------------------
# Тимур, ниже представлен переработанный код. В нём:
# 1. Динамически извлекаются признаки из изображений с помощью предобученной ResNet-50.
# 2. Не используются заранее сохранённые .npy с фичами изображений.
# 3. Добавлены преобразования изображений, датасет загружает изображения по image_id.
# 4. Код модели и обучение в целом оставлены похожими, но вы можете адаптировать архитектуру под свою задачу.
# 5. Добавлен дополнительный класс VisionEncoder, который извлекает визуальные признаки из изображений.
# 6. Предусмотрена возможность использования нескольких GPU (DataParallel).
# 7. Используется ранняя остановка и сохранение лучших чекпоинтов.
#
# Вам останется только убедиться, что у вас лежат нужные датасеты в правильных директориях:
# dataset_dir/
# ├─ train2014/ (изображения)
# ├─ val2014/ (изображения)
# ├─ v2_mscoco_train2014_annotations.json
# ├─ v2_mscoco_val2014_annotations.json
# ├─ v2_OpenEnded_mscoco_train2014_questions.json
# └─ v2_OpenEnded_mscoco_val2014_questions.json
#
# Имена изображений предполагаются стандартные для COCO, например:
# COCO_train2014_000000XXXXXX.jpg
# COCO_val2014_000000XXXXXX.jpg
#
# При необходимости вы можете адаптировать пути к изображениям под ваш формат.
# -----------------------------------------------------------------------------------------


# ----------------------- Гиперпараметры и пути -----------------------
dataset_dir = './Dataset/'
checkpoint_dir = './Checkpoints/'
max_len = 30            # Вопросы и ответы короче, чем 200 токенов, можно уменьшить для скорости
embed_size = 512        # Размер эмбеддингов может быть меньше, чем 2048 (2048 - очень большой)
num_heads = 8
num_encoder_layers = 4
num_decoder_layers = 4
dropout = 0.3
num_epochs = 10
batch_size = 32
learning_rate = 0.0001
pad_token_idx = 1       # Для BART pad_token_id обычно равен 1
# device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

def download_and_extract_dataset(dataset_dir, urls):
    """
    Скачивает и извлекает датасеты.

    Parameters:
        dataset_dir (str): Путь к директории для сохранения данных.
        urls (list): Список ссылок на файлы для скачивания.
    """
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)

    for url in urls:
        filename = url.split('/')[-1]
        file_path = os.path.join(dataset_dir, filename)

        # Скачиваем файл, если он ещё не скачан
        if not os.path.exists(file_path):
            print(f"Скачивание {filename}...")
            with requests.get(url, stream=True) as r:
                r.raise_for_status()
                with open(file_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
            print(f"Файл {filename} скачан.")
        else:
            print(f"Файл {filename} уже существует.")

        # Извлекаем архив, если он не извлечён
        if zipfile.is_zipfile(file_path):
            extract_dir = os.path.join(dataset_dir, filename.replace('.zip', ''))
            if not os.path.exists(extract_dir):
                print(f"Извлечение {filename}...")
                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                    zip_ref.extractall(dataset_dir)
                print(f"Архив {filename} извлечён.")
            else:
                print(f"Архив {filename} уже извлечён.")
        else:
            print(f"{filename} не является ZIP-архивом.")


# Список ссылок на датасеты
urls = [
    "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Train_mscoco.zip",
    "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip",
    "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Train_mscoco.zip",
    "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip",
    "http://images.cocodataset.org/zips/train2014.zip",
    "http://images.cocodataset.org/zips/val2014.zip",
    "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Complementary_Pairs_Train_mscoco.zip",
    "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Complementary_Pairs_Val_mscoco.zip"
]

# Скачиваем и извлекаем
download_and_extract_dataset(dataset_dir, urls)

# ----------------------- Загрузка данных аннотаций и вопросов -----------------------
def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

train_annotations = load_json(os.path.join(dataset_dir, 'v2_mscoco_train2014_annotations.json'))
val_annotations = load_json(os.path.join(dataset_dir, 'v2_mscoco_val2014_annotations.json'))
train_questions = load_json(os.path.join(dataset_dir, 'v2_OpenEnded_mscoco_train2014_questions.json'))
val_questions = load_json(os.path.join(dataset_dir, 'v2_OpenEnded_mscoco_val2014_questions.json'))

def extract_data(questions, annotations):
    questions_list = [q['question'] for q in questions['questions']]
    answers_list = [a['multiple_choice_answer'] for a in annotations['annotations']]
    qids = [q['question_id'] for q in questions['questions']]
    img_ids = [q['image_id'] for q in questions['questions']]
    return questions_list, answers_list, qids, img_ids

train_questions_list, train_answers_list, train_qids, train_imgids = extract_data(train_questions, train_annotations)
val_questions_list, val_answers_list, val_qids, val_imgids = extract_data(val_questions, val_annotations)

# ----------------------- Токенизация -----------------------
# Используем BART токенизатор
tokenizer = BartTokenizer.from_pretrained('facebook/bart-base', clean_up_tokenization_spaces=True)

encoded_train_questions = tokenizer.batch_encode_plus(
    train_questions_list,
    padding='max_length',
    truncation=True,
    max_length=max_len,
    return_tensors="pt"
)
encoded_val_questions = tokenizer.batch_encode_plus(
    val_questions_list,
    padding='max_length',
    truncation=True,
    max_length=max_len,
    return_tensors="pt"
)
encoded_train_answers = tokenizer.batch_encode_plus(
    train_answers_list,
    padding='max_length',
    truncation=True,
    max_length=max_len,
    return_tensors="pt"
)
encoded_val_answers = tokenizer.batch_encode_plus(
    val_answers_list,
    padding='max_length',
    truncation=True,
    max_length=max_len,
    return_tensors="pt"
)

train_input_ids = encoded_train_questions['input_ids']
val_input_ids = encoded_val_questions['input_ids']
train_answer_ids = encoded_train_answers['input_ids']
val_answer_ids = encoded_val_answers['input_ids']

# ----------------------- Подготовка трансформаций для изображений -----------------------
image_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # стандартные статистики для ImageNet
        std=[0.229, 0.224, 0.225]
    )
])

# ----------------------- Класс VisionEncoder для извлечения фич из изображений -----------------------
class VisionEncoder(nn.Module):
    def __init__(self, embed_size=embed_size):
        super(VisionEncoder, self).__init__()
        # Используем предобученный ResNet-50
        resnet = models.resnet50(pretrained=True)
        # Заморозим веса ранних слоёв при необходимости (для экономии ресурсов), можно закомментировать:
        for param in resnet.parameters():
            param.requires_grad = False

        # Удаляем последний слой, чтобы взять фичи до классификации
        # Это pool фичи размером [batch_size, 2048]
        self.base_model = nn.Sequential(*list(resnet.children())[:-1])
        self.fc = nn.Linear(2048, embed_size)

    def forward(self, images):
        with torch.no_grad():
            features = self.base_model(images)  # (N, 2048, 1, 1)
        features = features.view(features.size(0), -1)  # (N, 2048)
        features = self.fc(features)  # (N, embed_size)
        return features

# ----------------------- Датасет с динамическим извлечением фич -----------------------
class VQADataset(Dataset):
    def __init__(self, questions, answers, img_ids, split='train', transform=None, dataset_dir='./Dataset'):
        self.questions = questions
        self.answers = answers
        self.img_ids = img_ids
        self.transform = transform
        self.split = split
        self.dataset_dir = dataset_dir
        # Определяем директорию с изображениями
        if split == 'train':
            self.image_dir = os.path.join(dataset_dir, 'train2014')
            self.prefix = 'COCO_train2014_'
        else:
            self.image_dir = os.path.join(dataset_dir, 'val2014')
            self.prefix = 'COCO_val2014_'

    def __len__(self):
        return self.questions.size(0)

    def __getitem__(self, idx):
        question = self.questions[idx]
        answer = self.answers[idx]

        img_id = self.img_ids[idx]
        # Формируем имя файла изображения. image_id: int -> '000000' + id
        img_filename = f"{self.prefix}{img_id:012d}.jpg"
        img_path = os.path.join(self.image_dir, img_filename)

        image = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        return question, image, answer

train_dataset = VQADataset(train_input_ids, train_answer_ids, train_imgids, split='train', transform=image_transforms, dataset_dir=dataset_dir)
val_dataset = VQADataset(val_input_ids, val_answer_ids, val_imgids, split='val', transform=image_transforms, dataset_dir=dataset_dir)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=8,
    pin_memory=True
)
val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=8,
    pin_memory=True
)

# ----------------------- Вспомогательные функции -----------------------
def generate_mask(sequence, pad_token_idx):
    return (sequence != pad_token_idx).unsqueeze(1).unsqueeze(2)  # (N, 1, 1, seq_len)

def check_answer_range(answers, vocab_size):
    if (answers >= vocab_size).any() or (answers < 0).any():
        raise ValueError("Некоторые значения в answers выходят за пределы словаря.")

def compute_rouge_bleu(outputs, labels, tokenizer):
    predicted_ids = torch.argmax(outputs, dim=-1).cpu().numpy()  # (batch_size, seq_len)
    labels = labels.cpu().numpy()  # (batch_size, seq_len)
    rouge_scorer_fn = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    smooth_fn = SmoothingFunction().method4

    rouge_scores, bleu_scores = [], []
    for pred, label in zip(predicted_ids, labels):
        pred_text = tokenizer.decode(pred, skip_special_tokens=True)
        label_text = tokenizer.decode(label, skip_special_tokens=True)
        if not label_text.strip():
            continue
        rouge_score = rouge_scorer_fn.score(label_text, pred_text)['rougeL'].fmeasure
        bleu_score = sentence_bleu([label_text.split()], pred_text.split(), smoothing_function=smooth_fn)
        rouge_scores.append(rouge_score)
        bleu_scores.append(bleu_score)

    avg_rouge = np.mean(rouge_scores) if rouge_scores else 0.0
    avg_bleu = np.mean(bleu_scores) if bleu_scores else 0.0
    return avg_rouge, avg_bleu

class EarlyStopping:
    def __init__(self, patience=5, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.delta = delta

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
            if self.verbose:
                print(f"Initial validation loss set to {self.best_loss:.4f}")
        elif val_loss > self.best_loss - self.delta:
            self.counter += 1
            if self.verbose:
                print(f"Early stopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
            if self.verbose:
                print(f"Validation loss improved to {self.best_loss:.4f}. Resetting counter.")

# ----------------------- Определение модели -----------------------
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, heads, encoder=False):
        super(MultiHeadAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        assert self.head_dim * heads == embed_size, "Embed size must be divisible by heads"

        self.values = nn.Linear(embed_size, embed_size, bias=False)
        self.keys = nn.Linear(embed_size, embed_size, bias=False)
        self.queries = nn.Linear(embed_size, embed_size, bias=False)
        self.fc_out = nn.Linear(embed_size, embed_size)
        self.encoder = encoder

    def forward(self, values, keys, query, mask):
        N, query_len, _ = query.shape

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(query)

        values = values.view(N, -1, self.heads, self.head_dim).permute(0,2,1,3)
        keys = keys.view(N, -1, self.heads, self.head_dim).permute(0,2,1,3)
        queries = queries.view(N, -1, self.heads, self.head_dim).permute(0,2,1,3)

        energy = torch.matmul(queries, keys.permute(0,1,3,2))
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        attention = F.softmax(energy, dim=-1)
        out = torch.matmul(attention, values)

        out = out.permute(0,2,1,3).contiguous().view(N, query_len, self.embed_size)
        return self.fc_out(out)

class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_size, heads, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(embed_size, heads, encoder=True)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, embed_size * 4),
            nn.ReLU(),
            nn.Linear(embed_size * 4, embed_size)
        )
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask):
        attention = self.attention(x, x, x, mask)
        x = self.norm1(attention + x)
        x = self.dropout1(x)
        forward = self.feed_forward(x)
        x = self.norm2(forward + x)
        x = self.dropout2(x)
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, embed_size, heads, num_layers, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList(
            [TransformerEncoderLayer(embed_size, heads, dropout) for _ in range(num_layers)]
        )

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return x

class TransformerDecoderLayer(nn.Module):
    def __init__(self, embed_size, heads, dropout=0.1):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(embed_size, heads)
        self.cross_attn = MultiHeadAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.norm3 = nn.LayerNorm(embed_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, embed_size * 4),
            nn.ReLU(),
            nn.Linear(embed_size * 4, embed_size)
        )
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x, visual_features, enc_output, mask):
        N, query_len, _ = x.size()
        # Self-attention
        self_attention_mask = mask[:, :, :, :query_len]
        self_attention = self.self_attn(x, x, x, self_attention_mask)
        x = self.norm1(self_attention + x)
        x = self.dropout1(x)

        # Добавляем визуальные признаки и затем кросс-аттеншен
        # visual_features: (N, embed_size) -> (N, 1, embed_size)
        visual_features = visual_features.unsqueeze(1)
        combined_context = torch.cat([visual_features, enc_output], dim=1)

        cross_attention_mask = torch.cat([
            torch.ones(mask.size(0), 1, query_len, 1, device=mask.device),
            mask.expand(-1, -1, query_len, -1)
        ], dim=-1)

        cross_attention = self.cross_attn(combined_context, combined_context, x, cross_attention_mask)
        x = self.norm2(cross_attention + x)
        x = self.dropout2(x)

        forward = self.feed_forward(x)
        x = self.norm3(forward + x)
        x = self.dropout3(x)
        return x

class TransformerDecoder(nn.Module):
    def __init__(self, embed_size, heads, num_layers, dropout=0.1):
        super(TransformerDecoder, self).__init__()
        self.layers = nn.ModuleList(
            [TransformerDecoderLayer(embed_size, heads, dropout) for _ in range(num_layers)]
        )

    def forward(self, x, visual_features, enc_output, mask):
        for layer in self.layers:
            x = layer(x, visual_features, enc_output, mask)
        return x

class VQATransformer(nn.Module):
    def __init__(self, embed_size, heads, num_encoder_layers, num_decoder_layers, dropout=0.1, vocab_size=None):
        super(VQATransformer, self).__init__()
        self.question_embedding = nn.Embedding(vocab_size, embed_size, padding_idx=pad_token_idx)
        self.answer_embedding = nn.Embedding(vocab_size, embed_size, padding_idx=pad_token_idx)
        self.encoder = TransformerEncoder(embed_size, heads, num_encoder_layers, dropout)
        self.decoder = TransformerDecoder(embed_size, heads, num_decoder_layers, dropout)
        self.fc_out = nn.Linear(embed_size, vocab_size)

        # Визуальный энкодер
        self.vision_encoder = VisionEncoder(embed_size=embed_size)

    def forward(self, questions, images, question_mask, answer_mask):
        embedded_questions = self.question_embedding(questions)
        embedded_answers = self.answer_embedding(questions)  # тут можно сдвигать ответы для обучения teacher forcing

        enc_output = self.encoder(embedded_questions, question_mask)
        visual_features = self.vision_encoder(images)  # (N, embed_size)

        decoded_answers = self.decoder(embedded_answers, visual_features, enc_output, answer_mask)
        out = self.fc_out(decoded_answers)
        return out

vocab_size = tokenizer.vocab_size
model = VQATransformer(
    embed_size,
    num_heads,
    num_encoder_layers,
    num_decoder_layers,
    dropout,
    vocab_size
).to(device)

criterion = nn.CrossEntropyLoss(ignore_index=pad_token_idx)
optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)
early_stopping = EarlyStopping(patience=5, verbose=True)


def validate_vqa_transformer(model, val_loader, criterion, device, pad_token_idx, tokenizer):
    model.eval()
    total_loss = 0
    all_rouge_scores = []
    all_bleu_scores = []
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="[Validation]", leave=False):
            questions, images, answers = batch
            check_answer_range(answers, vocab_size)
            questions = questions.to(device)
            images = images.to(device)
            answers = answers.to(device)

            question_mask = generate_mask(questions, pad_token_idx).to(device)
            answer_mask = generate_mask(answers, pad_token_idx).to(device)

            outputs = model(questions, images, question_mask, answer_mask)
            loss = criterion(outputs.view(-1, outputs.size(-1)), answers.view(-1))
            total_loss += loss.item()

            avg_rouge, avg_bleu = compute_rouge_bleu(outputs, answers, tokenizer)
            all_rouge_scores.append(avg_rouge)
            all_bleu_scores.append(avg_bleu)

    avg_loss = total_loss / len(val_loader)
    avg_rouge = np.mean(all_rouge_scores) if all_rouge_scores else 0.0
    avg_bleu = np.mean(all_bleu_scores) if all_bleu_scores else 0.0
    return avg_loss, avg_rouge, avg_bleu


# ----------------------- Обёртка модели в DataParallel (при наличии нескольких GPU) -----------------------
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs for training.")
    model = nn.DataParallel(model)
model.to(device)


def save_checkpoint(model, optimizer, epoch, loss, save_dir, filename):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, filename)

    model_state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()

    torch.save({
        'epoch': epoch,
        'model_state_dict': model_state,
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, save_path)
    print(f"Checkpoint saved at {save_path}")

def load_checkpoint(model, optimizer, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    if isinstance(model, nn.DataParallel):
        model.module.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print(f"Checkpoint loaded from {checkpoint_path}, Epoch: {epoch}, Loss: {loss:.4f}")
    return epoch, loss

def train_vqa_transformer(model, train_loader, val_loader, optimizer, criterion, device, num_epochs, pad_token_idx,
                          tokenizer, scheduler=None, early_stopping=None, checkpoint_dir='./Checkpoints'):
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for batch in tqdm(train_loader, desc=f"[Training Epoch {epoch + 1}/{num_epochs}]", leave=True):
            questions, images, answers = batch
            check_answer_range(answers, vocab_size)
            questions = questions.to(device)
            images = images.to(device)
            answers = answers.to(device)

            question_mask = generate_mask(questions, pad_token_idx).to(device)
            answer_mask = generate_mask(answers, pad_token_idx).to(device)

            outputs = model(questions, images, question_mask, answer_mask)
            loss = criterion(outputs.view(-1, outputs.size(-1)), answers.view(-1))
            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {avg_loss:.4f}")

        avg_val_loss, avg_rouge, avg_bleu = validate_vqa_transformer(
            model, val_loader, criterion, device, pad_token_idx, tokenizer
        )
        print(
            f"Epoch {epoch + 1}/{num_epochs}, Validation Loss: {avg_val_loss:.4f}, ROUGE: {avg_rouge:.4f}, BLEU: {avg_bleu:.4f}"
        )

        # Сохранение чекпоинта после каждой эпохи
        save_checkpoint(model, optimizer, epoch, avg_loss, checkpoint_dir, f'checkpoint_epoch_{epoch + 1}.pth')

        # Сохранение лучшей модели
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_checkpoint(model, optimizer, epoch, avg_val_loss, checkpoint_dir, 'best_model.pth')

        if scheduler is not None:
            scheduler.step(avg_val_loss)

        if early_stopping is not None:
            early_stopping(avg_val_loss)
            if early_stopping.early_stop:
                print("Early stopping triggered.")
                break

# ----------------------- Запуск обучения -----------------------
train_vqa_transformer(
    model,
    train_loader,
    val_loader,
    optimizer,
    criterion,
    device,
    num_epochs,
    pad_token_idx,
    tokenizer,
    scheduler,
    early_stopping
)
