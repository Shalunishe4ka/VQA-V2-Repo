import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import BartTokenizer
from tqdm import tqdm
# Метрики: BLEU, ROUGE, METEOR и CIDEr
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from pycocoevalcap.cider.cider import Cider
from torchvision import models, transforms
from torchvision.ops import roi_align
from PIL import Image
import requests
import zipfile

# Гиперпараметры и пути
dataset_dir = './Dataset/'
checkpoint_dir = './Checkpoints/'
max_len = 100  # Увеличили максимальную длину последовательностей
embed_size = 768
num_heads = 8
num_encoder_layers = 6  # Увеличили количество слоёв
num_decoder_layers = 6
dropout = 0.3
num_epochs = 20  # Увеличили количество эпох
batch_size = 64
learning_rate = 5e-5  # Снижаем learning rate
pad_token_idx = 1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Функция скачивания и распаковки датасета
def download_and_extract_dataset(dataset_dir, urls):
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)
    for url in urls:
        filename = url.split('/')[-1]
        file_path = os.path.join(dataset_dir, filename)
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

# Пример ссылок (при необходимости скачать датасет)
urls = [
    "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Train_mscoco.zip",
    "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip",
    "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Train_mscoco.zip",
    "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip",
    "http://images.cocodataset.org/zips/train2014.zip",
    "http://images.cocodataset.org/zips/val2014.zip",
]

# Если датасет ещё не скачан, раскомментируйте:
# download_and_extract_dataset(dataset_dir, urls)

# Загрузка аннотаций и вопросов
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

# Токенизация
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

# Преобразования изображений
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomRotation(10),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Bottom-Up Attention Encoder
class BottomUpAttentionEncoder(nn.Module):
    def __init__(self, embed_size, num_regions=36):
        super(BottomUpAttentionEncoder, self).__init__()
        self.num_regions = num_regions
        from torchvision.models.detection import fasterrcnn_resnet50_fpn
        from torchvision.models.detection.faster_rcnn import FasterRCNN_ResNet50_FPN_Weights

        self.detector = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1)
        self.detector.transform.min_size = (224,)
        self.detector.transform.max_size = 224
        self.detector.eval()
        for param in self.detector.parameters():
            param.requires_grad = False
        self.fc = nn.Linear(256 * 7 * 7, embed_size)

    def train(self, mode=True):
        # Гарантируем, что детектор всегда остаётся в eval mode
        super(BottomUpAttentionEncoder, self).train(False)
        self.detector.eval()
        return self

    def forward(self, images):
        assert images.dim() == 4, "Входные изображения должны иметь размерность (N, C, H, W)"
        assert images.shape[1] == 3, "Количество каналов должно быть равно 3"

        with torch.no_grad():
            self.detector.eval()  # Убеждаемся, что детектор в режиме оценки
            detections = self.detector(images)
            region_features = []
            for i, det in enumerate(detections):
                boxes = det['boxes']
                scores = det['scores']
                if boxes.size(0) > self.num_regions:
                    topk = scores.topk(self.num_regions)
                    boxes = boxes[topk.indices]
                num_boxes = boxes.size(0)
                if num_boxes < self.num_regions:
                    pad = torch.zeros((self.num_regions - num_boxes, 4), device=boxes.device)
                    boxes = torch.cat([boxes, pad], dim=0)
                features = self.detector.backbone(images[i:i+1])
                feat_map = features['0']
                _, C, H_feat, W_feat = feat_map.shape
                scale_x = W_feat / 224.0
                scale_y = H_feat / 224.0
                boxes_feat = boxes.clone()
                boxes_feat[:, [0, 2]] = boxes_feat[:, [0, 2]] * scale_x
                boxes_feat[:, [1, 3]] = boxes_feat[:, [1, 3]] * scale_y
                batch_index = torch.zeros((boxes_feat.shape[0], 1), device=boxes_feat.device)
                rois = torch.cat([batch_index, boxes_feat], dim=1)
                pooled = roi_align(feat_map, rois, output_size=(7, 7))
                pooled = pooled.view(pooled.size(0), -1)
                pooled = self.fc(pooled)
                region_features.append(pooled)
            region_features = torch.stack(region_features, dim=0)
            return region_features

# Класс датасета
class VQADataset(Dataset):
    def __init__(self, questions, answers, img_ids, split='train', transform=None, dataset_dir='./Dataset'):
        self.questions = questions
        self.answers = answers
        self.img_ids = img_ids
        self.transform = transform
        self.split = split
        self.dataset_dir = dataset_dir
        if split == 'train':
            self.image_dir = os.path.join(dataset_dir, 'train2014')
            self.prefix = 'COCO_train2014_'
        else:
            self.image_dir = os.path.join(dataset_dir, 'val2014')
            self.prefix = 'COCO_val2014_'

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        question = self.questions[idx]
        answer = self.answers[idx]
        img_id = self.img_ids[idx]
        img_filename = f"{self.prefix}{img_id:012d}.jpg"
        img_path = os.path.join(self.image_dir, img_filename)
        image = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return question, image, answer

train_dataset = VQADataset(train_input_ids, train_answer_ids, train_imgids, split='train', transform=train_transforms, dataset_dir=dataset_dir)
val_dataset = VQADataset(val_input_ids, val_answer_ids, val_imgids, split='val', transform=val_transforms, dataset_dir=dataset_dir)

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

# Вспомогательные функции
def generate_mask(sequence, pad_token_idx):
    return (sequence != pad_token_idx).unsqueeze(1).unsqueeze(2)

def compute_metrics(outputs, labels, tokenizer):
    predicted_ids = torch.argmax(outputs, dim=-1).cpu().numpy()
    labels = labels.cpu().numpy()
    rouge_scorer_fn = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    smooth_fn = SmoothingFunction().method4
    cider_scorer = Cider()

    rouge_scores, bleu_scores, meteor_scores, cider_scores = [], [], [], []
    for pred, label in zip(predicted_ids, labels):
        pred_text = tokenizer.decode(pred, skip_special_tokens=True)
        label_text = tokenizer.decode(label, skip_special_tokens=True)
        if not label_text.strip():
            continue
        rouge_val = rouge_scorer_fn.score(label_text, pred_text)['rougeL'].fmeasure
        bleu_val = sentence_bleu([label_text.split()], pred_text.split(), smoothing_function=smooth_fn)
        meteor_val = meteor_score([label_text], pred_text)
        cider_val, _ = cider_scorer.compute_score({0: [label_text]}, {0: [pred_text]})
        rouge_scores.append(rouge_val)
        bleu_scores.append(bleu_val)
        meteor_scores.append(meteor_val)
        cider_scores.append(cider_val)

    metrics = {
        "ROUGE": np.mean(rouge_scores) if rouge_scores else 0.0,
        "BLEU": np.mean(bleu_scores) if bleu_scores else 0.0,
        "METEOR": np.mean(meteor_scores) if meteor_scores else 0.0,
        "CIDEr": np.mean(cider_scores) if cider_scores else 0.0,
    }
    return metrics

# Архитектура Transformer
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, heads, encoder=False):
        super(MultiHeadAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        assert self.head_dim * heads == embed_size, "Размер эмбеддинга должен делиться на количество голов"
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

        values = values.view(N, -1, self.heads, self.head_dim).permute(0, 2, 1, 3)
        keys = keys.view(N, -1, self.heads, self.head_dim).permute(0, 2, 1, 3)
        queries = queries.view(N, -1, self.heads, self.head_dim).permute(0, 2, 1, 3)

        energy = torch.matmul(queries, keys.permute(0, 1, 3, 2))
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))
        attention = torch.softmax(energy, dim=-1)
        out = torch.matmul(attention, values)
        out = out.permute(0, 2, 1, 3).contiguous().view(N, query_len, self.embed_size)
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
        attn = self.attention(x, x, x, mask)
        x = self.norm1(attn + x)
        x = self.dropout1(x)
        ff = self.feed_forward(x)
        x = self.norm2(ff + x)
        x = self.dropout2(x)
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, embed_size, heads, num_layers, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([TransformerEncoderLayer(embed_size, heads, dropout) for _ in range(num_layers)])

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
        self_mask = mask[:, :, :, :query_len]
        self_attn = self.self_attn(x, x, x, self_mask)
        x = self.norm1(self_attn + x)
        x = self.dropout1(x)

        combined_context = torch.cat([visual_features, enc_output], dim=1)
        cross_mask = torch.ones(mask.size(0), 1, query_len, visual_features.size(1) + enc_output.size(1), device=mask.device)
        cross_attn = self.cross_attn(combined_context, combined_context, x, cross_mask)
        x = self.norm2(cross_attn + x)
        x = self.dropout2(x)

        ff = self.feed_forward(x)
        x = self.norm3(ff + x)
        x = self.dropout3(x)
        return x

class TransformerDecoder(nn.Module):
    def __init__(self, embed_size, heads, num_layers, dropout=0.1):
        super(TransformerDecoder, self).__init__()
        self.layers = nn.ModuleList([TransformerDecoderLayer(embed_size, heads, dropout) for _ in range(num_layers)])

    def forward(self, x, visual_features, enc_output, mask):
        for layer in self.layers:
            x = layer(x, visual_features, enc_output, mask)
        return x

class VQATransformer(nn.Module):
    def __init__(self, embed_size, heads, num_encoder_layers, num_decoder_layers, dropout, vocab_size):
        super(VQATransformer, self).__init__()
        self.question_embedding = nn.Embedding(vocab_size, embed_size, padding_idx=pad_token_idx)
        self.answer_embedding = nn.Embedding(vocab_size, embed_size, padding_idx=pad_token_idx)
        self.encoder = TransformerEncoder(embed_size, heads, num_encoder_layers, dropout)
        self.decoder = TransformerDecoder(embed_size, heads, num_decoder_layers, dropout)
        self.fc_out = nn.Linear(embed_size, vocab_size)
        self.vision_encoder = BottomUpAttentionEncoder(embed_size=embed_size, num_regions=100)

    def forward(self, questions, images, question_mask, answer_mask):
        embedded_questions = self.question_embedding(questions)
        embedded_answers = self.answer_embedding(questions)
        enc_output = self.encoder(embedded_questions, question_mask)
        visual_features = self.vision_encoder(images)
        decoded = self.decoder(embedded_answers, visual_features, enc_output, answer_mask)
        out = self.fc_out(decoded)
        return out

vocab_size = tokenizer.vocab_size
model = VQATransformer(
    embed_size=embed_size,
    heads=num_heads,
    num_encoder_layers=num_encoder_layers,
    num_decoder_layers=num_decoder_layers,
    dropout=dropout,
    vocab_size=vocab_size
).to(device)



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
                print(f"Начальное значение валидационной ошибки: {self.best_loss:.4f}")
        elif val_loss > self.best_loss - self.delta:
            self.counter += 1
            if self.verbose:
                print(f"Счётчик ранней остановки: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
            if self.verbose:
                print(f"Ошибка улучшилась до {self.best_loss:.4f}. Сброс счётчика.")

criterion = nn.CrossEntropyLoss(ignore_index=pad_token_idx)
optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)
early_stopping = EarlyStopping(patience=5, verbose=True)

# Обёртка для нескольких GPU
if torch.cuda.device_count() > 1:
    print(f"Используем {torch.cuda.device_count()} GPUs для обучения.")
    model = nn.DataParallel(model)
model.to(device)

# Функции валидации и сохранения
def validate_vqa_transformer(model, val_loader, criterion, device, pad_token_idx, tokenizer):
    model.eval()
    total_loss = 0
    all_metrics = {"ROUGE": [], "BLEU": [], "METEOR": [], "CIDEr": []}
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="[Validation]", leave=False):
            questions, images, answers = batch
            questions = questions.to(device)
            images = images.to(device)
            answers = answers.to(device)
            question_mask = generate_mask(questions, pad_token_idx).to(device)
            answer_mask = generate_mask(answers, pad_token_idx).to(device)
            outputs = model(questions, images, question_mask, answer_mask)
            loss = criterion(outputs.view(-1, outputs.size(-1)), answers.view(-1))
            total_loss += loss.item()
            metrics = compute_metrics(outputs, answers, tokenizer)
            for key in all_metrics:
                all_metrics[key].append(metrics[key])
    avg_loss = total_loss / len(val_loader)
    avg_metrics = {k: np.mean(v) if v else 0.0 for k, v in all_metrics.items()}
    return avg_loss, avg_metrics

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
    print(f"Чекпоинт сохранён: {save_path}")

def load_checkpoint(model, optimizer, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    if isinstance(model, nn.DataParallel):
        model.module.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print(f"Загружен чекпоинт: {checkpoint_path} (Эпоха {epoch}, Loss: {loss:.4f})")
    return epoch, loss

# Функция обучения
def train_vqa_transformer(model, train_loader, val_loader, optimizer, criterion, device, num_epochs, pad_token_idx, tokenizer, scheduler=None, early_stopping=None, checkpoint_dir='./Checkpoints'):
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"[Epoch {epoch+1}/{num_epochs} - Training]", leave=True):
            questions, images, answers = batch
            questions = questions.to(device)
            images = images.to(device)
            answers = answers.to(device)
            question_mask = generate_mask(questions, pad_token_idx).to(device)
            answer_mask = generate_mask(answers, pad_token_idx).to(device)
            optimizer.zero_grad()
            outputs = model(questions, images, question_mask, answer_mask)
            loss = criterion(outputs.view(-1, outputs.size(-1)), answers.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_train_loss = total_loss / len(train_loader)
        print(f"Эпоха {epoch+1}/{num_epochs}, Training Loss: {avg_train_loss:.4f}")

        avg_val_loss, metrics = validate_vqa_transformer(model, val_loader, criterion, device, pad_token_idx, tokenizer)
        print(f"Эпоха {epoch+1}/{num_epochs}, Validation Loss: {avg_val_loss:.4f}")
        print(f"Метрики: ROUGE: {metrics['ROUGE']:.4f}, BLEU: {metrics['BLEU']:.4f}, METEOR: {metrics['METEOR']:.4f}, CIDEr: {metrics['CIDEr']:.4f}")

        save_checkpoint(model, optimizer, epoch, avg_train_loss, checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth')
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_checkpoint(model, optimizer, epoch, avg_val_loss, checkpoint_dir, 'best_model.pth')

        if scheduler is not None:
            scheduler.step(avg_val_loss)

        if early_stopping is not None:
            early_stopping(avg_val_loss)
            if early_stopping.early_stop:
                print("Ранняя остановка сработала.")
                break

# Запуск обучения
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
    early_stopping,
    checkpoint_dir
)