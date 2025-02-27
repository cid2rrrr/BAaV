import os,sys,librosa,torchaudio

ris_path = os.path.abspath("./ASDA")
if ris_path not in sys.path:
    sys.path.append(ris_path)

import torch
import numpy as np
import torch.nn.functional as F
import cv2
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Normalize
from transformers import BlipProcessor, BlipForConditionalGeneration
from msclap import CLAP
import spacy
import clip
import pandas as pd

from model.model import *
from engine.engine import *
from dataset.data_loader import *
from utils.losses import *
from utils.parsing_metrics import *
from utils.utils import *
from utils.checkpoint import load_pretrain


blip_processor = None
blip_model = None
clap_model = None
ris_model = None
nlp_model = None
beats_model = None
beats_checkpoint = None


def initialize_models(args, device, method):
    """선택한 method에 따라 필요한 모델만 로드"""
    global blip_processor, blip_model, clap_model, ris_model, nlp_model, beats_model, beats_checkpoint

    if ris_model is None:
        ris_model = Model(clip_model=args.clip_model, tunelang=args.tunelang, num_query=args.num_query, fusion_dim=args.fusion_dim).to(device)
        ris_model = load_pretrain(ris_model, args, None, 1).eval()

    if method == 'cross_modal':
        if blip_processor is None:
            blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        if blip_model is None:
            blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
        if clap_model is None:
            clap_model = CLAP(version='2023', use_cuda=(device == 'cuda'))
        if nlp_model is None:
            nlp_model = spacy.load("en_core_web_trf")

    elif method == 'caption':
        if blip_model is None:
            blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

    elif method == 'classify':
        if beats_model is None:
            beats_model = torch.hub.load('facebookresearch/BEATs', 'BEATs_large')
            beats_model.eval()
        if beats_checkpoint is None:
            beats_checkpoint = torch.load('/home/cid2r/GitHub/BEATs_checkpoint.pth', map_location=device)


def center_crop_resize(img, size=224):
    """이미지를 지정된 크기로 중심에서 크롭 후 리사이징"""
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


def letterbox(img, size=224):
    """Letterbox 방식으로 이미지 크기 조정"""
    h, w = img.shape[:2]
    scale = size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    
    img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    padded_img = np.full((size, size, 3), 114, dtype=np.uint8)
    
    y_offset = (size - new_h) // 2
    x_offset = (size - new_w) // 2
    padded_img[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = img_resized
    
    return padded_img


@torch.no_grad()
def evalutate(args, img_path=None, wav_path=None):
    """선택한 method에 따라 평가 수행"""
    global blip_processor, blip_model, clap_model, ris_model, nlp_model, beats_model, beats_checkpoint

    if img_path is None:
        img_path = args.img_path
    if wav_path is None:
        wav_path = args.wav_path

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    initialize_models(args, device, args.method)

    input_transform = Compose([
        ToTensor(),
        Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
    ])

    image = Image.open(img_path).convert("RGB")
    img = np.array(image)

    if args.resize_mode == 'letterbox':
        img = letterbox(img, args.size)
    else:
        img = center_crop_resize(img, args.size)

    img = input_transform(img).to(device)

    if args.method == 'cross_modal':
        inputs = blip_processor(image, return_tensors="pt").to(device)
        outputs = blip_model.generate(**inputs, max_length=17, repetition_penalty=2.5, early_stopping=True)
        caption = blip_processor.decode(outputs[0], skip_special_tokens=True)

        doc = nlp_model(caption)
        nouns = [chunk.text for chunk in doc.noun_chunks]

        if not nouns:
            caption = "A photo of something"

        text_embed = clap_model.get_text_embeddings([caption])
        audio_embed = clap_model.get_audio_embeddings([wav_path])
        cos_sim = clap_model.compute_similarity(audio_embed, text_embed)
        pred_idx = cos_sim.argmax()
        caption = f"A photo of {nouns[pred_idx]}" if nouns else "A photo of something"

    elif args.method == 'caption':
        waveform, sr = librosa.load(wav_path, sr=32000, mono=True)
        waveform = torch.tensor(waveform).to(device)

        max_length = 32000 * 10
        if len(waveform) > max_length:
            waveform = waveform[:max_length]
        else:
            waveform = F.pad(waveform, [0, max_length - len(waveform)], "constant", 0.0)

        waveform = waveform.unsqueeze(0)
        caption = blip_model.generate(samples=waveform, num_beams=3)

    elif args.method == 'classify':
        y, sr = torchaudio.load(wav_path)
        if y.shape[0] != 1:
            y = torch.mean(y, dim=0, keepdim=True)
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
            y = resampler(y)

        padding_mask = torch.zeros(1, 10000).bool().to(device)
        probs = beats_model.extract_features(y.to(device), padding_mask=padding_mask)[0]
        
        df = pd.read_csv('/home/cid2r/GitHub/class_labels_indices.csv')
        mid_to_display_name = dict(zip(df['mid'], df['display_name']))
        
        caption = mid_to_display_name[beats_checkpoint['label_dict'][probs.argmax().item()]]

    word_id = clip.tokenize(caption, 17, truncate=True).to(device)
    word_mask = ~(word_id == 0)

    mask_out = ris_model(img.unsqueeze(0), word_id, word_mask)
    return F.interpolate(mask_out.sigmoid(), size=(224, 224), mode='bilinear', align_corners=False)



@torch.no_grad()
def evaluate_batch(args, img_paths, wav_paths, resize_mode='center_crop', method='cross_modal'):
    """다중 배치를 지원하는 평가 함수"""
    global blip_processor, blip_model, clap_model, ris_model, nlp_model

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 필요한 모델만 로드
    initialize_models(args, device, method)

    # 입력 변환
    input_transform = Compose([
        ToTensor(),
        Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
    ])

    # 1. **다중 이미지 전처리**
    images = []
    raw_images = []
    for img_path in img_paths:
        image = Image.open(img_path).convert("RGB")
        raw_images.append(image)  # 원본 저장 (Overlay 용)
        img = np.array(image)

        if resize_mode == 'letterbox':
            img = letterbox(img, args.size)
        else:
            img = center_crop_resize(img, args.size)

        images.append(input_transform(img).to(device))

    images = torch.stack(images)  # (B, C, H, W)

    # 2. **BLIP 다중 캡션 생성**
    inputs = blip_processor(raw_images, return_tensors="pt", padding=True).to(device)
    outputs = blip_model.generate(**inputs, max_length=17, repetition_penalty=2.5, early_stopping=True)
    captions = [blip_processor.decode(out, skip_special_tokens=True) for out in outputs]

    # 3. **명사 추출**
    batch_nouns = []
    for caption in captions:
        doc = nlp_model(caption)
        nouns = [chunk.text for chunk in doc.noun_chunks]
        batch_nouns.append(nouns if nouns else ["something"])  # 명사 없으면 기본값

    # 4. **CLAP 다중 텍스트 임베딩**
    flat_nouns = [noun for nouns in batch_nouns for noun in nouns]  # 모든 명사를 펼쳐서 리스트로 변환
    text_embeddings = clap_model.get_text_embeddings(flat_nouns)  # (N_total, D)

    # 명사 개수가 다 다르므로 각 이미지 별로 임베딩 인덱스를 계산
    text_emb_indices = []
    start = 0
    for nouns in batch_nouns:
        text_emb_indices.append((start, start + len(nouns)))
        start += len(nouns)

    # 5. **CLAP 다중 오디오 임베딩**
    audio_embeddings = clap_model.get_audio_embeddings(wav_paths)  # (B, D)

    # 6. **오디오-텍스트 유사도 계산 및 최적 명사 선택**
    selected_nouns = []
    for i, audio_emb in enumerate(audio_embeddings):
        text_emb_range = text_emb_indices[i]  # 해당 이미지에 해당하는 명사 임베딩 범위
        text_embs = text_embeddings[text_emb_range[0]:text_emb_range[1]]  # (N_i, D)
        cos_sim = clap_model.compute_similarity(audio_emb.unsqueeze(0), text_embs)  # (1, N_i)
        best_idx = cos_sim.argmax().item()
        selected_nouns.append(batch_nouns[i][best_idx])

    # 7. **CLIP 토큰화 및 RIS 예측**
    word_ids = clip.tokenize([f"A photo of {noun}" for noun in selected_nouns], 17, truncate=True).to(device)
    word_mask = ~(word_ids == 0)

    mask_outs = ris_model(images, word_ids, word_mask)  # (B, 1, H, W)
    preds = F.interpolate(mask_outs.sigmoid(), size=(224, 224), mode='bilinear', align_corners=False)

    return preds



