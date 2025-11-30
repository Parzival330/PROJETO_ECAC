import numpy as np
import os
import torch
import sys
from collections import Counter

# Importar funções locais (da Parte A e do Extrator)
from mainActivity import load_participantes, features_window, window_majority_label
from embeddings_extractor import load_model, resample_to_30hz_5s

# --- CONFIGURAÇÕES GLOBAIS ---
FS = 50.0   
WINDOW_SEC = 5.0 
STEP_SEC = 5.0   
DATASET_BASE_PATH = r"FORTH_TRACE_DATASET-master"
FILENAME_NPZ = "datasets_partB.npz"

# Forçar CPU para evitar erros de compatibilidade com a RTX 5060
DEVICE = "cpu"
print(f"--- A USAR DISPOSITIVO: {DEVICE.upper()} ---")

def filtrar_atividades(data):
    """ Mantém apenas atividades 1 a 7. """
    atividades = data[:, 11]
    mask = (atividades >= 1) & (atividades <= 7)
    return data[mask]

def processar_janelas(data_participante, embed_model):
    """
    Processa os dados de UM participante.
    Retorna listas de features, embeddings e labels.
    """
    list_feats = []
    list_embeds = []
    list_labels = []
    
    total_samples = data_participante.shape[0]
    win_samples = int(WINDOW_SEC * FS)
    step_samples = int(STEP_SEC * FS)
    
    # Loop janela a janela
    for start in range(0, total_samples - win_samples, step_samples):
        end = start + win_samples
        segment = data_participante[start:end, :]
        
        label = window_majority_label(segment)
        if label < 1 or label > 7:
            continue
            
        # 1. Extrair Features Manuais (Parte A)
        f_vec, _ = features_window(segment, FS)
        
        # 2. Extrair Embeddings (Parte B)
        acc_xyz = segment[:, 1:4] # Apenas Aceleração X, Y, Z
        acc_resampled, _ = resample_to_30hz_5s(acc_xyz, FS)
        
        # Converter para Tensor PyTorch
        tensor_input = torch.from_numpy(acc_resampled).float().to(DEVICE)
        tensor_input = tensor_input.unsqueeze(0).permute(0, 2, 1) # (1, 3, Time)
        
        with torch.no_grad():
            embedding = embed_model(tensor_input)
            emb_vec = embedding.cpu().numpy().flatten()
            
        list_feats.append(f_vec)
        list_embeds.append(emb_vec)
        list_labels.append(label)
            
    return list_feats, list_embeds, list_labels

if __name__ == "__main__":
    print("[Modelo] A carregar modelo HARNet5...")
    embed_model = load_model()
    embed_model.to(DEVICE)
    embed_model.eval()

    # Listas Globais para guardar tudo
    all_feats = []
    all_embeds = []
    all_labels = []
    all_subjects = [] # Guarda o ID do participante

    print("1. A processar participantes (0 a 14)...")
    
    # CORREÇÃO CRÍTICA: Range 0 a 15 (exclui o 15), para apanhar part0 até part14
    for pid in range(0, 15):
        print(f" -> Participante {pid}...", end="")
        
        # 1. Carregar
        d = load_participantes(pid, base_path=DATASET_BASE_PATH)
        if d.size == 0: 
            print(" (Sem dados - ignorado)")
            continue
            
        # 2. Filtrar
        d = filtrar_atividades(d)
        
        # 3. Processar Janelas
        feats, embeds, labels = processar_janelas(d, embed_model)
        
        if len(feats) > 0:
            all_feats.extend(feats)
            all_embeds.extend(embeds)
            all_labels.extend(labels)
            # Regista que estas X janelas pertencem ao participante 'pid'
            all_subjects.extend([pid] * len(feats))
            print(f" OK ({len(feats)} janelas)")
        else:
            print(" (0 janelas válidas)")

    # Converter e Guardar
    X_feats = np.array(all_feats)
    X_embeds = np.array(all_embeds)
    y = np.array(all_labels)
    subjects = np.array(all_subjects)

    print("\n--- EXTRAÇÃO CONCLUÍDA ---")
    print(f"Features Shape:   {X_feats.shape}")
    print(f"Embeddings Shape: {X_embeds.shape}")
    print(f"Labels Shape:     {y.shape}")
    print(f"Subjects Shape:   {subjects.shape}")
    
    np.savez(FILENAME_NPZ, 
             X_feats=X_feats, 
             X_embeds=X_embeds, 
             y=y, 
             subjects=subjects)
    
    print(f"Ficheiro '{FILENAME_NPZ}' guardado com sucesso.")