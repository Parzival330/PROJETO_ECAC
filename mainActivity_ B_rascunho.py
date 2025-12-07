import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, f1_score, precision_score, recall_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from scipy.stats import ttest_rel
import sys
import os

# Importar versão real do ReliefF e outras utils
# Certifica-se que consegue importar do diretório atual
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from mainActivity import relieff, calcula_modulos, features_window

FILENAME_NPZ = "datasets_partB.npz"

def load_data():
    try:
        # Tenta carregar do diretório local
        script_dir = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(script_dir, FILENAME_NPZ)
        data = np.load(path)
        return data['X_feats'], data['X_embeds'], data['y'], data['subjects']
    except FileNotFoundError:
        print(f"ERRO: ficherio '{FILENAME_NPZ}' não encontrado. Execute 'mainActivity_B.py' primeiro.")
        exit()

# --- Funções Auxiliares Solicitadas ---

def get_relieff_indices(X, y, n_features_to_keep=15):
    """ Utiliza o algoritmo ReliefF importado de mainActivity. """
    print(f"     -> Calculando ReliefF para {X.shape[1]} features...")
    scores = relieff(X, y, n_neighbors=5, sample_size=500) # Sample size reduzido para performance
    return np.argsort(scores)[::-1][:n_features_to_keep]

def verify_balance(y):
    """
    Analisa o balanceamento das classes e calcula o Coeficiente de Variação (CV).
    """
    unique, counts = np.unique(y, return_counts=True)
    mean_counts = np.mean(counts)
    std_counts = np.std(counts)
    cv = std_counts / mean_counts
    
    print("\n" + "="*50)
    print(" RELATÓRIO DE BALANCEAMENTO (Atividades 1-7)")
    print("="*50)
    
    print(f"{'Atividade':<10} | {'Contagem':<10} | {'%':<10}")
    print("-" * 36)
    for u, c in zip(unique, counts):
        print(f" {int(u):<9} | {c:<10} | {c/len(y)*100:.1f}%")
    print("-" * 36)
    
    print(f"\n Estatísticas:")
    print(f"  -> Média de amostras/classe: {mean_counts:.1f}")
    print(f"  -> Desvio Padrão: {std_counts:.1f}")
    print(f"  -> Coeficiente de Variação (CV): {cv:.3f}")
    
    # Critério de balanceamento
    if cv < 0.2:
        print("\n [CONCLUSÃO] O dataset é equilibrado (CV < 0.2).")
    else:
        print(f"\n [CONCLUSÃO] O dataset NÃO é balanceado (CV={cv:.2f} > 0.2).")
        print("  -> Recomendação: Aplicar SMOTE ou Undersampling.")
    print("="*50 + "\n")

def generate_synthetic_samples(X_class, k_samples):
    """
    Gera K novas amostras sintéticas usando lógica SMOTE (interpolação de vizinhos).
    """
    from sklearn.neighbors import NearestNeighbors
    if len(X_class) < 2: return np.array([])
    
    nbrs = NearestNeighbors(n_neighbors=min(len(X_class), 5)+1).fit(X_class)
    indices = nbrs.kneighbors(X_class, return_distance=False)
    
    synthetic = []
    for _ in range(k_samples):
        # Escolhe um ponto base aleatório
        idx_base = np.random.randint(0, len(X_class))
        # Escolhe um vizinho aleatório (excluindo o próprio, que é o indice 0)
        idx_neighbor = indices[idx_base, np.random.randint(1, indices.shape[1])]
        
        # Interpolação
        base = X_class[idx_base]
        neighbor = X_class[idx_neighbor]
        gap = np.random.rand()
        new_sample = base + gap * (neighbor - base)
        synthetic.append(new_sample)
        
    return np.array(synthetic)

def plot_smote_specific(X, y, subjects):
    """ 
    Visualiza as atividades do Participante 3, destacando os dados sintéticos gerados para a Atividade 4.
    """
    print("\n[SMOTE] Gerando visualização (Participante 3, Atividade 4)...")
    
    # 1. Dados Reais do Participante 3
    mask_p3 = (subjects == 3)
    X_p3 = X[mask_p3]
    y_p3 = y[mask_p3]
    
    # 2. Gerar 3 sintéticos para Act 4 usando apenas dados do P3
    mask_act4 = (y_p3 == 4)
    X_act4 = X_p3[mask_act4]
    
    # Usar a função de geração de amostras
    X_synth = generate_synthetic_samples(X_act4, k_samples=3)

    # 3. Visualização (Side-by-Side: Features 0 vs 1 e Features 2 vs 3)
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # --- Subplot 1: Features 0 vs 1 ---
    ax1 = axes[0]
    for act in np.unique(y_p3):
        mask = (y_p3 == act)
        label_txt = f'Act {act}' + (' (Orig)' if act == 4 else '')
        ax1.scatter(X_p3[mask, 0], X_p3[mask, 1], label=label_txt, alpha=0.5, s=25)
    
    if len(X_synth) > 0:
        ax1.scatter(X_synth[:, 0], X_synth[:, 1], c='black', marker='*', s=350, 
                   label='Sintético (Act 4)', edgecolors='white', zorder=10)
        
    ax1.set_title("SMOTE: Features 0 vs 1")
    ax1.set_xlabel("Feature 0")
    ax1.set_ylabel("Feature 1")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # --- Subplot 2: Features 2 vs 3 (Extra) ---
    ax2 = axes[1]
    for act in np.unique(y_p3):
        mask = (y_p3 == act)
        label_txt = f'Act {act}' + (' (Orig)' if act == 4 else '')
        ax2.scatter(X_p3[mask, 2], X_p3[mask, 3], label=label_txt, alpha=0.5, s=25)
    
    if len(X_synth) > 0:
        ax2.scatter(X_synth[:, 2], X_synth[:, 3], c='black', marker='*', s=350, 
                   label='Sintético (Act 4)', edgecolors='white', zorder=10)

    ax2.set_title("SMOTE: Features 2 vs 3")
    ax2.set_xlabel("Feature 2")
    ax2.set_ylabel("Feature 3")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("smote_participant3_all_acts_enhanced.png")
    print(" -> Gráfico guardado: 'smote_participant3_all_acts_enhanced.png'")
    plt.close()

    # 4. Validação Estatística (Comparação de Médias)
    print("\n [SMOTE VALIDATION] Comparação Estatística (Originais vs Sintéticos):")
    mean_orig = np.mean(X_act4, axis=0)
    mean_synth = np.mean(X_synth, axis=0)
    
    # Distância Euclidiana entre as médias (quanto menor, melhor)
    diff_vector = mean_orig - mean_synth
    dist = np.linalg.norm(diff_vector)
    
    print(f"  Feature | {'Média Original':<15} | {'Média Sintética':<15} | {'Diff':<10}")
    print("-" * 65)
    for i in range(min(5, X.shape[1])): # Mostrar apenas primeiras 5 features
        print(f"  Feat {i}  | {mean_orig[i]:<15.4f} | {mean_synth[i]:<15.4f} | {abs(mean_orig[i]-mean_synth[i]):<10.4f}")
    print("-" * 65)
    print(f"  Similarity Score (Distância entre médias): {dist:.4f} (Menor é melhor)")

# --- Tuning e Avaliação ---

def tune_k(X_train, y_train, X_val, y_val, k_range=range(1, 21)):
    """ Encontra o melhor K para o KNN usando os conjuntos de Treino e Validação. """
    best_k = 1
    best_acc = -1
    
    # Se os sets forem muito grandes, podemos limitar o treino do tuning
    # Mas aqui vamos usar tudo
    
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train) # Treina no Train
        acc = knn.score(X_val, y_val) # Avalia no Val
        if acc > best_acc:
            best_acc = acc
            best_k = k
            
    return best_k, best_acc

def calculate_metrics(y_true, y_pred):
    """ Calcula e retorna métricas de avaliação: Accuracy, F1, Precision, Recall. """
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "F1-Score": f1_score(y_true, y_pred, average='weighted'),
        "Precision": precision_score(y_true, y_pred, average='weighted', zero_division=0),
        "Recall": recall_score(y_true, y_pred, average='weighted', zero_division=0)
    }

def train_validate_test_pipeline(X_train, y_train, X_val, y_val, X_test, y_test, method_name):
    # 1. Normalização (Fit no TRAIN apenas)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)
    
    # 2. Redução de Dimensionalidade / Seleção SE aplicável
    # O pipeline exato depende do método.
    # Vamos tratar disso fora ou passar flags. Aqui assumimos que o X já vem "formatado".
    
    # LOGICA MOVIDA PARA O LOOP PRINCIPAL PARA SER MAIS CLARO
    pass 

# --- Função de Deployment (Ponto 6) ---

def deployment_pipeline(raw_data_256_9, model, scaler, pca_model=None, selected_indices=None):
    """
    Simula o sistema em produção.
    Recebe raw data (256 samples, 9 eixos), e devolve a classe.
    """
    # 1. Feature Extraction
    # Assume que 'raw_data' é uma janela de 5s a 50Hz (aprox. 250 samples)
    # Tem de ter shape (N, 12) para a função features_window.
    # O input desta função é apenas os 9 eixos. Vamos criar um dummy array.
    
    dummy_data = np.zeros((raw_data_256_9.shape[0], 12))
    dummy_data[:, 1:10] = raw_data_256_9 # Preenche Acc, Gyro, Mag
    
    # Extrair features
    # Nota: features_window retorna lista plana.
    f_vec, _ = features_window(dummy_data, fs=50.0) 
    f_vec = f_vec.reshape(1, -1) # (1, 110)
    
    # 2. Normalização
    f_norm = scaler.transform(f_vec)
    
    # 3. Redução (PCA ou ReliefF)
    f_final = f_norm
    if pca_model is not None:
        f_final = pca_model.transform(f_norm)
    elif selected_indices is not None:
        f_final = f_norm[:, selected_indices]
        
    # 4. Classificação
    pred = model.predict(f_final)
    return pred[0]


if __name__ == "__main__":
    X_feats, X_embeds, y, subjects = load_data()
    print(f"Dados Carregados: {len(y)} amostras.")
    
    # 1.1 Balanceamento
    verify_balance(y)
    
    # 1.3 SMOTE Viz
    plot_smote_specific(X_feats, y, subjects)
    
    results_store = {} # Para guardar resultados para testes estatísticos
    
    datasets = {
        "Features": X_feats,
        "Embeddings": X_embeds
    }
    
    strategies = ["Within", "Between"]
    
    print("\n=== INICIANDO PIPELINE DE AVALIAÇÃO (TVT) ===")
    
    for strategy in strategies:
        print(f"\n>> Estratégia: {strategy.upper()}")
        
        # --- 3. Splitting Strategy (60-20-20) ---
        indices = np.arange(len(y))
        
        if strategy == "Within":
            # Split estratificado por sujeito/classe.
            
            # 1. Train (60%) vs Temp (40%)
            X_idx_train, X_idx_temp, y_train_full, y_temp = train_test_split(
                indices, y, test_size=0.4, stratify=y, random_state=42
            )
            # 2. Val (20%) vs Test (20%) -> (50% de 40%)
            X_idx_val, X_idx_test, y_val, y_test = train_test_split(
                X_idx_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
            )
            
        else: # Between
            # 9 Train, 3 Val, 3 Test (Total 15 sujeitos)
            sub_train = range(0, 9)   # 9
            sub_val   = range(9, 12)  # 3
            sub_test  = range(12, 15) # 3
            
            X_idx_train = np.isin(subjects, sub_train)
            X_idx_val   = np.isin(subjects, sub_val)
            X_idx_test  = np.isin(subjects, sub_test)
            
            # Ajuste labels
            y_train_full = y[X_idx_train]
            y_val = y[X_idx_val]
            y_test = y[X_idx_test]

        print(f"   Split Sizes: Train={len(y_train_full)}, Val={len(y_val)}, Test={len(y_test)}")
        
        # Para cada dataset (Feats vs Embeds)
        for data_name, X_full in datasets.items():
            
            X_train_raw = X_full[X_idx_train]
            X_val_raw   = X_full[X_idx_val]
            X_test_raw  = X_full[X_idx_test]
            
            # Processamentos (All, PCA, ReliefF)
            processing_methods = ["All"]
            if data_name == "Features": 
                processing_methods.extend(["PCA", "ReliefF"])
            else:
                # Embeddings já são reduzidos (512 dims), mas PCA ainda pode ser útil.
                processing_methods.extend(["PCA"]) 
            
            for method in processing_methods:
                model_name = f"{strategy}_{data_name}_{method}"
                print(f"   -> Processando: {model_name}...")
                
                # 1. Normalização
                scaler = StandardScaler()
                X_train_s = scaler.fit_transform(X_train_raw)
                X_val_s   = scaler.transform(X_val_raw)
                X_test_s  = scaler.transform(X_test_raw)
                
                # 2. Transformação (PCA / ReliefF)
                X_train_final = X_train_s
                X_val_final   = X_val_s
                X_test_final  = X_test_s
                
                # Guardar objectos para deployment se for o melhor modelo
                pca_model = None
                selected_idxs = None
                
                if method == "PCA":
                    # Manter 90% variância
                    pca = PCA(n_components=0.90)
                    X_train_final = pca.fit_transform(X_train_s)
                    X_val_final   = pca.transform(X_val_s)
                    X_test_final  = pca.transform(X_test_s)
                    pca_model = pca
                    print(f"      PCA Components: {pca.n_components_}")
                    
                elif method == "ReliefF":
                    # Top 15 features
                    top_idx = get_relieff_indices(X_train_s, y_train_full, n_features_to_keep=15)
                    X_train_final = X_train_s[:, top_idx]
                    X_val_final   = X_val_s[:, top_idx]
                    X_test_final  = X_test_s[:, top_idx]
                    selected_idxs = top_idx
                
                # 3. SMOTE (Data Augmentation apenas no treino)
                # "Generate synthetic examples to add variety to the training set"
                smote = SMOTE(random_state=42)
                X_train_bal, y_train_bal = smote.fit_resample(X_train_final, y_train_full)
                
                # 4. Hyperparameter Tuning
                best_k, best_val_score = tune_k(X_train_bal, y_train_bal, X_val_final, y_val)
                print(f"      Melhor K: {best_k} (Val Acc: {best_val_score:.3f})")
                
                # 5. Retrain com Treino + Validação (Usando o melhor K)
                # Combina dados e labels para retreino final
                X_tv = np.vstack([X_train_final, X_val_final])
                y_tv = np.concatenate([y_train_full, y_val])
                
                # Reaplicar SMOTE no conjunto combinado Train+Val
                X_tv_bal, y_tv_bal = smote.fit_resample(X_tv, y_tv)
                
                final_model = KNeighborsClassifier(n_neighbors=best_k)
                final_model.fit(X_tv_bal, y_tv_bal)
                
                
                # 6. Avaliação no Test set
                y_pred = final_model.predict(X_test_final)
                metrics = calculate_metrics(y_test, y_pred)
                
                print(f"      {metrics}")
                
                # Guardar resultados
                results_store[model_name] = {
                    "y_true": y_test,
                    "y_pred": y_pred,
                    "metrics": metrics,
                    "model": final_model,
                    "scaler": scaler,
                    "pca": pca_model,
                    "relief_idx": selected_idxs
                }
                
                # Matriz Confusão se for Features_All (Exemplo)
                if method == "All" and data_name == "Features":
                    cm = confusion_matrix(y_test, y_pred)
                    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
                    disp.plot(cmap='Blues')
                    plt.title(f"CM: {model_name}")
                    plt.savefig(f"cm_{model_name}.png")
                    plt.close()

    print("\n=== COMPARAÇÃO ESTATÍSTICA ===")
    # Exemplo: Comparar Between_Features_All vs Between_Embeddings_All
    
    name_a = "Between_Features_All"
    name_b = "Between_Embeddings_All" # Se existir no loop
    
    if name_a in results_store and name_b in results_store:
        acc_a = results_store[name_a]["metrics"]["Accuracy"]
        acc_b = results_store[name_b]["metrics"]["Accuracy"]
        
        # Teste T emparelhado nos vetores de acerto (aproximação comum)
        hits_a = (results_store[name_a]["y_pred"] == results_store[name_a]["y_true"]).astype(int)
        hits_b = (results_store[name_b]["y_pred"] == results_store[name_b]["y_true"]).astype(int)
        
        # Garantir mesmo tamanho (devem ter, pois é o mesmo Test set para o mesmo Strategy)
        if len(hits_a) == len(hits_b):
            t, p = ttest_rel(hits_a, hits_b)
            print(f"Comparison {name_a} ({acc_a:.1%}) vs {name_b} ({acc_b:.1%})")
            print(f"T-Stat: {t:.4f}, P-Value: {p:.4e}")
            if p < 0.05: print(" Diferença Significativa!")
            else: print(" Diferença NÃO Significativa.")
    
    # --- Simulação de Deployment ---
    print("\n=== DEPLOYMENT TEST (Simulação) ===")
    # Usar 'Between_Features_ReliefF' como exemplo de pipeline completo (se existir)
    model_key = "Between_Features_ReliefF"
    if model_key in results_store:
        pkg = results_store[model_key]
        print(f"Testando deployment com modelo: {model_key}")
        
        # Simular input raw (256 samples, 9 eixos)
        # Vamos pegar num segmento real do dataset original para validar
        # Precisamos de carregar dados Raw. Como não temos aqui fácil, vamos gerar ruído.
        fake_input = np.random.randn(256, 9)
        
        pred = deployment_pipeline(
            fake_input, 
            pkg["model"], 
            pkg["scaler"], 
            pca_model=pkg["pca"], 
            selected_indices=pkg["relief_idx"]
        )
        print(f"Predição para input aleatório: Atividade {pred}")
    else:
        print("Modelo para deployment não encontrado nos resultados.")