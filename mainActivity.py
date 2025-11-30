# IMPORTS
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy import stats
import time

# Controlar se os gráficos devem abrir janelas
ENABLE_PLOTS = False

# =======================
# FUNÇÕES DE ANÁLISE E VISUALIZAÇÃO BASE

def calcula_modulos(data_array):
    """Calcula os módulos dos vetores de aceleração, giroscópio e magnetómetro."""
    mod_acc = np.sqrt(data_array[:,1]**2 + data_array[:,2]**2 + data_array[:,3]**2)
    mod_gyro = np.sqrt(data_array[:,4]**2 + data_array[:,5]**2 + data_array[:,6]**2)
    mod_mag = np.sqrt(data_array[:,7]**2 + data_array[:,8]**2 + data_array[:,9]**2)
    atividades = data_array[:,11]
    return mod_acc, mod_gyro, mod_mag, atividades

def load_participantes(participant_id, base_path=r"FORTH_TRACE_DATASET-master"):
    """
    Assunto 0.2: Carrega todos os ficheiros CSV de um participante e devolve um único NumPy array.
    """
    data_list = [] 
    part_folder = os.path.join(base_path, f"part{participant_id}") 
    for device_id in range(1, 6): 
        filename = os.path.join(part_folder, f"part{participant_id}dev{device_id}.csv")
        # Verifica se ficheiro existe antes de carregar
        if os.path.exists(filename):
            data = np.loadtxt(filename, delimiter=",")
            data_list.append(data)
    
    if not data_list:
        return np.empty((0, 12)) # Retorna vazio se não encontrar nada

    data_array = np.vstack(data_list)
    return data_array

# =======================
# ASSUNTO 1: ANÁLISE E TRATAMENTO DE OUTLIERS

# 3.1 - Boxplots
def plot_modulos_boxplot(data_array):
    """
    Assunto 1.1: Calcula os módulos dos vetores e faz boxplots por atividade.
    """
    mod_acc, mod_gyro, mod_mag, atividades = calcula_modulos(data_array)
    atividades_unicas = np.unique(atividades)

    boxplot_data_acc = []
    boxplot_data_gyro = []
    boxplot_data_mag = []

    for atividade in atividades_unicas:
        idx = atividades == atividade
        boxplot_data_acc.append(mod_acc[idx])
        boxplot_data_gyro.append(mod_gyro[idx])
        boxplot_data_mag.append(mod_mag[idx])

    plt.figure(figsize=(15,5))
    atividades_legenda = [str(int(a)) for a in atividades_unicas]

    plt.subplot(1,3,1)
    # CORREÇÃO: tick_labels em vez de labels (para corrigir o warning)
    plt.boxplot(boxplot_data_acc, tick_labels=atividades_legenda)
    plt.title('Aceleração')
    plt.xlabel('Atividade')
    plt.ylabel('Módulo')
    
    plt.subplot(1,3,2)
    plt.boxplot(boxplot_data_gyro, tick_labels=atividades_legenda)
    plt.title('Giroscópio')
    plt.xlabel('Atividade')
    
    plt.subplot(1,3,3)
    plt.boxplot(boxplot_data_mag, tick_labels=atividades_legenda)
    plt.title('Magnetómetro')
    plt.xlabel('Atividade')
    
    plt.tight_layout()
    if ENABLE_PLOTS:
        plt.show()
    else:
        plt.savefig('boxplot_modulos.png')
        plt.close(plt.gcf())

# 3.2 - Densidade Outliers (Pulso Direito)
def densidade_outliers_por_atividade(data_array):
    """
    Assunto 1.2: Analisa a densidade de outliers (IQR) no pulso direito.
    """
    # Filtra apenas sensor pulso direito (ID=2 na coluna 0)
    pulso_direito = data_array[data_array[:,0] == 2] 
    if pulso_direito.size == 0:
        return {'acc': {}, 'gyro': {}, 'mag': {}}

    mod_acc = np.sqrt(pulso_direito[:,1]**2 + pulso_direito[:,2]**2 + pulso_direito[:,3]**2)
    mod_gyro = np.sqrt(pulso_direito[:,4]**2 + pulso_direito[:,5]**2 + pulso_direito[:,6]**2)
    mod_mag = np.sqrt(pulso_direito[:,7]**2 + pulso_direito[:,8]**2 + pulso_direito[:,9]**2)

    atividades = pulso_direito[:,11]
    atividades_unicas = np.unique(atividades)
    densidades = {'acc': {}, 'gyro': {}, 'mag': {}}

    print("\n--- Densidade de outliers por atividade (pulso direito) ---")
    for atividade in atividades_unicas:
        idx = atividades == atividade
        for nome, chave, mod in zip(["Aceleração", "Giroscópio", "Magnetómetro"], ["acc", "gyro", "mag"], [mod_acc, mod_gyro, mod_mag]):
            dados = mod[idx]
            if len(dados) == 0: continue
            
            q1 = np.percentile(dados, 25)
            q3 = np.percentile(dados, 75)
            iqr = q3 - q1
            lim_inf = q1 - 1.5 * iqr
            lim_sup = q3 + 1.5 * iqr
            outliers = (dados < lim_inf) | (dados > lim_sup)
            n0 = np.sum(outliers)
            nr = len(dados)
            densidade = n0 / nr * 100 if nr > 0 else 0
            atividade_int = int(atividade)
            print(f"Atividade {atividade_int} - {nome}: densidade = {densidade:.2f}% ({n0}/{nr})")
            densidades[chave][atividade_int] = densidade
    
    return densidades

# Gráfico de Barras da Densidade de Outliers
def plot_densidade_outliers(densidades):
    if not densidades['acc']: return

    atividades = sorted(densidades['acc'].keys())
    acc_dens = [densidades['acc'][a] for a in atividades]
    gyro_dens = [densidades['gyro'][a] for a in atividades]
    mag_dens = [densidades['mag'][a] for a in atividades]

    x = np.arange(len(atividades)) 
    width = 0.25 

    fig, ax = plt.subplots(figsize=(15, 6))
    ax.bar(x - width, acc_dens, width, label='Aceleração', color='blue')
    ax.bar(x, gyro_dens, width, label='Giroscópio', color='orange')
    ax.bar(x + width, mag_dens, width, label='Magnetómetro', color='green')

    ax.set_ylabel('Densidade de Outliers (%)')
    ax.set_xlabel('Atividade')
    ax.set_title('Densidade de Outliers (IQR) por Atividade no Pulso Direito')
    ax.set_xticks(x)
    ax.set_xticklabels(atividades)
    ax.legend()
    ax.grid(axis='y', linestyle='--')
    plt.tight_layout()
    if not ENABLE_PLOTS:
        plt.savefig('densidade_outliers_barras.png')
        plt.close(fig)

# 3.3 - Identificar outliers Z-Score
def identifica_outliers_zscore(amostras, k=3):
    media = np.mean(amostras)
    std = np.std(amostras)
    if std == 0: return np.zeros(len(amostras), dtype=bool)
    z_scores = (amostras - media) / std
    outliers = np.abs(z_scores) > k
    return outliers

# 3.4 - Plot outliers Z-score
def plot_outliers_modulo(modulo, atividades, nome, k, filename):
    outliers = identifica_outliers_zscore(modulo, k)
    plt.figure(figsize=(10, 6))
    plt.scatter(atividades[~outliers], modulo[~outliers], color='blue', s=5, label='Normal')
    plt.scatter(atividades[outliers], modulo[outliers], color='red', s=5, label='Outlier')
    plt.title(f'{nome} - Deteção de Outliers (Z-Score k={k})')
    plt.xlabel('Atividade')
    plt.ylabel('Módulo')
    plt.legend()
    plt.tight_layout()
    if not ENABLE_PLOTS:
        plt.savefig(filename)
        plt.close()

# 3.5 - Histograma Comparativo de Limites
def plot_limites_outliers_comparativo(modulo, atividade_label, nome_sensor, atividade_id=4):
    idx = atividade_label == atividade_id
    dados = modulo[idx]
    if len(dados) < 10: return

    # 1. Limites IQR
    q1 = np.percentile(dados, 25)
    q3 = np.percentile(dados, 75)
    iqr = q3 - q1
    lim_iqr_sup = q3 + 1.5 * iqr
    lim_iqr_inf = q1 - 1.5 * iqr

    # 2. Limites Z-Score
    media = np.mean(dados)
    std = np.std(dados)
    lim_z_sup_3 = media + 3 * std
    lim_z_sup_4 = media + 4 * std
    lim_z_inf_3 = media - 3 * std
    lim_z_inf_4 = media - 4 * std

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(dados, bins=50, density=True, alpha=0.6, color='gray', label='Distribuição dos Dados')
    
    ax.axvline(lim_iqr_sup, color='green', linestyle='-', linewidth=2, label='IQR Superior (+1.5 IQR)')
    ax.axvline(lim_z_sup_3, color='red', linestyle='--', linewidth=1.5, label='Z-Score Superior (k=3)')
    ax.axvline(lim_z_sup_4, color='orange', linestyle=':', linewidth=1.5, label='Z-Score Superior (k=4)')
    ax.axvline(lim_iqr_inf, color='green', linestyle='-', linewidth=2)
    ax.axvline(lim_z_inf_3, color='red', linestyle='--', linewidth=1.5)
    ax.axvline(lim_z_inf_4, color='orange', linestyle=':', linewidth=1.5)

    ax.set_title(f'Comparação de Limites (IQR vs Z-Score) - Ativ. {atividade_id} - {nome_sensor}')
    ax.set_xlabel(f'Módulo {nome_sensor}')
    ax.set_ylabel('Densidade')
    ax.legend()
    plt.tight_layout()
    if not ENABLE_PLOTS:
        plt.savefig(f'comparativo_limites_outliers_act{atividade_id}_{nome_sensor}.png')
        plt.close(fig)

# 3.6 - K-Means
def kmeans_clusters(data, n_clusters, max_iter=100, tol=1e-4):
    rng = np.random.default_rng()
    # Segurança para caso haja menos dados que clusters
    if data.shape[0] < n_clusters:
        return data, np.zeros(data.shape[0])
        
    indices = rng.choice(data.shape[0], n_clusters, replace=False)
    centroids = data[indices]
    labels = np.zeros(data.shape[0])
    
    for _ in range(max_iter):
        dists = np.linalg.norm(data[:, None] - centroids[None, :], axis=2)
        labels = np.argmin(dists, axis=1)
        new_centroids = np.array([data[labels == k].mean(axis=0) if np.any(labels == k) else centroids[k] for k in range(n_clusters)])
        if np.all(np.linalg.norm(new_centroids - centroids, axis=1) < tol):
            break
        centroids = new_centroids
    return centroids, labels

# 3.7 - 3D Plot
def plot_3d_outliers(mod_acc, mod_gyro, mod_mag, outliers, titulo, filename):
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')
    # Plotar com downsampling para não ficar muito pesado se houver muitos pontos
    step = 10 if len(mod_acc) > 50000 else 1
    
    ax.scatter(mod_acc[~outliers][::step], mod_gyro[~outliers][::step], mod_mag[~outliers][::step], c='blue', s=2, label='Normal')
    ax.scatter(mod_acc[outliers][::step], mod_gyro[outliers][::step], mod_mag[outliers][::step], c='red', s=5, label='Outlier')
    
    ax.set_xlabel('Módulo Acc')
    ax.set_ylabel('Módulo Gyro')
    ax.set_zlabel('Módulo Mag')
    ax.set_title(titulo)
    ax.legend()
    if not ENABLE_PLOTS:
        plt.savefig(filename)
        plt.close(fig)

def outliers_kmeans(mod_acc, mod_gyro, mod_mag, n_clusters=3, frac_outlier=0.05):
    X = np.column_stack([mod_acc, mod_gyro, mod_mag])
    centroids, labels = kmeans_clusters(X, n_clusters)
    dists = np.linalg.norm(X - centroids[labels], axis=1)
    outliers = np.zeros(X.shape[0], dtype=bool)
    for k in range(n_clusters):
        idx = np.where(labels == k)[0]
        if len(idx) == 0: continue
        dists_cluster = dists[idx]
        n_out = max(1, int(frac_outlier * len(idx)))
        out_idx = idx[np.argsort(dists_cluster)[-n_out:]]
        outliers[out_idx] = True
    return outliers

# =======================
# ASSUNTO 2: ESTATÍSTICA E FEATURES

def normality_per_activity(modulo, atividades):
    results = {}
    atividades_unicas = np.unique(atividades)
    for atividade in atividades_unicas:
        idx = atividades == atividade
        sample = modulo[idx]
        if sample.size < 5 or np.std(sample) == 0:
            results[int(atividade)] = (np.nan, np.nan)
            continue
        # KS Test vs Normal (após standardização)
        mu = np.mean(sample)
        sigma = np.std(sample, ddof=1)
        z = (sample - mu) / sigma
        stat, p = stats.kstest(z, 'norm')
        results[int(atividade)] = (stat, p)
    return results

def compare_means_across_activities(modulo, atividades):
    atividades_unicas = np.unique(atividades)
    groups = [modulo[atividades == a] for a in atividades_unicas if len(modulo[atividades == a]) >= 3]
    
    if len(groups) < 2:
        return 'N/A', np.nan, np.nan

    # Usar Kruskal-Wallis (pois os dados não são normais)
    stat, p = stats.kruskal(*groups)
    return 'Kruskal-Wallis', stat, p

def run_point_4_1(all_data):
    mod_acc, mod_gyro, mod_mag, atividades = calcula_modulos(all_data)
    for name, modulo in [('Aceleração', mod_acc), ('Giroscópio', mod_gyro), ('Magnetómetro', mod_mag)]:
        print(f"\n=== 4.1 {name} ===")
        norm_results = normality_per_activity(modulo, atividades)
        print("Normalidade por atividade (KS p>=0.05 ~ normal):")
        for atividade, (stat, p) in sorted(norm_results.items()):
            status = 'normal' if p is not None and p >= 0.05 else 'não normal'
            print(f"Atividade {atividade}: KS stat={stat:.3f} p={p:.3g} => {status}")
        method, stat, p = compare_means_across_activities(modulo, atividades)
        print(f"Teste de médias ({method}): stat={stat:.3f}, p={p:.3g}")

# --- FEATURES ---
def window_indices(n_samples, window_size, step):
    start = 0
    while start + window_size <= n_samples:
        yield start, start + window_size
        start += step

def temporal_features_axis(x):
    x = np.asarray(x)
    feats = {}
    feats['mean'] = float(np.mean(x))
    feats['std'] = float(np.std(x, ddof=1)) if len(x) > 1 else 0.0
    feats['median'] = float(np.median(x))
    feats['min'] = float(np.min(x))
    feats['max'] = float(np.max(x))
    feats['rms'] = float(np.sqrt(np.mean(x * x)))
    q1, q3 = np.percentile(x, [25, 75])
    feats['iqr'] = float(q3 - q1)
    
    # Zero Crossing Rate simplificado
    zc = np.sum(np.abs(np.diff(np.sign(x))) > 0)
    feats['zcr'] = zc / max(1, (len(x) - 1))
    
    # Mean Crossing Rate
    mc = np.sum(np.abs(np.diff(np.sign(x - feats['mean']))) > 0)
    feats['mcr'] = mc / max(1, (len(x) - 1))
    return feats

def spectral_features_axis(x, fs):
    x = np.asarray(x)
    X = np.fft.rfft(x * np.hanning(len(x)))
    psd = (np.abs(X) ** 2) / len(x)
    psd_sum = np.sum(psd) + 1e-12
    freqs = np.fft.rfftfreq(len(x), d=1.0 / fs)

    energy = float(psd_sum)
    p = psd / psd_sum
    spectral_entropy = float(-(p * (np.log(p + 1e-12))).sum())
    k_max = int(np.argmax(psd))
    peak_amp = float(psd[k_max])

    return {
        'spec_energy': energy,
        'spec_entropy': spectral_entropy,
        'spec_peak_amp': peak_amp,
    }

def features_window(data_window, fs):
    ax, ay, az = data_window[:,1], data_window[:,2], data_window[:,3]
    gx, gy, gz = data_window[:,4], data_window[:,5], data_window[:,6]
    mx, my, mz = data_window[:,7], data_window[:,8], data_window[:,9]

    feats = []
    names = []

    def add_axis_feats(prefix, x):
        t = temporal_features_axis(x)
        s = spectral_features_axis(x, fs)
        for k, v in t.items():
            feats.append(v)
            names.append(f"{prefix}_{k}")
        for k, v in s.items():
            feats.append(v)
            names.append(f"{prefix}_{k}")

    add_axis_feats('acc_x', ax)
    add_axis_feats('acc_y', ay)
    add_axis_feats('acc_z', az)
    add_axis_feats('gyro_x', gx)
    add_axis_feats('gyro_y', gy)
    add_axis_feats('gyro_z', gz)
    add_axis_feats('mag_x', mx)
    add_axis_feats('mag_y', my)
    add_axis_feats('mag_z', mz)
    
    # SMA
    sma_a = float((np.sum(np.abs(ax)) + np.sum(np.abs(ay)) + np.sum(np.abs(az))) / len(ax))
    feats.append(sma_a); names.append('acc_sma')
    
    sma_g = float((np.sum(np.abs(gx)) + np.sum(np.abs(gy)) + np.sum(np.abs(gz))) / len(gx))
    feats.append(sma_g); names.append('gyro_sma')

    return np.array(feats, dtype=float), names

def window_majority_label(data_window):
    acts = data_window[:, 11].astype(int)
    if acts.size == 0: return -1
    vals, counts = np.unique(acts, return_counts=True)
    return int(vals[np.argmax(counts)])

def extract_features_and_labels_over_time(all_data, fs=50.0, window_sec=2.0, step_sec=0.5, max_windows=None, progress_every=1000):
    n = all_data.shape[0]
    win = int(round(window_sec * fs))
    step = int(round(step_sec * fs))
    feature_list = []
    labels = []
    names = None
    num_windows = 0
    
    for i0, i1 in window_indices(n, win, step):
        dw = all_data[i0:i1, :]
        fvec, fnames = features_window(dw, fs)
        feature_list.append(fvec)
        labels.append(window_majority_label(dw))
        num_windows += 1
        if names is None: names = fnames
        
        if progress_every and (num_windows % progress_every == 0):
            print(f"[4.x] Progresso: {num_windows} janelas processadas...")
        if max_windows is not None and num_windows >= max_windows:
            print(f"[4.x] Limite de janelas atingido ({max_windows}).")
            break
            
    if not feature_list: return np.empty((0, 0)), [], np.array([])
    return np.vstack(feature_list), names, np.array(labels, dtype=int)

# =======================
# ASSUNTO 3: REDUÇÃO E SELEÇÃO

def standardize_features(F):
    mean = np.mean(F, axis=0)
    std = np.std(F, axis=0, ddof=1)
    std_safe = np.where(std == 0, 1.0, std)
    Fz = (F - mean) / std_safe
    return Fz, mean, std_safe

def pca_fit(Fz):
    N, D = Fz.shape
    # CORREÇÃO: np.maximum(0, ...) para evitar NaN em sqrt de erros de float negativos
    if N >= D:
        C = (Fz.T @ Fz) / max(1, (N - 1))
        evals, evecs = np.linalg.eigh(C)
        order = np.argsort(evals)[::-1]
        evals = evals[order]
        evecs = evecs[:, order]
        components = evecs.T
        explained_var = evals
        var_ratio = explained_var / np.sum(explained_var)
        S = np.sqrt(np.maximum(0, explained_var) * max(1, (N - 1))) 
        return components, S, var_ratio
    else:
        U, S, Vt = np.linalg.svd(Fz, full_matrices=False)
        components = Vt
        explained_var = (S ** 2) / max(1, (N - 1))
        var_ratio = explained_var / np.sum(explained_var)
        return components, S, var_ratio

def pca_transform(Fz, components, n_components):
    W = components[:n_components, :]
    return Fz @ W.T

def run_point_4_3(all_data, fs=50.0, max_windows=None):
    print(f"[4.3] A extrair features para PCA...")
    F, names, labels = extract_features_and_labels_over_time(all_data, fs=fs, max_windows=max_windows, progress_every=500)
    
    if F.size == 0: return None

    Fz, mean, std = standardize_features(F)
    components, S, var_ratio = pca_fit(Fz)
    
    cum = np.cumsum(var_ratio)
    dims_75 = np.argmax(cum >= 0.75) + 1
    
    print("[4.4] Variância explicada (Top 5):")
    for i in range(5):
        print(f"  PC{i+1}: {var_ratio[i]:.4f}, acumulada: {cum[i]:.4f}")
    print(f"[4.4] Dimensões para explicar 75%: {dims_75}")
    
    # Projeção
    F_proj = pca_transform(Fz, components, n_components=2)
    
    return F_proj, labels, var_ratio

def plot_pca_analise(var_ratio, F_proj, labels):
    # 1. Scree Plot
    cum_var = np.cumsum(var_ratio)
    k_75 = np.argmax(cum_var >= 0.75) + 1
    
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(range(1, len(cum_var)+1), cum_var, marker='o', linestyle='-', markersize=4)
    ax1.axhline(y=0.75, color='r', linestyle='--', label='Limiar 75%')
    ax1.axvline(x=k_75, color='g', linestyle='--', label=f'{k_75} Comps')
    ax1.set_title('PCA Scree Plot')
    ax1.legend()
    ax1.grid(True)
    if not ENABLE_PLOTS:
        plt.savefig('pca_scree_plot.png')
        plt.close(fig1)

    # 2. Scatter 2D
    fig2, ax2 = plt.subplots(figsize=(10, 8))
    scatter = ax2.scatter(F_proj[:, 0], F_proj[:, 1], c=labels, cmap='jet', alpha=0.6, s=15)
    ax2.set_title('Projeção PCA 2D (PC1 vs PC2)')
    plt.colorbar(scatter, ax=ax2, label='Atividade')
    if not ENABLE_PLOTS:
        plt.savefig('pca_2d_scatter.png')
        plt.close(fig2)

def fisher_score(F, y):
    y = np.asarray(y)
    F = np.asarray(F)
    N, D = F.shape
    scores = np.zeros(D, dtype=float)
    mu = np.mean(F, axis=0)
    classes = np.unique(y)
    for d in range(D):
        num, den = 0.0, 0.0
        for c in classes:
            idx = (y == c)
            if not np.any(idx): continue
            Fc = F[idx, d]
            varc = np.var(Fc, ddof=0)
            muc = np.mean(Fc)
            nc = Fc.shape[0]
            num += nc * (muc - mu[d]) ** 2
            den += nc * varc
        scores[d] = num / (den + 1e-12)
    return scores

def relieff(F, y, n_neighbors=5, sample_size=500):
    rng = np.random.default_rng()
    F = np.asarray(F)
    y = np.asarray(y)
    N, D = F.shape
    # Standardize rápido
    Fz = (F - np.mean(F, axis=0)) / (np.std(F, axis=0) + 1e-12)
    
    m = min(sample_size, N)
    idx_samples = rng.choice(N, size=m, replace=False)
    classes = np.unique(y)
    W = np.zeros(D, dtype=float)
    
    for i in idx_samples:
        xi = Fz[i]
        yi = y[i]
        diff = Fz - xi
        dists = np.sqrt(np.sum(diff**2, axis=1))
        dists[i] = np.inf
        
        # Hit (mesma classe)
        hit_mask = (y == yi)
        hit_mask[i] = False
        hit_idx = np.where(hit_mask)[0]
        if len(hit_idx) > 0:
            nn_hit = hit_idx[np.argsort(dists[hit_idx])[:n_neighbors]]
            W -= np.mean(np.abs(Fz[nn_hit] - xi), axis=0) / m
            
        # Miss (outras classes)
        for cj in classes:
            if cj == yi: continue
            miss_idx = np.where(y == cj)[0]
            if len(miss_idx) > 0:
                nn_miss = miss_idx[np.argsort(dists[miss_idx])[:n_neighbors]]
                prob = len(miss_idx) / N
                W += prob * np.mean(np.abs(Fz[nn_miss] - xi), axis=0) / m
    return W

def run_point_4_5_6(all_data, fs=50.0, max_windows=None):
    print(f"[4.5] A calcular Rankings (Fisher e ReliefF)...")
    F, names, y = extract_features_and_labels_over_time(all_data, fs=fs, max_windows=max_windows, progress_every=0)
    
    fisher = fisher_score(F, y)
    relief = relieff(F, y) # usando defaults simplificados
    
    top_k = 10
    idx_fisher = np.argsort(fisher)[::-1][:top_k]
    idx_relief = np.argsort(relief)[::-1][:top_k]
    
    print("\nTop 10 Fisher:")
    for i in idx_fisher: print(f" - {names[i]} ({fisher[i]:.2f})")
    
    print("\nTop 10 ReliefF:")
    for i in idx_relief: print(f" - {names[i]} ({relief[i]:.4f})")
    
    # Gráfico de Barras Comparativo (SIMPLIFICADO: SEM MATRIZ DE CORRELAÇÃO)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    y_pos = np.arange(top_k)
    ax1.barh(y_pos, fisher[idx_fisher], align='center', color='skyblue')
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels([names[i] for i in idx_fisher])
    ax1.invert_yaxis()
    ax1.set_title(f'Top {top_k} Features - Fisher Score')
    
    ax2.barh(y_pos, relief[idx_relief], align='center', color='lightgreen')
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([names[i] for i in idx_relief])
    ax2.invert_yaxis()
    ax2.set_title(f'Top {top_k} Features - ReliefF')
    
    plt.tight_layout()
    if not ENABLE_PLOTS:
        plt.savefig('feature_ranking_bars.png')
        plt.close(fig)

# =======================
# EXECUÇÃO PRINCIPAL
# =======================
if __name__ == "__main__":
    
    # 0. Carregar Dados
    DATASET_BASE_PATH = r"FORTH_TRACE_DATASET-master"
    print(f"A carregar dados...")
    all_data_list = []
    for pid in range(0, 15): # 1 a 15 (o range para em 16)
        d = load_participantes(pid, base_path=DATASET_BASE_PATH)
        if d.size > 0: all_data_list.append(d)
    
    if not all_data_list:
        print("ERRO: Nenhum dado carregado. Verifica o caminho da pasta.")
        exit()
        
    all_data = np.vstack(all_data_list)
    mod_acc, mod_gyro, mod_mag, atividades = calcula_modulos(all_data)
    
    print("\n=== BLOCO 1: OUTLIERS ===")
    plot_modulos_boxplot(all_data)
    densidade_outliers_por_atividade(all_data)
    plot_densidade_outliers(densidade_outliers_por_atividade(all_data))
    
    # 3.4 e 3.5: Z-Scores e Comparativos (CUMPRINDO O ENUNCIADO)
    # Gera gráficos para k=3, 3.5 e 4
    for k_val in [3, 3.5, 4]:
        print(f"Gerando plot Z-Score para k={k_val}...")
        plot_outliers_modulo(mod_acc, atividades, 'Aceleração', k_val, f'outliers_zscore_acc_k{k_val}.png')
    
    plot_limites_outliers_comparativo(mod_acc, atividades, 'Aceleração', atividade_id=4)
    
    # 3.7: 3D Plot K-Means (Exemplo)
    outliers_km = outliers_kmeans(mod_acc, mod_gyro, mod_mag, n_clusters=3)
    plot_3d_outliers(mod_acc, mod_gyro, mod_mag, outliers_km, 'Outliers K-Means (3 Clusters)', 'outliers_kmeans_3d.png')

    print("\n=== BLOCO 2: ESTATÍSTICA ===")
    run_point_4_1(all_data)
    # NOTA: O gráfico de Violino foi REMOVIDO por ser desnecessário dada a clareza dos testes estatísticos.

    print("\n=== BLOCO 3: REDUÇÃO E SELEÇÃO ===")
    # Usar max_windows para ser mais rápido nos testes
    pca_res = run_point_4_3(all_data, fs=50.0, max_windows=2000)
    if pca_res:
        F_proj, labels, var_ratio = pca_res
        plot_pca_analise(var_ratio, F_proj, labels)
        
    run_point_4_5_6(all_data, fs=50.0, max_windows=2000)
    
    print("\nFIM! Verifica os ficheiros PNG gerados na pasta.")