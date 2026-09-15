# SCRIPT_NAME: core/cuda_imaging_pipeline.py
# SEGMENT 1 SUR 2 — CONFIGURATION, BLOC-MATCHING ET WARPING GPU
import numpy as np
import math
import os
from numba import cuda

import utils.Matlab_reader_01 as Matlab_reader_01

@cuda.jit(device=True)
def bilinear_interpolate(img, x, y, w, h):
    """Interpolation bilinéaire sécurisée aux frontières sur le GPU."""
    x0 = int(math.floor(x))
    x1 = x0 + 1
    y0 = int(math.floor(y))
    y1 = y0 + 1
    
    # Contrainte stricte pour éviter les violations d'accès hors-limites (Anti-Crash PCIe)
    x0 = max(0, min(x0, w - 1))
    x1 = max(0, min(x1, w - 1))
    y0 = max(0, min(y0, h - 1))
    y1 = max(0, min(y1, h - 1))
    
    wx = x - math.floor(x)
    wy = y - math.floor(y)
    
    v00 = img[y0, x0]
    v10 = img[y0, x1]
    v01 = img[y1, x0]
    v11 = img[y1, x1]
    
    return (1.0 - wx) * (1.0 - wy) * v00 + wx * (1.0 - wy) * v10 + (1.0 - wx) * wy * v01 + wx * wy * v11

@cuda.jit(fastmath=True)
def block_matching_subpixel_kernel(d_ref, d_src, d_motion_vectors, w, h, search_radius, block_size):
    """Kernel CUDA d'alignement sub-pixel par bloc-matching."""
    block_idx_x = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    block_idx_y = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    
    num_blocks_x = w // block_size
    num_blocks_y = h // block_size
    
    if block_idx_x >= num_blocks_x or block_idx_y >= num_blocks_y:
        return

    best_dx = 0.0
    best_dy = 0.0
    min_sad = 1e10

    # Étape 1 : Recherche grossière au pixel près (Integer Matching)
    for dy_int in range(-search_radius, search_radius + 1):
        for dx_int in range(-search_radius, search_radius + 1):
            sad = 0.0
            for by in range(block_size):
                for bx in range(block_size):
                    ref_x = block_idx_x * block_size + bx
                    ref_y = block_idx_y * block_size + by
                    src_x = ref_x + dx_int
                    src_y = ref_y + dy_int
                    
                    if 0 <= src_x < w and 0 <= src_y < h:
                        sad += abs(d_ref[ref_y, ref_x] - d_src[src_y, src_x])
                    else:
                        sad += 1.0  # Pénalité de débordement géométrique
                        
            if sad < min_sad:
                min_sad = sad
                best_dx = float(dx_int)
                best_dy = float(dy_int)

    # Étape 2 : Raffinement sub-pixel continu (Pas de 0.25 pixel)
    refine_range = 1
    sub_step = 0.25
    fine_dx = best_dx
    fine_dy = best_dy
    
    for s_dy in range(-refine_range, refine_range + 1):
        for s_dx in range(-refine_range, refine_range + 1):
            test_dx = best_dx + s_dx * sub_step
            test_dy = best_dy + s_dy * sub_step
            
            sad = 0.0
            for by in range(block_size):
                for bx in range(block_size):
                    ref_x = block_idx_x * block_size + bx
                    ref_y = block_idx_y * block_size + by
                    src_x = float(ref_x) + test_dx
                    src_y = float(ref_y) + test_dy
                    
                    val_src = bilinear_interpolate(d_src, src_x, src_y, w, h)
                    sad += abs(d_ref[ref_y, ref_x] - val_src)
                    
            if sad < min_sad:
                min_sad = sad
                fine_dx = test_dx
                fine_dy = test_dy

    # Écriture du champ vectoriel résultant
    d_motion_vectors[block_idx_y, block_idx_x, 0] = fine_dx
    d_motion_vectors[block_idx_y, block_idx_x, 1] = fine_dy

@cuda.jit(fastmath=True)
def warp_image_kernel(d_src, d_motion_vectors, d_dst, w, h, block_size):
    """Kernel appliquant le champ de vecteurs pour ré-échantillonner l'image source."""
    x, y = cuda.grid(2)
    if x >= w or y >= h:
        return
        
    block_idx_x = x // block_size
    block_idx_y = y // block_size
    
    dx = d_motion_vectors[block_idx_y, block_idx_x, 0]
    dy = d_motion_vectors[block_idx_y, block_idx_x, 1]
    
    src_x = float(x) + dx
    src_y = float(y) + dy
    
    d_dst[y, x] = bilinear_interpolate(d_src, src_x, src_y, w, h)

def align_burst_frame(ref_img, src_img, tile_size=512, block_size=16, search_radius=8):
    """
    Framework de traitement spatial par tuiles.
    Garantit l'étanchéité absolue de la VRAM pour l'architecture Maxwell.
    """
    if ref_img.dtype != np.float32:
        ref_img = ref_img.astype(np.float32)
    if src_img.dtype != np.float32:
        src_img = src_img.astype(np.float32)

    h, w = ref_img.shape[:2]
    aligned_result = np.zeros_like(src_img, dtype=np.float32)
    
    threads_match = (8, 8)
    threads_warp = (16, 16)

    # Double boucle spatiale de streaming par tuiles immuables
    for y in range(0, h, tile_size):
        for x in range(0, w, tile_size):
            tile_h = min(tile_size, h - y)
            tile_w = min(tile_size, w - x)
            
            tile_ref = np.ascontiguousarray(ref_img[y:y+tile_h, x:x+tile_w])
            tile_src = np.ascontiguousarray(src_img[y:y+tile_h, x:x+tile_w])
            
            d_ref = cuda.to_device(tile_ref)
            d_src = cuda.to_device(tile_src)
            d_dst = cuda.to_device(np.zeros((tile_h, tile_w), dtype=np.float32))
            
            local_bx = tile_w // block_size
            local_by = tile_h // block_size
            d_mv = cuda.to_device(np.zeros((local_by, local_bx, 2), dtype=np.float32))
            
            blocks_match = (
                (local_bx + threads_match[0] - 1) // threads_match[0],
                (local_by + threads_match[1] - 1) // threads_match[1]
            )
            
            blocks_warp = (
                (tile_w + threads_warp[0] - 1) // threads_warp[0],
                (tile_h + threads_warp[1] - 1) // threads_warp[1]
            )
            
            block_matching_subpixel_kernel[blocks_match, threads_match](
                d_ref, d_src, d_mv, tile_w, tile_h, search_radius, block_size
            )
            
            warp_image_kernel[blocks_warp, threads_warp](
                d_src, d_mv, d_dst, tile_w, tile_h, block_size
            )
            
            cuda.synchronize()
            aligned_result[y:y+tile_h, x:x+tile_w] = d_dst.copy_to_host()
            
            del d_ref
            del d_src
            del d_dst
            del d_mv
            
            with cuda.defer_cleanup():
                pass

    cuda.synchronize()
    return aligned_result

# SCRIPT_NAME: core/cuda_imaging_pipeline.py
# SEGMENT 2 SUR 2 — FILTRAGE BILATÉRAL, TONE MAPPING HDR, FUSION ET ORCHESTRATION GLOBAL

@cuda.jit(device=True)
def bilateral_filter_local(img, cx, cy, w, h, spatial_sigma, range_sigma, radius):
    """Filtre bilatéral local exécuté à la volée sur le GPU."""
    num = 0.0
    den = 0.0
    center_val = img[cy, cx]
    
    # Constantes mathématiques pré-calculées
    spatial_factor = -1.0 / (2.0 * spatial_sigma * spatial_sigma)
    range_factor = -1.0 / (2.0 * range_sigma * range_sigma)
    
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            nx = cx + dx
            ny = cy + dy
            
            # Sécurité des frontières de la tuile active
            if 0 <= nx < w and 0 <= ny < h:
                neighbor_val = img[ny, nx]
                
                # Distance spatiale et distance d'intensité
                spatial_dist_sq = float(dx*dx + dy*dy)
                range_dist_sq = (center_val - neighbor_val) ** 2
                
                # Évaluation de la fonction gaussienne bilatérale
                w_spatial = math.exp(spatial_dist_sq * spatial_factor)
                w_range = math.exp(range_dist_sq * range_factor)
                weight = w_spatial * w_range
                
                num += neighbor_val * weight
                den += weight
                
    if den > 1e-5:
        return num / den
    return center_val

@cuda.jit(fastmath=True)
def local_tone_mapping_kernel(d_src, d_dst, w, h, spatial_sigma, range_sigma, radius, compression_factor):
    """
    Kernel CUDA d'extension de dynamique HDR (Local Tone Mapping).
    Sépare et compresse la couche de base tout en ré-injectant les détails fins.
    """
    x, y = cuda.grid(2)
    
    if x >= w or y >= h:
        return
        
    # Éviter le plantage mathématique sur le log avec un epsilon de sécurité
    input_val = max(1e-6, d_src[y, x])
    
    # 1. Passage en espace logarithmique
    log_val = math.log10(input_val)
    
    # 2. Extraction de la couche de base (basses fréquences du contraste)
    base_val = bilateral_filter_local(d_src, x, y, w, h, spatial_sigma, range_sigma, radius)
    log_base = math.log10(max(1e-6, base_val))
    
    # 3. Extraction de la couche de détails (hautes fréquences)
    log_detail = log_val - log_base
    
    # 4. Compression adaptative de la couche de base uniquement
    compressed_log_base = log_base * compression_factor
    
    # 5. Recomposition dans le domaine linéaire (Base compressée + Détails intacts)
    output_val = math.pow(10.0, compressed_log_base + log_detail)
    
    # Contrainte de sécurité pour maintenir le signal entre 0.0 et 1.0
    d_dst[y, x] = max(0.0, min(1.0, output_val))

def hdr_local_tone_mapping(fused_img, tile_size=512, spatial_sigma=3.0, range_sigma=0.1, radius=5, compression_factor=0.4):
    """
    Framework interchangeable d'extension dynamique HDR par tuiles indépendantes.
    Prend la matrice 2D débruitée et renvoie l'image aux ombres débouchées.
    """
    if fused_img.dtype != np.float32:
        fused_img = fused_img.astype(np.float32)
        
    h, w = fused_img.shape[:2]
    hdr_result = np.zeros_like(fused_img, dtype=np.float32)
    
    threadsperblock = (16, 16)

    # Double boucle spatiale immuable de streaming par tuiles strictes
    for y in range(0, h, tile_size):
        for x in range(0, w, tile_size):
            tile_h = min(tile_size, h - y)
            tile_w = min(tile_size, w - x)
            
            bg_x = (tile_w + threadsperblock[0] - 1) // threadsperblock[0]
            bg_y = (tile_h + threadsperblock[1] - 1) // threadsperblock[1]
            local_blocks = (bg_x, bg_y)
            
            tile_src = np.ascontiguousarray(fused_img[y:y+tile_h, x:x+tile_w])
            
            d_src = cuda.to_device(tile_src)
            d_dst = cuda.to_device(np.zeros((tile_h, tile_w), dtype=np.float32))
            
            local_tone_mapping_kernel[local_blocks, threadsperblock](
                d_src, d_dst, tile_w, tile_h, spatial_sigma, range_sigma, radius, compression_factor
            )
            
            cuda.synchronize()
            hdr_result[y:y+tile_h, x:x+tile_w] = d_dst.copy_to_host()
            
            del d_src
            del d_dst
            
            with cuda.defer_cleanup():
                pass

    cuda.synchronize()
    return hdr_result

@cuda.jit(fastmath=True)
def robust_temporal_fusion_kernel(d_ref, d_src, d_accum_num, d_accum_den, w, h, noise_variance, threshold_sigma):
    """
    Kernel CUDA de fusion robuste avec rejet de mouvement (Anti-Ghosting).
    Calcule une pondération adaptative basée sur la variance locale pour éliminer les fantômes.
    """
    x, y = cuda.grid(2)
    
    if x >= w or y >= h:
        return
        
    ref_val = d_ref[y, x]
    src_val = d_src[y, x]
    diff = abs(ref_val - src_val)
    
    sigma_threshold = threshold_sigma * math.sqrt(noise_variance)
    
    if diff < sigma_threshold:
        weight = 1.0 - (diff / sigma_threshold) ** 2
    else:
        weight = 0.0
        
    d_accum_num[y, x] += src_val * weight
    d_accum_den[y, x] += weight

@cuda.jit(fastmath=True)
def resolve_fusion_kernel(d_ref, d_accum_num, d_accum_den, d_dst, w, h):
    """Kernel final de normalisation : produit l'image débruitée."""
    x, y = cuda.grid(2)
    
    if x >= w or y >= h:
        return
        
    den = d_accum_den[y, x]
    
    if den > 1e-4:
        d_dst[y, x] = d_accum_num[y, x] / den
    else:
        d_dst[y, x] = d_ref[y, x]

def robust_temporal_fusion(ref_img, aligned_frames_list, tile_size=512, noise_variance=0.001, threshold_sigma=3.5):
    """
    Framework de fusion temporelle par streaming de tuiles.
    Accumule N images alignées sur une image de référence.
    """
    if ref_img.dtype != np.float32:
        ref_img = ref_img.astype(np.float32)
        
    h, w = ref_img.shape[:2]
    fused_result = np.zeros_like(ref_img, dtype=np.float32)
    
    threadsperblock = (16, 16)

    for y in range(0, h, tile_size):
        for x in range(0, w, tile_size):
            tile_h = min(tile_size, h - y)
            tile_w = min(tile_size, w - x)
            
            bg_x = (tile_w + threadsperblock[0] - 1) // threadsperblock[0]
            bg_y = (tile_h + threadsperblock[1] - 1) // threadsperblock[1]
            local_blocks = (bg_x, bg_y)
            
            tile_ref = np.ascontiguousarray(ref_img[y:y+tile_h, x:x+tile_w])
            
            d_ref = cuda.to_device(tile_ref)
            d_accum_num = cuda.to_device(tile_ref.copy())
            d_accum_den = cuda.to_device(np.ones((tile_h, tile_w), dtype=np.float32))
            d_dst = cuda.to_device(np.zeros((tile_h, tile_w), dtype=np.float32))
            
            for src_frame in aligned_frames_list:
                tile_src = np.ascontiguousarray(src_frame[y:y+tile_h, x:x+tile_w])
                d_src = cuda.to_device(tile_src)
                
                robust_temporal_fusion_kernel[local_blocks, threadsperblock](
                    d_ref, d_src, d_accum_num, d_accum_den, tile_w, tile_h, noise_variance, threshold_sigma
                )
                del d_src
                
            resolve_fusion_kernel[local_blocks, threadsperblock](
                d_ref, d_accum_num, d_accum_den, d_dst, tile_w, tile_h
            )
            
            cuda.synchronize()
            fused_result[y:y+tile_h, x:x+tile_w] = d_dst.copy_to_host()
            
            del d_ref
            del d_accum_num
            del d_accum_den
            del d_dst
            
            with cuda.defer_cleanup():
                pass

    cuda.synchronize()
    return fused_result

def Imaging_pipeline_01(root):
    """Orchestrateur GPU d'alignement sub-pixel et de fusion temporelle anti-ghosting."""
    # 1. Chargement de l'image de référence (Frame 0)
    ref_file_path = os.path.join(root, "payload_N000_sbl.mat")
    ref_img = Matlab_reader_01.matlab_reader_01(ref_file_path, 512)
    
    rafale_alignee = []
    
    # 2. Boucle automatique d'alignement GPU pour les 9 frames esclaves de la rafale Google HDR+
    for i in range(1, 10):
        suffixe = f"{i:03d}"
        chemin_mat = os.path.join(root, f"payload_N{suffixe}_sbl.mat")
        
        if os.path.exists(chemin_mat):
            src_img = Matlab_reader_01.matlab_reader_01(chemin_mat, 512)
            
            # Phase d'estimation géométrique et resampling par blocs
            src_img_alignee = align_burst_frame(ref_img, src_img, tile_size=512, block_size=16, search_radius=8)
            rafale_alignee.append(src_img_alignee)
            
    # 3. Lancement de la fusion statistique robuste et du tone-mapping local
    image_finale_debruitee = robust_temporal_fusion(
        ref_img=ref_img, 
        aligned_frames_list=rafale_alignee, 
        tile_size=512, 
        noise_variance=0.001,
        threshold_sigma=3.5
    )
    
    image_hdr_finale = hdr_local_tone_mapping(
        image_finale_debruitee, 
        tile_size=512, 
        spatial_sigma=3.0, 
        range_sigma=0.1, 
        radius=5, 
        compression_factor=0.4
    )
    
    return image_hdr_finale
