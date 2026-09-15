# SCRIPT_NAME: core/demosaicing.py
import numpy as np
import numba
from numba import cuda
import math

# Configuration matérielle optimisée pour l'architecture Maxwell (GTX 980M)
BLOCK_SIZE = 16
HALO = 1
SHARED_STRIDE = BLOCK_SIZE + 2 * HALO

@cuda.jit
def _demosaicing_bilinear_kernel(d_img_global, d_dst_rgb, width, height, pattern_id):
    """
    Kernel CUDA de dématriçage bilinéaire haute performance.
    Utilise la mémoire partagée (Shared Memory) pour éliminer les latences de textures.
    """
    # Allocation du cache local en SRAM partagée (Syntaxe certifiée JIT Numba)
    shared_raw = cuda.shared.array(shape=(SHARED_STRIDE, SHARED_STRIDE), dtype=numba.float32)

    # Identifiants locaux et globaux
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    bx = cuda.blockIdx.x * cuda.blockDim.x
    by = cuda.blockIdx.y * cuda.blockDim.y

    # Coordonnées globales du pixel géré par ce thread
    global_x = bx + tx
    global_y = by + ty

    # 1. Chargement coopératif synchrone de la tuile et de son halo de garde
    for s_y in range(ty, SHARED_STRIDE, BLOCK_SIZE):
        for s_x in range(tx, SHARED_STRIDE, BLOCK_SIZE):
            img_x = bx - HALO + s_x
            img_y = by - HALO + s_y
            
            # Encapsulation sécurisée aux frontières absolues de la VRAM
            img_x = max(0, min(img_x, width - 1))
            img_y = max(0, min(img_y, height - 1))
            
            shared_raw[s_y, s_x] = d_img_global[img_y, img_x]

    # Barrière matérielle : Synchronisation obligatoire de tous les warps avant calcul
    cuda.syncthreads()

    # Sortie prématurée si le thread est hors des limites réelles de l'image
    if global_x >= width or global_y >= height:
        return

    # Indexation interne de notre zone utile dans la mémoire partagée
    sx = tx + HALO
    sy = ty + HALO

    # Échantillonnage immédiat des stencils du voisinage direct à latence zéro
    v_center = shared_raw[sy, sx]
    
    v_ortho = (shared_raw[sy-1, sx] + shared_raw[sy+1, sx] + 
               shared_raw[sy, sx-1] + shared_raw[sy, sx+1]) * 0.25
               
    v_diag = (shared_raw[sy-1, sx-1] + shared_raw[sy-1, sx+1] + 
              shared_raw[sy+1, sx-1] + shared_raw[sy+1, sx+1]) * 0.25
              
    v_horiz = (shared_raw[sy, sx-1] + shared_raw[sy, sx+1]) * 0.5
    v_verti = (shared_raw[sy-1, sx] + shared_raw[sy+1, sx]) * 0.5

    # 2. Évaluation de la parité spatiale (Modulo 2 ultra-rapide par masque binaire)
    is_even_row = ((global_y & 1) == 0)
    is_even_col = ((global_x & 1) == 0)

    # Résolution des registres de couleurs RVB selon le pattern Bayer du capteur
    r, g, b = 0.0, 0.0, 0.0

    if pattern_id == 0:  # RGGB
        if is_even_row:
            if is_even_col: r, g, b = v_center, v_ortho, v_diag      # R
            else:           r, g, b = v_horiz, v_center, v_verti     # G1
        else:
            if is_even_col: r, g, b = v_verti, v_center, v_horiz     # G2
            else:           r, g, b = v_diag, v_ortho, v_center      # B
            
    elif pattern_id == 1:  # BGGR
        if is_even_row:
            if is_even_col: r, g, b = v_diag, v_ortho, v_center      # B
            else:           r, g, b = v_verti, v_center, v_horiz     # G1
        else:
            if is_even_col: r, g, b = v_horiz, v_center, v_verti     # G2
            else:           r, g, b = v_center, v_ortho, v_diag      # R
            
    elif pattern_id == 2:  # GRBG
        if is_even_row:
            if is_even_col: r, g, b = v_horiz, v_center, v_verti     # G1
            else:           r, g, b = v_center, v_ortho, v_diag      # R
        else:
            if is_even_col: r, g, b = v_diag, v_ortho, v_center      # B
            else:           r, g, b = v_verti, v_center, v_horiz     # G2
            
    elif pattern_id == 3:  # GBRG
        if is_even_row:
            if is_even_col: r, g, b = v_verti, v_center, v_horiz     # G1
            else:           r, g, b = v_diag, v_ortho, v_center      # B
        else:
            if is_even_col: r, g, b = v_center, v_ortho, v_diag      # R
            else:           r, g, b = v_horiz, v_center, v_verti     # G2

    # 3. Écriture coalescée directe dans le tenseur de destination tridimensionnel
    d_dst_rgb[global_y, global_x, 0] = max(0.0, min(1.0, r))
    d_dst_rgb[global_y, global_x, 1] = max(0.0, min(1.0, g))
    d_dst_rgb[global_y, global_x, 2] = max(0.0, min(1.0, b))

def universal_demosaicing_pipeline(img, tile_size=512, pattern_str="RGGB"):
    """
    Orchestrateur Hôte par tuiles spatiales asymétriques.
    Convertit une matrice 2D brute filtrée en image couleur 3D RVB.
    """
    if img.dtype != np.float32:
        img = img.astype(np.float32)

    h, w = img.shape
    result_rgb = np.zeros((h, w, 3), dtype=np.float32)
    
    # Mappage de l'identifiant textuel issu d'ExifTool/rawpy (Alignement RGBG->GRBG standard)
    pattern_map = {"RGGB": 0, "BGGR": 1, "GRBG": 2, "GBRG": 3, "RGBG": 2}
    p_id = pattern_map.get(pattern_str.upper(), 0)

    threads_per_block = (BLOCK_SIZE, BLOCK_SIZE)

    # Double boucle d'exploration spatiale étanche
    for y in range(0, h, tile_size):
        for x in range(0, w, tile_size):
            tile_h = min(tile_size, h - y)
            tile_w = min(tile_size, w - x)

            # Extraction contiguë de la tuile d'origine
            tile_src = np.ascontiguousarray(img[y:y+tile_h, x:x+tile_w])
            
            d_src = cuda.to_device(tile_src)
            d_dst = cuda.to_device(np.zeros((tile_h, tile_w, 3), dtype=np.float32))

            blocks_x = math.ceil(tile_w / BLOCK_SIZE)
            blocks_y = math.ceil(tile_h / BLOCK_SIZE)

            _demosaicing_bilinear_kernel[(blocks_x, blocks_y), threads_per_block](
                d_src, d_dst, tile_w, tile_h, p_id
            )

            cuda.synchronize()
            
            # Injection géométrique directe dans le volume tridimensionnel final
            result_rgb[y:y+tile_h, x:x+tile_w, :] = d_dst.copy_to_host()

            del d_src
            del d_dst
            with cuda.defer_cleanup():
                pass

    cuda.synchronize()
    return result_rgb
