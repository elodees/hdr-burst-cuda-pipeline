# SCRIPT_NAME: core/subtract_black_level.py
import numpy as np
from numba import cuda
import math

@cuda.jit(fastmath=True)
def _subtract_black_level_and_wb_kernel(d_src, d_dst, width, height, 
                                        b0, b1, b2, b3, 
                                        g0, g1, g2, g3, 
                                        white_level, 
                                        p00, p01, p10, p11,
                                        offset_x, offset_y):
    """
    Kernel CUDA d'élite : Soustraction du niveau de noir, normalisation,
    et application synchrone de la balance des blancs (Sécurisée contre le déphasage).
    """
    # Coordonnées locales à la tuile courante en VRAM
    local_x = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    local_y = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y

    if local_x < width and local_y < height:
        val = d_src[local_y, local_x]

        # 1. Résolution de la parité géométrique par rapport aux coordonnées ABSOLUES du capteur
        global_x = local_x + offset_x
        global_y = local_y + offset_y

        is_even_row = ((global_y & 1) == 0)
        is_even_col = ((global_x & 1) == 0)

        # Extraction de l'index de couleur du motif Bayer pour le pixel absolu courant
        if is_even_row:
            cfa_color = p00 if is_even_col else p01
        else:
            cfa_color = p10 if is_even_col else p11

        # 2. Assignation dynamique des constantes physiques du canal (Niveau de noir et Gain)
        if cfa_color == 0:    # Rouge
            b_level = b0
            w_gain  = g0
        elif cfa_color == 1:  # Vert 1
            b_level = b1
            w_gain  = g1
        elif cfa_color == 2:  # Bleu
            b_level = b2
            w_gain  = g2
        else:                 # Vert 2
            b_level = b3
            w_gain  = g3

        # 3. Soustraction analogique et normalisation mathématique unitaire
        norm_factor = float(white_level) - float(b_level)
        if norm_factor > 0.0:
            calibrated_val = (val - float(b_level)) / norm_factor
        else:
            calibrated_val = 0.0

        # Écrêtage de sécurité dans la plage basse [0.0, 1.0] avant amplification
        calibrated_val = max(0.0, calibrated_val)

        # 4. Application directe du gain de balance des blancs pour aligner les teintes
        output_val = calibrated_val * w_gain

        # Écrêtage final de saturation matérielle à la jauge maximale du convertisseur
        d_dst[local_y, local_x] = min(1.0, output_val)

def subtrack_black_level(img, raw_pattern, black_levels, white_level, camera_wb, tile_size=1024):
    """
    Orchestrateur Hôte par tuiles spatiales asymétriques.
    Calcule le niveau de noir absolu et injecte les gains spectraux de balance des blancs
    pour immuniser les fichiers .mat rechargés sous MATLAB contre toute dérive colorimétrique.
    """
    if img.dtype != np.float32:
        img = img.astype(np.float32)

    h, w = img.shape
    result = np.zeros((h, w), dtype=np.float32)
    threads_per_block = (16, 16)

    # Déballage sécurisé des 4 niveaux de noirs physiques du capteur
    b0, b1, b2, b3 = float(black_levels[0]), float(black_levels[1]), float(black_levels[2]), float(black_levels[3])

    # Extraction et sécurisation d'élite des coefficients EXIF (Élimination du maillage de zéros)
    g_r  = float(camera_wb[0])
    g_g1 = float(camera_wb[1])
    g_b  = float(camera_wb[2])
    g_g2 = float(camera_wb[3])

    # Système de repli automatique (Fallback) si le constructeur du smartphone omet le gain G2
    if g_g2 == 0.0:
        g_g2 = g_g1 if g_g1 != 0.0 else 1.0

    # Déballage de la topologie matricielle du pattern Bayer (ex: [[2, 3], [1, 0]])
    p00, p01 = int(raw_pattern[0][0]), int(raw_pattern[0][1])
    p10, p11 = int(raw_pattern[1][0]), int(raw_pattern[1][1])

    # Double boucle d'exploration spatiale étanche
    for y in range(0, h, tile_size):
        for x in range(0, w, tile_size):
            tile_h = min(tile_size, h - y)
            tile_w = min(tile_size, w - x)

            # Découpage contigu de la sous-région d'analyse en RAM hôte
            tile_src = np.ascontiguousarray(img[y:y+tile_h, x:x+tile_w])
            
            d_src = cuda.to_device(tile_src)
            d_dst = cuda.to_device(np.zeros((tile_h, tile_w), dtype=np.float32))

            blocks_x = math.ceil(tile_w / threads_per_block[0])
            blocks_y = math.ceil(tile_h / threads_per_block[1])

            # Lancement du kernel en injectant les offsets spatiaux absolus (x, y) de la tuile
            _subtract_black_level_and_wb_kernel[ (blocks_x, blocks_y), threads_per_block ](
                d_src, d_dst, tile_w, tile_h,
                b0, b1, b2, b3,
                g_r, g_g1, g_b, g_g2,
                white_level,
                p00, p01, p10, p11,
                x, y
            )

            # Attente de la stabilisation de la VRAM pour cette tuile
            cuda.synchronize()
            
            # Rapatriement et injection géométrique dans la matrice finale de l'hôte
            result[y:y+tile_h, x:x+tile_w] = d_dst.copy_to_host()

            # Purge explicite des descripteurs
            del d_src
            del d_dst
            with cuda.defer_cleanup():
                pass

    cuda.synchronize()
    return result
