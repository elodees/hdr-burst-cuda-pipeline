import numpy as np
from numba import cuda
import math

@cuda.jit
def _dead_hot_pixels_kernel(src, dst, width, height, threshold):
    """
    Kernel CUDA d'élite : Élimination sélective des pixels défectueux (impulsionnels).
    Utilise un stencil 5x5 croisé à pas double pour cibler le même canal Bayer.
    """
    x = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    y = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y

    if x < width and y < height:
        val = src[y, x]
        
        # Sécurité géométrique : Pas de voisinage complet sur les bordures extrêmes de la tuile paddée
        if x < 2 or x >= width - 2 or y < 2 or y >= height - 2:
            dst[y, x] = val
            return

        # Échantillonnage des 4 voisins les plus proches appartenant RIGOUREUSEMENT au même canal CFA
        v0 = src[y - 2, x]
        v1 = src[y + 2, x]
        v2 = src[y, x - 2]
        v3 = src[y, x + 2]

        # Calcul rapide de la médiane des voisins via un réseau de tri câblé en registres
        if v0 > v1: v0, v1 = v1, v0
        if v2 > v3: v2, v3 = v3, v2
        if v0 > v2: v0, v2 = v2, v0
        if v1 > v3: v1, v3 = v3, v1
        median_neighbors = (v1 + v2) * 0.5

        # Évaluation de la déviation absolue du signal local
        deviation = abs(val - median_neighbors)
        
        # Remplacement si le pixel s'écarte anormalement du comportement de son canal
        if deviation > threshold:
            dst[y, x] = median_neighbors
        else:
            dst[y, x] = val

def remove_dead_hot_pixels(img, threshold=0.05, tile_size=1024):
    """
    Orchestrateur GPU par tuiles bidimensionnelles avec gestion du recouvrement spatial.
    Format d'entrée : float32 (2D RAW normalisé)
    Format de sortie : float32 (2D RAW filtré)
    """
    h, w = img.shape
    result = np.zeros((h, w), dtype=np.float32)
    threads_per_block = (16, 16)
    
    # Marge de garde (Padding) pour le stencil 5x5 croisé (pas de 2)
    pad = 2

    for y in range(0, h, tile_size):
        for x in range(0, w, tile_size):
            # Calcul des coordonnées de la tuile source avec débordement contrôlé pour le voisinage
            y_min = max(0, y - pad)
            y_max = min(h, y + tile_size + pad)
            x_min = max(0, x - pad)
            x_max = min(w, x + tile_size + pad)

            tile_src = img[y_min:y_max, x_min:x_max]
            
            d_src = cuda.to_device(np.ascontiguousarray(tile_src))
            d_dst = cuda.to_device(np.zeros(tile_src.shape, dtype=np.float32))

            tile_h_pad, tile_w_pad = tile_src.shape
            blocks_x = math.ceil(tile_w_pad / threads_per_block[0])
            blocks_y = math.ceil(tile_h_pad / threads_per_block[1])

            _dead_hot_pixels_kernel[(blocks_x, blocks_y), threads_per_block](
                d_src, d_dst, tile_w_pad, tile_h_pad, threshold
            )

            # On bloque Python tant que le GPU n'a pas fini CETTE tuile
            cuda.synchronize()
            
            # Rapatriement de la mémoire stable vers l'hôte
            out_tile = d_dst.copy_to_host()
            
            # Calcul rigoureux des dimensions de la zone utile pour cette tuile spécifique
            actual_h = min(tile_size, h - y)
            actual_w = min(tile_size, w - x)
            
            # Détermination exacte de l'origine locale dans la matrice paddée
            local_y_start = pad if y > 0 else 0
            local_x_start = pad if x > 0 else 0

            # Injection parfaite sans décalage spatial sur les bords du capteur
            result[y:y+actual_h, x:x+actual_w] = out_tile[
                local_y_start:local_y_start+actual_h, 
                local_x_start:local_x_start+actual_w
            ]

            # Destruction explicite des pointeurs de tableaux périphériques
            del d_src
            del d_dst

            # On force Numba à libérer la VRAM de la tuile courante
            with cuda.defer_cleanup():
                pass

    # Fermeture propre et vidange définitive du contexte CUDA
    cuda.synchronize()
    return result
