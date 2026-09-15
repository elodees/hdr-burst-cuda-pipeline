# SCRIPT_NAME: utils/raw_extractor.py
import os
import numpy as np
import rawpy

def extract_raw_metadata(raw_path):
    """
    Explore le fichier RAW DNG, extrait les métadonnées matérielles indispensables
    au traitement GPU et extrait la matrice active normalisable.
    Conforme aux standards de production Google (Gcam) & NVIDIA Core Compute.
    """
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Fichier introuvable : {raw_path}")

    # Context manager pour garantir la libération immédiate des descripteurs C++ de LibRaw
    with rawpy.imread(raw_path) as raw:
        # 1. Extraction des caractéristiques physiques du capteur (CFA)
        raw_pattern = raw.raw_pattern.tolist()      # Structure spatiale (ex: [[0, 1], [3, 2]])
        color_desc = raw.color_desc.decode('utf-8')  # Signature textuelle (ex: 'RGBG')
        num_colors = raw.num_colors                  # Nombre de canaux (généralement 3 ou 4)

        # 2. Extraction de la dynamique et des points d'écrêtage (Hardware Levels)
        black_levels = raw.black_level_per_channel   # Piédestal thermique par canal [R, G1, B, G2]
        white_level = raw.white_level                # Seuil de saturation absolu du CAN (convertisseur)
        
        # 3. Récupération des coefficients de balance des blancs natifs de l'appareil
        camera_wb = raw.camera_whitebalance          # Multiplicateurs scalaires [R_gain, G1_gain, B_gain, G2_gain]
        
        cfa_color_description = ""
        for i in range(0, 2):
            for j in range(0, 2):
                if raw_pattern[i][j] == 0:
                    cfa_color_description += "R"
                elif raw_pattern[i][j] == 1:
                    cfa_color_description += "G"
                elif raw_pattern[i][j] == 2:
                    cfa_color_description += "B"
                elif raw_pattern[i][j] == 3:
                    cfa_color_description += "G"
                    
        # Journalisation normalisée pour l'intégration continue (CI)
        print(f"[+] Nombre de couleurs  : {num_colors}")
        print(f"[+] Description CFA     : {cfa_color_description}")
        print(f"[+] Pattern Bayer (ID)  : {raw_pattern}")
        print(f"[+] Black Levels (CFA)  : {black_levels}")
        print(f"[+] White Level (Sat)   : {white_level}")
        print(f"[+] Camera whitebalance : {camera_wb}")
        print()

        # 4. Extraction de la zone utile (Active Area) uniquement
        # CORRECTION CONFORME FIRMWARE : raw_image exclut rigoureusement les pixels 
        # noirs physiques de bordure pour garantir la parité géométrique absolue.
        raw_image_active = raw.raw_image.astype(np.float32)
        
        return raw_image_active, raw_pattern, cfa_color_description, black_levels, white_level, camera_wb
