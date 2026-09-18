# SCRIPT_NAME: hdr_images_fusion_pipeline.py
import os
import glob
import matplotlib.pyplot as plt

# Ingestion et abstraction d'API conformes au protocole d'ingénierie élite
import utils.raw_extractor as raw_extractor
import utils.RAW_2D_or_3D_export_to_MATLAB_file_01 as writer_raw_2D

import core.subtract_black_level as sbl
from core import remove_dead_hot_pixels

import core.cuda_imaging_pipeline as cuda_imaging_pipeline
import core.demosaicing as dem

def list_hdr_burst_files(data_directory):
    """
    Explore le répertoire cible, liste les fichiers .dng de la rafale,
    et applique séquentiellement les traitements en affichant la matrice texte 
    puis la fenêtre graphique épurée (sans graduations ni gras) pour chaque étape.
    """
    # 1. Capture et tri ordonné des fichiers DNG pour garantir la cohérence temporelle
    search_path = os.path.join(data_directory, "*.dng")
    dng_files = sorted(glob.glob(search_path))
    
    if not dng_files:
        raise FileNotFoundError(f"Aucun fichier .dng trouvé dans le répertoire : {data_directory}")
    
    # Variables de stockage pour conserver les métadonnées de la dernière frame lue
    last_raw_pattern = None
    last_color_desc = None
    last_black_levels = None
    last_white_level = None
    last_camera_wb = None
    
    # 2. Boucle stricte sur chaque image de la rafale (Traitement séquentiel sans état résiduel)
    for idx, dng_path in enumerate(dng_files):
        filename = os.path.basename(dng_path)
        print(f"[DNG {idx+1}/{len(dng_files)}] Traitement du dossier : {data_directory}")
        print()
        
        # Résolution portable et robuste du chemin d'accès au fichier
        full_dng_path = os.path.join(data_directory, filename)
        
        # Extraction des données brutes et des constantes matérielles du capteur via le package utils
        raw_image_visible, raw_pattern, color_desc, black_levels, white_level, camera_wb = raw_extractor.extract_raw_metadata(full_dng_path)
        
        # Sauvegarde des métadonnées structurelles invariantes pour la Phase 2
        last_raw_pattern = raw_pattern
        last_color_desc = color_desc
        last_black_levels = black_levels
        last_white_level = white_level
        last_camera_wb = camera_wb
        
        print("\n[+] Matrice RAW d'origine extraite :")
        print(raw_image_visible)
        print()
        
        # =========================================================================
        # ÉTAPE 1 : SUBTRACT BLACK LEVEL & WHITE BALANCE FUSED
        # =========================================================================
        subtrack_black_level_matrix = sbl.subtrack_black_level(
            img=raw_image_visible, 
            raw_pattern=raw_pattern, 
            black_levels=black_levels, 
            white_level=white_level, 
            camera_wb=camera_wb,
            tile_size=1024
        )
        
        print("[✓] Matrice après Subtract Black Level & white balance (Normalisée [0.0, 1.0]) :")
        print()
        print(subtrack_black_level_matrix)
        print()
        
        plt.figure(num=f"1. Subtract Black Level  & white balance - {filename}", figsize=(7, 7))
        plt.imshow(subtrack_black_level_matrix, cmap='gray', vmin=0.0, vmax=1.0)
        plt.title("1. Subtract Black Level  & white balance (Normalisé [0.0, 1.0])")
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()
        plt.show()  # Bloquant jusqu'à la fermeture de cette fenêtre spécifique
        print()
        
        # =========================================================================
        # ÉTAPE 2 : REMOVE HOT DEAD PIXELS
        # =========================================================================
        clean_pixels_matrix = remove_dead_hot_pixels(
            img=subtrack_black_level_matrix, 
            threshold=float(0.1), 
            tile_size=1024
        )
        
        print("[✓] Matrice après Remove Hot Dead Pixels :")
        print()
        print(clean_pixels_matrix)
        print()
        
        plt.figure(num=f"2. Remove Hot Dead Pixels - {filename}", figsize=(7, 7))
        plt.imshow(clean_pixels_matrix, cmap='gray', vmin=0.0, vmax=1.0)
        plt.title("2. Remove Hot Dead Pixels (Filtrage Impulsionnel)")
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()
        plt.show()  # Bloquant jusqu'à la fermeture de cette fenêtre spécifique
        print()

        # =========================================================================
        # EXPORTATION MATLAB COMPATIBLE AVEC LA NOMENCLATURE DE LA PHASE 2
        # =========================================================================
        # idx de la boucle (0 à 9) est re-formaté sous le motif N000, N001...
        suffixe_hdr = f"N{idx:03d}"
        matlab_output_path = os.path.join(data_directory, f"payload_{suffixe_hdr}_sbl.mat")
        
        print(f"[+] Exportation binaire de la matrice calibrée vers : {os.path.basename(matlab_output_path)}")
        writer_raw_2D.RAW_2D_or_3D_export_to_matlab_file(clean_pixels_matrix, matlab_output_path, 512)
        print()
        
    # CORRECTION : Le retour des métadonnées s'effectue RIGOUREUSEMENT après la fin du traitement des 10 trames
    return last_raw_pattern, last_color_desc, last_black_levels, last_white_level, last_camera_wb
        
def list_mat_burst_files(raw_pattern, color_desc, black_levels, white_level, camera_wb, data_directory):
    """
    Explore le répertoire cible, récupère les fichiers .mat sérialisés de la rafale,
    lance la fusion temporelle, le tone-mapping CUDA et applique le démosaïquage 3D final.
    """
    # Exécution unique du pipeline de fusion lourd (Alignement + Fusion temporelle + Tone mapping)
    hdr_burst = cuda_imaging_pipeline.Imaging_pipeline_01(data_directory)
    
    print("[✓] Matrice après hdr-burst :")
    print()
    print(hdr_burst)
    print()
    
    plt.figure(num="3. hdr-burst-cuda-pipeline", figsize=(7, 7))
    plt.imshow(hdr_burst, cmap='gray', vmin=0.0, vmax=1.0)
    plt.title("3. hdr-burst")
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.show()  # Bloquant jusqu'à la fermeture de cette fenêtre spécifique
    print()
    
    # Conversion 2D débruitée vers tenseur couleur RVB 3D final via le module dématriçage
    Demosaic = dem.universal_demosaicing_pipeline(hdr_burst, tile_size=512, pattern_str=color_desc)
    
    print("[✓] Matrice après Démosaïquage :")
    print()
    print(Demosaic)
    print()
    
    plt.figure(num="4. Démosaïquage", figsize=(7, 7))
    plt.imshow(Demosaic, vmin=0.0, vmax=1.0)
    plt.title("4. Démosaïquage")
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.show()  # Bloquant jusqu'à la fermeture de cette fenêtre spécifique
    print()
    
    plt.imsave("apercu_hdr_fusion.png", Demosaic)
    
if __name__ == "__main__":
    # Définition du chemin d'accès absolu pointant rigoureusement à la racine du projet GitHub
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    BURST_DIR = os.path.join(BASE_DIR, "33TJ_20150722_171315_319")
    
    try:
        # Phase 1 : Génération synchrone des 10 matrices calibrées intermédiaires
        raw_pattern, color_desc, black_levels, white_level, camera_wb = list_hdr_burst_files(data_directory=BURST_DIR)
        
        # Phase 2 : Ingestion séquentielle, recalage sub-pixel, fusion robuste et reconstruction couleur 3D
        list_mat_burst_files(raw_pattern, color_desc, black_levels, white_level, camera_wb, data_directory=BURST_DIR)
    except Exception as e:
        print(f"\n[ERREUR] Échec critique de l'orchestration : {str(e)}")
