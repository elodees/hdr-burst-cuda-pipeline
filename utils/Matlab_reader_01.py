import os
import struct
import zlib
import numpy as np
from numba import cuda

def matlab_reader_01(filename, tile_size):
    """
    Ouvre un fichier binaire MATLAB v5 (.mat), extrait et reconstruit
    le tableau NumPy RAW 2D ou 3D float32 d'origine.
    Conforme aux standards de production : sans dépendance externe (pas de scipy.io).
    
    Parameters:
    ----------
    filename : str
        Nom du fichier .mat à lire.
    tile_size : int
        Taille de tuile (conservée pour la signature de l'enchaînement des filtres).
        
    Returns:
    -------
    np.ndarray
        Le tableau NumPy float32 (2D ou 3D) restauré avec sa géométrie d'origine.
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Le fichier binaire spécifié est introuvable : {filename}")

    with open(filename, 'rb') as f:
        header = f.read(128)
        if len(header) < 128:
            raise ValueError("Structure d'entête MAT v5 corrompue.")
        
        endian = '<' if header[126:128] == b'IM' else '>'
        
        tag_bytes = f.read(8)
        if len(tag_bytes) < 8:
            raise ValueError("Tag d'encapsulation principal manquant.")
            
        data_type, data_len = struct.unpack(f'{endian}II', tag_bytes)
        if data_type != 15:
            raise ValueError("Le décodeur requiert un flux compressé miCOMPRESSED (Type 15).")
        
        compressed_data = f.read(data_len)
        matrix_body = zlib.decompress(compressed_data)

    offset = 0
    dims = []
    raw_data_bytes = b''
    
    # Parcours séquentiel des sous-éléments de l'objet miMATRIX
    while offset < len(matrix_body):
        if offset + 8 > len(matrix_body):
            break
        sub_type, sub_len = struct.unpack(f'{endian}II', matrix_body[offset:offset+8])
        offset += 8
        
        # Interprétation des conteneurs compressés à données courtes (SDE)
        is_small_element = (sub_type & 0xFFFF0000) != 0
        if is_small_element:
            actual_sub_type = sub_type & 0x0000FFFF
            actual_sub_len = (sub_type & 0xFFFF0000) >> 16
            sub_data = matrix_body[offset-4:offset-4+actual_sub_len]
            sub_type = actual_sub_type
            sub_len = actual_sub_len
        else:
            padding = (8 - (sub_len % 8)) % 8
            sub_data = matrix_body[offset:offset+sub_len]
            offset += sub_len + padding

        if sub_type == 5:  # miINT32 : Extraction de la forme structurelle (Shape)
            num_elements = sub_len // 4
            dims = list(struct.unpack(f'{endian}{num_elements}i', sub_data))
        elif sub_type == 7:  # miSINGLE : Extraction du flux de voxels / pixels
            raw_data_bytes = sub_data

    if not dims or not raw_data_bytes:
        raise ValueError("Erreur critique : Métadonnées ou flux de pixels manquants.")

    # RECONSTRUCTION ABSOLUE :
    # En spécifiant l'ordre 'F' (Fortran) lors de la reconstruction par rapport aux dimensions lues,
    # NumPy va mapper les octets bruts directement selon le stockage initial, rétablissant ainsi
    # la géométrie Row-Major native (3120, 4208) sans aucune dérive ni besoin de transposer manuellement.
    img = np.frombuffer(raw_data_bytes, dtype=f'{endian}f4').reshape(dims, order='F').copy()

    # Vidange et libération de la VRAM CUDA pour sécuriser les futurs filtres
    cuda.synchronize()
    with cuda.defer_cleanup():
        pass
    
    return img
