import os
import struct
import zlib
import datetime
import numpy as np
from numba import cuda

def RAW_2D_or_3D_export_to_matlab_file(img, filename, tile_size):
    """
    Enregistre un tableau NumPy RAW 2D ou 3D float32 au format binaire MATLAB v5.
    Conforme aux standards de production : sans dépendance externe (pas de scipy.io).
    
    Parameters:
    ----------
    img : np.ndarray
        Image RAW d'entrée, de type float32 (Shape: HxW ou HxWx3).
    filename : str
        Nom du fichier .mat de sortie.
    tile_size : int
        Taille de tuile (conservée pour la signature de l'enchaînement des filtres).
        
    Returns:
    -------
    np.ndarray
        Retourne l'image d'entrée inchangée pour permettre le chaînage d'exécution.
    """
    # 1. Validation de sécurité et conformité des données d'entrée
    if not isinstance(img, np.ndarray):
        raise TypeError("Le paramètre 'img' doit être un conteneur np.ndarray.")
    if img.dtype != np.float32:
        raise TypeError(f"Le type de données doit être float32. Reçu : {img.dtype}")

    # Génération de l'entête standardisé MATLAB 5.0 (128 octets)
    date_str = datetime.datetime.now().strftime("%a %b %d %H:%M:%S %Y")
    header_text = f"MATLAB 5.0 MAT-file, Platform: Python, Created on: {date_str}".encode('ascii')
    header_text = header_text.ljust(116, b' ')
    header = header_text + struct.pack('<Q', 0) + struct.pack('<H', 0x0100) + b'IM'

    # Sous-élément 1 : Array Flags (miUINT32 = 6, Taille = 8 octets, Classe = mxSINGLE_CLASS = 7)
    array_flags = struct.pack('<IIII', 6, 8, 7, 0)

    # Sous-élément 2 : Dimensions (miINT32 = 5)
    # L'ordre d'écriture des dimensions doit refléter la topologie NumPy pour la lecture séquentielle
    shape_dims = img.shape
    num_dims = len(shape_dims)
    dims_bytes = struct.pack(f'<{num_dims}i', *shape_dims)
    dims_len = len(dims_bytes)
    dimensions = struct.pack('<II', 5, dims_len) + dims_bytes + (b'\x00' * ((8 - (dims_len % 8)) % 8))

    # Sous-élément 3 : Variable Name (miINT8 = 1, Nom = 'img')
    var_name = b"img"
    name_len = len(var_name)
    name_element = struct.pack('<II', 1, name_len) + var_name + (b'\x00' * ((8 - (name_len % 8)) % 8))

    # Sous-élément 4 : Matrice de données réelles (miSINGLE = 7)
    # On force l'extraction linéaire en ordre Fortran pour s'aligner sur la lecture native du moteur MATLAB
    flat_data = img.tobytes(order='F')
    data_len = len(flat_data)
    data_element = struct.pack('<II', 7, data_len) + flat_data + (b'\x00' * ((8 - (data_len % 8)) % 8))

    # Bloc de données compressé via l'algorithme DEFLATE (zlib)
    matrix_body = array_flags + dimensions + name_element + data_element
    compressed_body = zlib.compress(matrix_body)
    compressed_element = struct.pack('<II', 15, len(compressed_body)) + compressed_body
    
    # Écriture binaire sur disque avec gestion des répertoires parents
    try:
        dir_name = os.path.dirname(filename)
        if dir_name and not os.path.exists(dir_name):
            os.makedirs(dir_name, exist_ok=True)
        with open(filename, 'wb') as f:
            f.write(header)
            f.write(compressed_element)
    except IOError as e:
        raise IOError(f"Échec de l'écriture physique du fichier .mat : {e}")

    # Synchronisation matérielle de sécurité CUDA
    cuda.synchronize()
    with cuda.defer_cleanup():
        pass

    return img
