import requests
from data_master_eng_ml.utils.twitch_api import fetch_data_with_pagination, build_query

# Definindo a URL base da API
URL_TWITCH_BASE = "https://api.igdb.com/v4"


# Função para obter mapeamentos de uma API com base em um endpoint e campos específicos
def get_mapping_from_api(endpoint: str, fields: list) -> dict:
    """
    Obtém mapeamentos de uma API com base em um endpoint e campos específicos.

    Args:
        endpoint (str): O endpoint da API a ser consultado.
        fields (list): Os campos a serem selecionados na consulta.

    Returns:
        dict: Um dicionário com o mapeamento de IDs para nomes.
    """
    url = f"{URL_TWITCH_BASE}/{endpoint}"
    data_frame = fetch_data_with_pagination(url, build_query, fields)
    return data_frame.set_index("id")["slug"].to_dict()


# Mapeamentos obtidos via API
player_perspectives_mapping = get_mapping_from_api("player_perspectives", ["slug"])
genres_mapping = get_mapping_from_api("genres", ["slug"])
game_modes_mapping = get_mapping_from_api("game_modes", ["slug"])

region_mapping_inverted = {
    1: "europe",
    2: "north_america",
    3: "australia",
    4: "new_zealand",
    5: "japan",
    6: "china",
    7: "asia",
    8: "worldwide",
    9: "korea",
    10: "brazil",
}
plataform_mapping = {
    59: "classic_console",
    66: "classic_console",
    60: "classic_console",
    68: "classic_console",
    67: "classic_console",
    18: "classic_console",
    19: "classic_console",
    29: "classic_console",
    78: "classic_console",
    30: "classic_console",
    32: "classic_console",
    23: "classic_console",
    7: "classic_console",
    8: "classic_console",
    11: "classic_console",
    4: "classic_console",
    21: "classic_console",
    5: "classic_console",
    9: "modern_console",
    48: "modern_console",
    167: "modern_console",
    12: "modern_console",
    49: "modern_console",
    169: "modern_console",
    41: "modern_console",
    130: "modern_console",
    33: "portable_console",
    22: "portable_console",
    24: "portable_console",
    20: "portable_console",
    159: "portable_console",
    37: "portable_console",
    137: "portable_console",
    38: "portable_console",
    46: "portable_console",
    62: "less_common_portable_console",
    61: "less_common_portable_console",
    50: "less_common_portable_console",
    150: "less_common_portable_console",
    136: "less_common_portable_console",
    57: "less_common_portable_console",
    86: "less_common_portable_console",
    80: "less_common_portable_console",
    240: "less_common_portable_console",
    379: "less_common_portable_console",
    309: "less_common_portable_console",
    6: "pc",
    14: "pc",
    3: "pc",
    16: "pc",
    15: "pc",
    63: "pc",
    27: "pc",
    53: "pc",
    26: "pc",
    75: "pc",
    121: "pc",
    34: "mobile",
    39: "mobile",
    405: "mobile",
    74: "mobile",
    73: "mobile",
    72: "mobile",
    417: "mobile",
    385: "vr",
    165: "vr",
    471: "vr",
    164: "vr",
    161: "vr",
    162: "vr",
    384: "vr",
    386: "vr",
    52: "others",
    170: "others",
    113: "others",
    412: "others",
    474: "others",
}
date_format_mapping_inverted = {
    0: "YYYYMMMMDD",
    1: "YYYYMMMM",
    2: "YYYY",
    3: "YYYYQ1",
    4: "YYYYQ2",
    5: "YYYYQ3",
    6: "YYYYQ4",
    7: "TBD",
}

age_rating_mapping = {
    1: "3+",
    2: "6+",
    3: "12+",
    4: "16+",
    5: "18+",
    6: "Rating Pending",
    7: "All Ages",
    8: "All Ages",
    9: "10+",
    10: "12+",
    11: "18+",
    12: "18+",
    13: "All Ages",
    14: "12+",
    15: "16+",
    16: "18+",
    17: "18+",
    18: "All Ages",
    19: "6+",
    20: "12+",
    21: "16+",
    22: "18+",
    23: "All Ages",
    24: "12+",
    25: "15+",
    26: "18+",
    27: "18+",
    28: "All Ages",
    29: "10+",
    30: "12+",
    31: "14+",
    32: "16+",
    33: "18+",
    34: "All Ages",
    35: "6+",
    36: "15+",
    37: "15+",
    38: "18+",
    39: "18+",
}
