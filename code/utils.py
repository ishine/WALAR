import json
import csv
import logging
from typing import List

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoConfig, MistralForCausalLM
from safetensors.torch import load_file
from tqdm import *

lang_dict = {
    "afr": "Afrikaans",
    "amh": "Amharic",
    "ara": "Arabic",
    "hye": "Armenian",
    "asm": "Assamese",
    "ast": "Asturian",
    "azj": "Azerbaijani",
    "bel": "Belarusian",
    "ben": "Bengali",
    "bos": "Bosnian",
    "bul": "Bulgarian",
    "mya": "Burmese",
    "cat": "Catalan",
    "ceb": "Cebuano",
    "zho_simpl": "Chinese",
    "hrv": "Croatian",
    "ces": "Czech",
    "dan": "Danish",
    "ary": "Darija",
    "nld": "Dutch",
    "arz": "Egyptian Arabic",
    "eng": "English",
    "est": "Estonian",
    "tgl": "Filipino",
    "fin": "Finnish",
    "fra": "French",
    "ful": "Fulah",
    "glg": "Galician",
    "lug": "Ganda",
    "kat": "Georgian",
    "deu": "German",
    "ell": "Greek",
    "guj": "Gujarati",
    "hau": "Hausa",
    "heb": "Hebrew",
    "hin": "Hindi",
    "hun": "Hungarian",
    "isl": "Icelandic",
    "ibo": "Igbo",
    "ind": "Indonesian",
    "gle": "Irish",
    "ita": "Italian",
    "jpn": "Japanese",
    "jav": "Javanese",
    "kea": "Kabuverdianu",
    "kam": "Kamba",
    "kan": "Kannada",
    "kaz": "Kazakh",
    "kik": "Kikuyu",
    "khm": "Khmer",
    "kor": "Korean",
    "kir": "Kyrgyz",
    "lao": "Lao",
    "lav": "Latvian",
    "lin": "Lingala",
    "lit": "Lithuanian",
    "luo": "Luo",
    "ltz": "Luxembourgish",
    "mkd": "Macedonian",
    "msa": "Malay",
    "mal": "Malayalam",
    "mlt": "Maltese",
    "mri": "Maori",
    "mar": "Marathi",
    "mon": "Mongolian",
    "nob": "Norwegian",
    "npi": "Nepali",
    "nso": "Northern Sotho",
    "nya": "Nyanja",
    "oci": "Occitan",
    "ory": "Oriya",
    "orm": "Oromo",
    "pus": "Pashto",
    "fas": "Persian",
    "pol": "Polish",
    "por": "Portuguese",
    "pan": "Punjabi",
    "ron": "Romanian",
    "rus": "Russian",
    "srp": "Serbian",
    "sna": "Shona",
    "snd": "Sindhi",
    "slk": "Slovak",
    "slv": "Slovenian",
    "som": "Somali",
    "ckb": "Sorani Kurdish",
    "spa": "Spanish",
    "swh": "Swahili",
    "swe": "Swedish",
    "tgk": "Tajik",
    "tam": "Tamil",
    "tel": "Telugu",
    "tha": "Thai",
    "tur": "Turkish",
    "twi": "Twi",
    "ukr": "Ukrainian",
    "umb": "Umbundu",
    "urd": "Urdu",
    "uzb": "Uzbek",
    "vie": "Vietnamese",
    "cym": "Welsh",
    "wol": "Wolof",
    "xho": "Xhosa",
    "yor": "Yoruba",
    "zul": "Zulu",
}

mm_dict = {
  "af": "Afrikaans",
  "am": "Amharic",
  "ar": "Arabic",
  "hy": "Armenian",
  "as": "Assamese",
  "ast": "Asturian",
  "az": "Azerbaijani",
  "eu": "Basque",
  "be": "Belarusian",
  "bn": "Bengali",
  "bs": "Bosnian",
  "bg": "Bulgarian",
  "my": "Burmese",
  "ca": "Catalan",
  "ceb": "Cebuano",
  "zh": "Chinese",
  "zho": "Chinese",
  "hr": "Croatian",
  "cs": "Czech",
  "da": "Danish",
  "nl": "Dutch",
  "en": "English",
  "et": "Estonian",
  "tl": "Filipino",
  "fi": "Finnish",
  "fr": "French",
  "ff": "Fulah",
  "gl": "Galician",
  "lg": "Ganda",
  "ka": "Georgian",
  "de": "German",
  "el": "Greek",
  "gu": "Gujarati",
  "ha": "Hausa",
  "he": "Hebrew",
  "hi": "Hindi",
  "hu": "Hungarian",
  "is": "Icelandic",
  "ig": "Igbo",
  "id": "Indonesian",
  "ga": "Irish",
  "it": "Italian",
  "ja": "Japanese",
  "jv": "Javanese",
  "kea": "Kabuverdianu",
  "kam": "Kamba",
  "kn": "Kannada",
  "kk": "Kazakh",
  "km": "Khmer",
  "ko": "Korean",
  "ky": "Kyrgyz",
  "lo": "Lao",
  "lv": "Latvian",
  "ln": "Lingala",
  "lt": "Lithuanian",
  "luo": "Luo",
  "lb": "Luxembourgish",
  "mk": "Macedonian",
  "ms": "Malay",
  "ml": "Malayalam",
  "mt": "Maltese",
  "mi": "Maori",
  "mr": "Marathi",
  "mn": "Mongolian",
  "ne": "Nepali",
  "ns": "Northern Sotho",
  "no": "Norwegian",
  "ny": "Nyanja",
  "oc": "Occitan",
  "or": "Oriya",
  "om": "Oromo",
  "ps": "Pashto",
  "fa": "Persian",
  "pl": "Polish",
  "pt": "Portuguese",
  "pa": "Punjabi",
  "ro": "Romanian",
  "ru": "Russian",
  "sr": "Serbian",
  "sn": "Shona",
  "sd": "Sindhi",
  "sk": "Slovak",
  "sl": "Slovenian",
  "so": "Somali",
  "ku": "Sorani Kurdish",
  "es": "Spanish",
  "sw": "Swahili",
  "sv": "Swedish",
  "tg": "Tajik",
  "ta": "Tamil",
  "te": "Telugu",
  "th": "Thai",
  "tr": "Turkish",
  "uk": "Ukrainian",
  "umb": "Umbundu",
  "ur": "Urdu",
  "uz": "Uzbek",
  "vi": "Vietnamese",
  "cy": "Welsh",
  "wo": "Wolof",
  "xh": "Xhosa",
  "yo": "Yoruba",
  "zu": "Zulu",
}

three2two = {'afr': 'af', 'amh': 'am', 'ara': 'ar', 'hye': 'hy', 'asm': 'as', 'ast': 'ast', 'azj': 'az', 'bel': 'be', 'ben': 'bn', 'bos': 'bs', 'bul': 'bg', 'mya': 'my', 'cat': 'ca', 'ceb': 'ceb', 'zho_simpl': 'zho', 'hrv': 'hr', 'ces': 'cs', 'dan': 'da', 'nld': 'nl', 'eng': 'en', 'est': 'et', 'tgl': 'tl', 'fin': 'fi', 'fra': 'fr', 'ful': 'ff', 'glg': 'gl', 'lug': 'lg', 'kat': 'ka', 'deu': 'de', 'ell': 'el', 'guj': 'gu', 'hau': 'ha', 'heb': 'he', 'hin': 'hi', 'hun': 'hu', 'isl': 'is', 'ibo': 'ig', 'ind': 'id', 'gle': 'ga', 'ita': 'it', 'jpn': 'ja', 'jav': 'jv', 'kea': 'kea', 'kam': 'kam', 'kan': 'kn', 'kaz': 'kk', 'khm': 'km', 'kor': 'ko', 'kir': 'ky', 'lao': 'lo', 'lav': 'lv', 'lin': 'ln', 'lit': 'lt', 'luo': 'luo', 'ltz': 'lb', 'mkd': 'mk', 'msa': 'ms', 'mal': 'ml', 'mlt': 'mt', 'mri': 'mi', 'mar': 'mr', 'mon': 'mn', 'nob': 'no', 'npi': 'ne', 'nso': 'ns', 'nya': 'ny', 'oci': 'oc', 'ory': 'or', 'orm': 'om', 'pus': 'ps', 'fas': 'fa', 'pol': 'pl', 'por': 'pt', 'pan': 'pa', 'ron': 'ro', 'rus': 'ru', 'srp': 'sr', 'sna': 'sn', 'snd': 'sd', 'slk': 'sk', 'slv': 'sl', 'som': 'so', 'ckb': 'ku', 'spa': 'es', 'swa': 'sw', 'swe': 'sv', 'tgk': 'tg', 'tam': 'ta', 'tel': 'te', 'tha': 'th', 'tur': 'tr', 'ukr': 'uk', 'umb': 'umb', 'urd': 'ur', 'uzb': 'uz', 'vie': 'vi', 'cym': 'cy', 'wol': 'wo', 'xho': 'xh', 'yor': 'yo', 'zul': 'zu'}

two2three = {'af': 'afr', 'am': 'amh', 'ar': 'ara', 'hy': 'hye', 'as': 'asm', 'ast': 'ast', 'az': 'azj', 'be': 'bel', 'bn': 'ben', 'bs': 'bos', 'bg': 'bul', 'my': 'mya', 'ca': 'cat', 'ceb': 'ceb', 'zho': 'zho_simpl', 'hr': 'hrv', 'cs': 'ces', 'da': 'dan', 'nl': 'nld', 'en': 'eng', 'et': 'est', 'tl': 'tgl', 'fi': 'fin', 'fr': 'fra', 'ff': 'ful', 'gl': 'glg', 'lg': 'lug', 'ka': 'kat', 'de': 'deu', 'el': 'ell', 'gu': 'guj', 'ha': 'hau', 'he': 'heb', 'hi': 'hin', 'hu': 'hun', 'is': 'isl', 'ig': 'ibo', 'id': 'ind', 'ga': 'gle', 'it': 'ita', 'ja': 'jpn', 'jv': 'jav', 'kea': 'kea', 'kam': 'kam', 'kn': 'kan', 'kk': 'kaz', 'km': 'khm', 'ko': 'kor', 'ky': 'kir', 'lo': 'lao', 'lv': 'lav', 'ln': 'lin', 'lt': 'lit', 'luo': 'luo', 'lb': 'ltz', 'mk': 'mkd', 'ms': 'msa', 'ml': 'mal', 'mt': 'mlt', 'mi': 'mri', 'mr': 'mar', 'mn': 'mon', 'ne': 'npi', 'ns': 'nso', 'no': 'nob', 'ny': 'nya', 'oc': 'oci', 'or': 'ory', 'om': 'orm', 'ps': 'pus', 'fa': 'fas', 'pl': 'pol', 'pt': 'por', 'pa': 'pan', 'ro': 'ron', 'ru': 'rus', 'sr': 'srp', 'sn': 'sna', 'sd': 'snd', 'sk': 'slk', 'sl': 'slv', 'so': 'som', 'ku': 'ckb', 'es': 'spa', 'sw': 'swa', 'sv': 'swe', 'tg': 'tgk', 'ta': 'tam', 'te': 'tel', 'th': 'tha', 'tr': 'tur', 'uk': 'ukr', 'umb': 'umb', 'ur': 'urd', 'uz': 'uzb', 'vi': 'vie', 'cy': 'cym', 'wo': 'wol', 'xh': 'xho', 'yo': 'yor', 'zu': 'zul'}

qwen_langs = ["afr", "dan", "nld", "deu", "isl", "ltz", "nob", "swe", "eng", "ast", "cat", "fra", "glg", "oci", "por", "ron", "spa", "bel", "bos", "bul", "hrv", "ces", "mkd", "pol", "rus", "srp", "slk", "slv", "ukr", "asm", "ben", "guj", "hin", "mar", "npi", "ory", "pan", "snd", "urd", "hye", "ell", "gle", "cym", "ita", "lav", "lit", "fas", "tgk", "ceb", "tgl", "ind", "jav", "msa", "kea", "swh", "ara", "mlt", "azj", "kaz", "tur", "uzb", "kan", "mal", "tam", "tel", "mya", "zho_simpl", "zho_trad", "est", "fin", "hun", "kat", "heb", "jpn", "khm", "vie", "kor", "lao", "tha"]

qe_langs = ["afr", "dan", "nld", "deu", "isl", "ltz", "nob", "swe", "cat", "fra", "glg", "por", "ron", "spa", "bel", "bul", "ces", "mkd", "pol", "rus", "srp", "slk", "slv", "ukr", "ben", "guj", "hin", "mar", "npi", "pan", "snd", "urd", "hye", "ell", "gle", "cym", "ita", "lav", "lit", "pus", "fas", "ckb", "tgk", "ceb", "tgl", "ind", "jav", "msa", "mri", "ibo", "nso", "sna", "swh", "umb", "xho", "yor", "zul", "amh", "ara", "mlt", "som", "azj", "kaz", "kir", "tur", "uzb", "kan", "mal", "tam", "tel", "mya", "zho_simpl", "zho_trad", "est", "fin", "hun", "kat", "hau", "heb", "jpn", "khm", "vie", "kor", "lao", "tha", "mon"]

lang_detection_langs = ["af", "als", "am", "an", "ar", "arz", "as", "ast", "av", "az", "azb", "ba", "bar", "bcl", "be", "bg", "bh", "bn", "bo", "bpy", "br", "bs", "bxr", "ca", "cbk", "ce", "ceb", "ckb", "co", "cs", "cv", "cy", "da", "de", "diq", "dsb", "dty", "dv", "el", "eml", "en", "eo", "es", "et", "eu", "fa", "fi", "fr", "frr", "fy", "ga", "gd", "gl", "gn", "gom", "gu", "gv", "he", "hi", "hif", "hr", "hsb", "ht", "hu", "hy", "ia", "id", "ie", "ilo", "io", "is", "it", "ja", "jbo", "jv", "ka", "kk", "km", "kn", "ko", "krc", "ku", "kv", "kw", "ky", "la", "lb", "lez", "li", "lmo", "lo", "lrc", "lt", "lv", "mai", "mg", "mhr", "min", "mk", "ml", "mn", "mr", "mrj", "ms", "mt", "mwl", "my", "myv", "mzn", "nah", "nap", "nds", "ne", "new", "nl", "nn", "no", "oc", "or", "os", "pa", "pam", "pfl", "pl", "pms", "pnb", "ps", "pt", "qu", "rm", "ro", "ru", "rue", "sa", "sah", "sc", "scn", "sco", "sd", "sh", "si", "sk", "sl", "so", "sq", "sr", "su", "sv", "sw", "ta", "te", "tg", "th", "tk", "tl", "tr", "tt", "tyv", "ug", "uk", "ur", "uz", "vec", "vep", "vi", "vls", "vo", "wa", "war", "wuu", "xal", "xmf", "yi", "yo", "yue", "zh"]

training_langs = ["isl", "ltz", "bel", "ces", "mkd", "pol", "srp", "slk", "slv", "ukr", "ben", "guj", "hin", "mar", "npi", "pan", "urd", "hye", "ell", "cym", "lav", "lit", "fas", "ceb", "tgl", "jav", "ara", "azj", "kaz", "tur", "uzb", "kan", "mal", "tam", "tel", "mya", "est", "fin", "hun", "kat", "heb", "khm", "kor", "lao", ]

# we only care about the 
long2lang = {
  "isl_Latn": "Icelandic",
  "ltz_Latn": "Luxembourgish",
  "bel_Cyrl": "Belarusian",
  "ces_Latn": "Czech",
  "mkd_Cyrl": "Macedonian",
  "pol_Latn": "Polish",
  "srp_Cyrl": "Serbian",
  "slk_Latn": "Slovak",
  "slv_Latn": "Slovenian",
  "ukr_Cyrl": "Ukrainian",
  "ben_Beng": "Bengali",
  "guj_Gujr": "Gujarati",
  "hin_Deva": "Hindi",
  "mar_Deva": "Marathi",
  "npi_Deva": "Nepali",
  "pan_Guru": "Punjabi",
  "urd_Arab": "Urdu",
  "hye_Armn": "Armenian",
  "ell_Grek": "Greek",
  "lvs_Latn": "Latvian",
  "lit_Latn": "Lithuanian",
  "fas_Arab": "Persian",
  "cym_Latn": "Welsh",
  "ceb_Latn": "Cebuano",
  "jav_Latn": "Javanese",
  "arb_Arab": "Arabic",
  "azj_Latn": "Azerbaijani",
  "kaz_Cyrl": "Kazakh",
  "tur_Latn": "Turkish",
  "uzn_Latn": "Uzbek",
  "kan_Knda": "Kannada",
  "mal_Mlym": "Malayalam",
  "tam_Taml": "Tamil",
  "tel_Telu": "Telugu",
  "mya_Mymr": "Burmese",
  # "est_Latn": "Estonian",
  "ekk_Latn": "Estonian",
  "fin_Latn": "Finnish",
  "hun_Latn": "Hungarian",
  "kat_Geor": "Georgian",
  "heb_Hebr": "Hebrew",
  "khm_Khmr": "Khmer",
  "kor_Hang": "Korean",
  "lao_Laoo": "Lao",
  "fil_Latn": "Filipino",
}

lang2long = {v: k for k, v in long2lang.items()}

masklid_langs = ['__label__eng_Latn', '__label__arb_Arab', '__label__rus_Cyrl', '__label__por_Latn', '__label__pol_Latn', '__label__ekk_Latn', '__label__ell_Grek', '__label__slk_Latn', '__label__slv_Latn', '__label__nld_Latn', '__label__lvs_Latn', '__label__hun_Latn', '__label__dan_Latn', '__label__swe_Latn', '__label__lit_Latn', '__label__fin_Latn', '__label__mlt_Latn', '__label__cmn_Hani', '__label__nob_Latn', '__label__kor_Hang', '__label__ind_Latn', '__label__uzn_Latn', '__label__fil_Latn', '__label__ukr_Cyrl', '__label__hin_Deva', '__label__hin_Latn', '__label__afr_Latn', '__label__mar_Deva', '__label__ceb_Latn', '__label__ilo_Latn', '__label__zul_Latn', '__label__heb_Hebr', '__label__xho_Latn', '__label__vie_Latn', '__label__jpn_Jpan', '__label__guj_Gujr', '__label__hrv_Latn', '__label__tur_Latn', '__label__nya_Latn', '__label__tsn_Latn', '__label__sna_Latn', '__label__tso_Latn', '__label__tha_Thai', '__label__spa_Latn', '__label__deu_Latn', '__label__eus_Latn', '__label__bul_Cyrl', '__label__amh_Ethi', '__label__fra_Latn', '__label__ewe_Latn', '__label__mkd_Cyrl', '__label__nso_Latn', '__label__tam_Taml', '__label__lin_Latn', '__label__twi_Latn', '__label__yor_Latn', '__label__als_Latn', '__label__ibo_Latn', '__label__ben_Beng', '__label__ita_Latn', '__label__tpi_Latn', '__label__azj_Latn', '__label__run_Latn', '__label__mya_Mymr', '__label__kin_Latn', '__label__ron_Latn', '__label__ces_Latn', '__label__kat_Geor', '__label__urd_Arab', '__label__zsm_Latn', '__label__pap_Latn', '__label__bem_Latn', '__label__mal_Mlym', '__label__kir_Cyrl', '__label__hye_Armn', '__label__smo_Latn', '__label__sin_Sinh', '__label__fij_Latn', '__label__kan_Knda', '__label__pan_Guru', '__label__hau_Latn', '__label__epo_Latn', '__label__gaz_Latn', '__label__tir_Ethi', '__label__bos_Latn', '__label__srp_Cyrl', '__label__hat_Latn', '__label__pag_Latn', '__label__lua_Latn', '__label__war_Latn', '__label__tel_Telu', '__label__tat_Cyrl', '__label__sag_Latn', '__label__lug_Latn', '__label__tum_Latn', '__label__swh_Latn', '__label__umb_Latn', '__label__som_Latn', '__label__gle_Latn', '__label__kng_Latn', '__label__mos_Latn', '__label__lus_Latn', '__label__khk_Cyrl', '__label__asm_Beng', '__label__tuk_Latn', '__label__quy_Latn', '__label__ayr_Latn', '__label__luo_Latn', '__label__tgk_Cyrl', '__label__cat_Latn', '__label__ssw_Latn', '__label__nno_Latn', '__label__cym_Latn', '__label__kik_Latn', '__label__kmb_Latn', '__label__ory_Orya', '__label__bel_Cyrl', '__label__bho_Deva', '__label__apc_Arab', '__label__bak_Cyrl', '__label__jav_Latn', '__label__yue_Hani', '__label__pbt_Arab', '__label__khm_Khmr', '__label__npi_Deva', '__label__npi_Latn', '__label__gug_Latn', '__label__uig_Arab', '__label__fur_Latn', '__label__kbp_Latn', '__label__hne_Deva', '__label__kam_Latn', '__label__gla_Latn', '__label__kab_Latn', '__label__arz_Arab', '__label__kaz_Cyrl', '__label__mri_Latn', '__label__lim_Latn', '__label__srd_Latn', '__label__sun_Latn', '__label__plt_Latn', '__label__mni_Beng', '__label__isl_Latn', '__label__vec_Latn', '__label__glg_Latn', '__label__scn_Latn', '__label__fao_Latn', '__label__san_Deva', '__label__ltz_Latn', '__label__cjk_Latn', '__label__ast_Latn', '__label__lmo_Latn', '__label__szl_Latn', '__label__oci_Latn', '__label__fon_Latn', '__label__min_Latn', '__label__wol_Latn', '__label__lij_Latn', '__label__ajp_Arab', '__label__snd_Arab', '__label__dik_Latn', '__label__ary_Arab', '__label__lao_Laoo', '__label__ars_Arab', '__label__bjn_Latn', '__label__shn_Mymr', '__label__crh_Latn', '__label__aeb_Arab', '__label__ace_Latn', '__label__ckb_Arab', '__label__dyu_Latn', '__label__ltg_Latn', '__label__kmr_Latn', '__label__ban_Latn', '__label__mai_Deva', '__label__fuv_Latn', '__label__kac_Latn', '__label__taq_Latn', '__label__bam_Latn', '__label__sat_Olck', '__label__tzm_Tfng', '__label__bug_Latn', '__label__dzo_Tibt', '__label__kas_Deva', '__label__fas_Arab', '__label__nus_Latn', '__label__knc_Latn', '__label__mag_Deva', '__label__taq_Tfng', '__label__kas_Arab', '__label__knc_Arab', '__label__bjn_Arab', '__label__ace_Arab', '__label__kea_Latn', '__label__awa_Deva', '__label__acm_Arab', '__label__bod_Tibt', '__label__sot_Latn', '__label__ydd_Hebr', '__label__azb_Arab']
# "tgl_Latn", "est_Latn", ara

def my_load_dataset(path):
  dataset = []
  with open(path, 'r') as f:
    for line in f:
      try:
        dataset.append(json.loads(line))
      # in case we're processing flores dataset
      except:
        break
  return dataset


def preprocess_dataset(path):
  # Load with my own function because of potential error when loading low-resource languages
  name = ''
  if 'IndicMT' in path:
    name = 'IndicMT'
    ds = my_load_dataset(path)
    for data in ds:
      data['source'] = data.pop('src')
      data['hypothesis'] = data.pop('translation')
      data['reference'] = data.pop('ref')
  elif 'wmt23-dev' in path:
    name = 'dev23'
    ds = []
    with open(path, 'r', encoding='utf-8') as file:
      reader = csv.reader(file, delimiter='\t')
      next(reader)
      for row in reader:
        # import code; code.interact(local=locals())
        data = {
            'source': row[1],
            'hypothesis': row[2],
            "label": float(row[4]),
        }
        ds.append(data)
  elif 'wmt24-test' in path:
    name = 'test24'
    ds = []
    with open(path, 'r', encoding='utf-8') as file:
      lines = file.readlines()
      for line in lines:
        tmp = json.loads(line)
        data = {
          'source': tmp['src'],
          "hypothesis": tmp['hyp'],
          "label": float(tmp['score']),
        }
        ds.append(data)
  elif 'wmt' in path:
    name = 'wmt'
    with open(path, newline='') as f:
      reader = csv.DictReader(f, delimiter='\t')
      ds = []
      for row in reader:
        row['hypothesis'] = row.pop('target')
        ds.append(row)
  elif 'low-res' in path:
    name = 'low-res'
    with open(path, newline='\n') as f:
      reader = csv.DictReader(f, delimiter=',')
      ds = []
      next(reader)
      for row in reader:
        # import code; code.interact(local=locals())
        data = {
          'source': row['src'],
          'hypothesis': row['mt'],
          "label": float(row['raw_score']),
        }
        ds.append(data)
  elif 'afriMTE' in path:
    name = 'afriMTE'
    ds = my_load_dataset(path)
    for data in ds:
      data['source'] = data.pop('src')
      data['hypothesis'] = data.pop('hypothesis')
      data['reference'] = data.pop('reference')
  elif 'flores' in path:
    name = "flores"
    ds = my_load_dataset(path)
    for data in ds:
      data['source'] = data.pop('src')
      data['hypothesis'] = data.pop('pred')
      data['reference'] = data.pop('ref')
  else:
    raise ValueError(f"Unsupported dataset: {path}")
  return ds, name
  
def write_to_file(output_file, ds, predictions, model_name):
  with open(output_file, "w") as out:
    for pred, example in zip(predictions, ds):
      example["prediction"] = float(pred)
      out.write(json.dumps(example) + "\n")
  

def build_matching_lang_dict(lang_dict, mm_dict):
    # Invert mm_dict: value -> key for reverse lookup
    reversed_mm = {v: k for k, v in lang_dict.items()}
    
    # Result: keys from lang_dict, values from mm_dict
    matched_dict = {}
    
    for key1, value1 in mm_dict.items():
        if value1 in reversed_mm:
            matched_dict[key1] = reversed_mm[value1]
    
    return matched_dict

class RewardModel:
  def __init__(self, model_dir) -> None:
      config = AutoConfig.from_pretrained(model_dir)
      # config._attn_implementation = "flash_attention_2"
      self.device = torch.device('cuda')
      self.model = MistralForCausalLM(config)
      self.model.lm_head = nn.Linear(config.hidden_size, 1, bias=False)
      self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
      state_dict = load_file(f"{model_dir}/model.safetensors")
      self.model.load_state_dict(state_dict, strict=False)
      self.model.to(dtype=torch.bfloat16)
      self.model.to(device=self.device)
      self.model.eval()
      logging.info("Load model completed.")

  @torch.no_grad()
  def score(self, prompts, chosens, batch_size: int = 8) -> List[float]:
      # Pre-tokenize all sequences while preserving original structure
      tokenized_seqs = []
      for prompt, chosen in zip(prompts, chosens):
          prompt_tokens = self.tokenizer.encode(prompt)
          chosen_tokens = self.tokenizer.encode(chosen)
          seq = prompt_tokens + chosen_tokens + [self.tokenizer.eos_token_id]
          tokenized_seqs.append(seq)
      
      scores = []
      num_samples = len(tokenized_seqs)
      
      # Process in batches
      for i in range(0, num_samples, batch_size):
          batch_seqs = tokenized_seqs[i:i+batch_size]
          batch_size_actual = len(batch_seqs)
          
          # Create padded tensor with attention mask
          max_len = max(len(seq) for seq in batch_seqs)
          input_ids = torch.full(
              (batch_size_actual, max_len),
              self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
              dtype=torch.long,
              device=self.device
          )
          attention_mask = torch.zeros_like(input_ids)
          
          # Populate tensors and record last token positions
          last_token_positions = []
          for j, seq in tqdm(enumerate(batch_seqs), total=len(batch_seqs)):
              seq_len = len(seq)
              input_ids[j, :seq_len] = torch.tensor(seq, device=self.device)
              attention_mask[j, :seq_len] = 1
              last_token_positions.append(seq_len - 1)  # Position of last token (our EOS)
          
          # Model forward pass
          outputs = self.model(input_ids, attention_mask=attention_mask)
          logits = outputs.logits.squeeze(-1)  # (batch_size, seq_len)
          
          # Extract scores at last token positions
          batch_scores = logits[torch.arange(batch_size_actual), last_token_positions]
          # import code; code.interact(local=locals())
          scores.extend(batch_scores.tolist())
      
      return scores
      
if __name__ == '__main__':      
  # dct = build_matching_lang_dict(lang_dict, mm_dict)
  target_langs = [lang_dict[lang] for lang in training_langs]
  print(len([language for language in long2lang.values() if language in target_langs]))
  masklid_langs = [lang.replace("__label__", "") for lang in masklid_langs]
  print([lang for lang in long2lang.keys() if lang not in masklid_langs])
        
  # support_langs = []
  # for lang in training_langs:
  #   language = three2two.get(lang, None)
  #   if language is None:
  #     print(language)
  #     raise ValueError(f"Language {lang} not found in mapping.")
  #   if language in lang_detection_langs:
  #     support_langs.append(lang)
    
  import code; code.interact(local=locals())