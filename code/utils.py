import json
import csv

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
    "swa": "Swahili",
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

def my_load_dataset(path):
  dataset = []
  with open(path, 'r') as f:
    for line in f:
      dataset.append(json.loads(line))
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
  else:
    raise ValueError(f"Unsupported dataset: {path}")
  return ds, name
  
def write_to_file(output_file, ds, predictions, model_name):
  with open(output_file, "w") as out:
    for pred, example in zip(predictions, ds):
      example["prediction"] = float(pred)
      if model_name == "metricX":
        del example["input"]
        del example["input_ids"]
        del example["attention_mask"]
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

  
      
if __name__ == '__main__':      
  dct = build_matching_lang_dict(lang_dict, mm_dict)
  import code; code.interact(local=locals())