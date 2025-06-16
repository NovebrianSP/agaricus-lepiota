import pandas as pd

# Nama file input dan output
input_file = 'agaricus-lepiota.data'
output_file = 'agaricus-lepiota-mapped.csv'

# Header kolom sesuai dokumentasi UCI
columns = [
    'class','cap-shape','cap-surface','cap-color','bruises','odor',
    'gill-attachment','gill-spacing','gill-size','gill-color',
    'stalk-shape','stalk-root','stalk-surface-above-ring','stalk-surface-below-ring',
    'stalk-color-above-ring','stalk-color-below-ring','veil-type','veil-color',
    'ring-number','ring-type','spore-print-color','population','habitat'
]

# Mapping dictionary untuk setiap kolom
mapping_dict = {
    'class': {'e': 'bisa dimakan', 'p': 'beracun'},
    'cap-shape': {
        'b': 'lonceng',
        'c': 'kerucut',
        'x': 'cembung',
        'f': 'datar',
        'k': 'bertonjol',
        's': 'cekung'
    },
    'cap-surface': {
        'f': 'berserat',
        'g': 'beralur',
        'y': 'bersisik',
        's': 'halus'
    },
    'cap-color': {
        'n': 'coklat',
        'b': 'krem',
        'c': 'kayu manis',
        'g': 'abu-abu',
        'r': 'hijau',
        'p': 'merah muda',
        'u': 'ungu',
        'e': 'merah',
        'w': 'putih',
        'y': 'kuning'
    },
    'bruises': {
        't': 'memar',
        'f': 'tidak'
    },
    'odor': {
        'a': 'almond',
        'l': 'anis',
        'c': 'kreosot',
        'y': 'amis',
        'f': 'busuk',
        'm': 'apek',
        'n': 'tidak ada',
        'p': 'menyengat',
        's': 'pedas'
    },
    'gill-attachment': {
        'a': 'menempel',
        'd': 'menurun',
        'f': 'bebas',
        'n': 'berlekuk'
    },
    'gill-spacing': {
        'c': 'rapat',
        'w': 'padat',
        'd': 'jarang'
    },
    'gill-size': {
        'b': 'lebar',
        'n': 'sempit'
    },
    'gill-color': {
        'k': 'hitam',
        'n': 'coklat',
        'b': 'krem',
        'h': 'coklat tua',
        'g': 'abu-abu',
        'r': 'hijau',
        'o': 'oranye',
        'p': 'merah muda',
        'u': 'ungu',
        'e': 'merah',
        'w': 'putih',
        'y': 'kuning'
    },
    'stalk-shape': {
        'e': 'membesar',
        't': 'meruncing'
    },
    'stalk-root': {
        'b': 'berumbi',
        'c': 'klub',
        'u': 'cawan',
        'e': 'rata',
        'z': 'rhizomorf',
        'r': 'berakar',
        '?': 'tidak diketahui'
    },
    'stalk-surface-above-ring': {
        'f': 'berserat',
        'y': 'bersisik',
        'k': 'sutra',
        's': 'halus'
    },
    'stalk-surface-below-ring': {
        'f': 'berserat',
        'y': 'bersisik',
        'k': 'sutra',
        's': 'halus'
    },
    'stalk-color-above-ring': {
        'n': 'coklat',
        'b': 'krem',
        'c': 'kayu manis',
        'g': 'abu-abu',
        'o': 'oranye',
        'p': 'merah muda',
        'e': 'merah',
        'w': 'putih',
        'y': 'kuning'
    },
    'stalk-color-below-ring': {
        'n': 'coklat',
        'b': 'krem',
        'c': 'kayu manis',
        'g': 'abu-abu',
        'o': 'oranye',
        'p': 'merah muda',
        'e': 'merah',
        'w': 'putih',
        'y': 'kuning'
    },
    'veil-type': {
        'p': 'parsial',
        'u': 'universal'
    },
    'veil-color': {
        'n': 'coklat',
        'o': 'oranye',
        'w': 'putih',
        'y': 'kuning'
    },
    'ring-number': {
        'n': 'tidak ada',
        'o': 'satu',
        't': 'dua'
    },
    'ring-type': {
        'c': 'jaring laba-laba',
        'e': 'menghilang',
        'f': 'melebar',
        'l': 'besar',
        'n': 'tidak ada',
        'p': 'menjuntai',
        's': 'menyelubungi',
        'z': 'zona'
    },
    'spore-print-color': {
        'k': 'hitam',
        'n': 'coklat',
        'b': 'krem',
        'h': 'coklat tua',
        'r': 'hijau',
        'o': 'oranye',
        'u': 'ungu',
        'w': 'putih',
        'y': 'kuning'
    },
    'population': {
        'a': 'melimpah',
        'c': 'bergerombol',
        'n': 'banyak',
        's': 'tersebar',
        'v': 'beberapa',
        'y': 'soliter'
    },
    'habitat': {
        'g': 'rumput',
        'l': 'daun',
        'm': 'padang rumput',
        'p': 'jalan setapak',
        'u': 'perkotaan',
        'w': 'limbah',
        'd': 'hutan'
    }
}

# Membaca file data
df = pd.read_csv(input_file, header=None, names=columns)

# Mapping setiap kolom sesuai dictionary
for col in columns:
    if col in mapping_dict:
        df[col] = df[col].map(mapping_dict[col])

# Menyimpan ke file CSV hasil mapping
df.to_csv(output_file, index=False)

print(f"Preprocessing dan mapping selesai. File CSV disimpan sebagai {output_file}")