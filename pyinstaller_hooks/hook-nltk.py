'''
Customized hook for nltk for pyinstaller.

The reason to override the default one is to provide only the corpora
necessary for the stopwords rather than all the nltk data.

It is just a copy of the original one, but instead of copying the
entire data folder it copies only the stopwords one.
'''
import nltk
import os
from PyInstaller.utils.hooks import collect_data_files

# add datas for nltk
datas = collect_data_files('nltk', False)

# loop through the data directories and add them
for p in nltk.data.path:
    if os.path.exists(p):
        datas.append((p + "\\corpora\\stopwords", "nltk_data"))

# nltk.chunk.named_entity should be included
hiddenimports = ["nltk.chunk.named_entity"]