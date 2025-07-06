'''
Set of functions to be used in different part of the project.
'''
import string
from unidecode import unidecode

def preprocess_label(label: str) -> str:
    '''
    Preprocess a string by replacing the punctuation by whitespace,
    converting it to upper case, normalizing the ascii characters
    and removing the dots.
        
    :param label: str. The string to be preprocessed.
        
    return label: str. The preprocessed string.
    '''
    # Defining the map for treating punctuation
    mapping_translation = str.maketrans({
        element: ' ' for element in string.punctuation.replace('.',
                                                                '')})
    # Normalizing the strings with unidecode
    label = unidecode(label).upper()
    # Replacing punctuation (but dot) by white space
    label = label.translate(mapping_translation)
    # Removing dots
    label = label.replace('.', '')
    # Removing tab
    label = label.replace('\t', '')
    return label

def shingle_label(label: str, window: int) -> list[str]:
    '''
    Create a set of string by applying a rolling
    window on the token.

    return set. The set of shingle for the token.
    '''
    upper_bound = len(label)-window
    return set(label[i:i+window] for i in range(max(1, upper_bound+1)))

def tokenize(label: str, sep: str=None) -> list[str]:
    '''
    Split the label string according to the value of sep.
    
    label: str. The string to split.
    sep: str. The separator around which split the string.
    '''
    return label.split(sep)

def get_label_length(label: str) -> int:
    '''
    Get the number of characters in the input string.
    
    label: str. The input string.
    '''
    return len(label)

def get_labels_mask(labels: list[str], other_labels: list[str]) -> list[bool]:
    '''
    Get a list of booleans with value True
    where the elements in labels are in other_labels.
    
    labels: list[str]. A list of string.
    other_labels: list[str]. The list of string to look for matching elements.
    '''
    return [label in other_labels for label in labels]

def chain_shingles(shingle: set) -> str:
    '''
    Create a string composed by the sorted shingles, separated by ','.

    :param shingle: set. The set of shingles.

    return str. The string with the chained shingles.
    '''
    return ','.join(sorted(list(set.union(*shingle))))
