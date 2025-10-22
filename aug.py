import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas
import nlpaug.flow as nafc
import sys

from nlpaug.util import Action

text = sys.argv[1]

aug = nac.OcrAug()
augmented_texts = aug.augment(text, n=1)
print("Original:")
print(text)
print("Augmented Texts:")
print(augmented_texts)
