import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas
import nlpaug.flow as nafc
import sys

from nlpaug.util import Action

text = sys.argv[1]

aug = naw.ContextualWordEmbsAug()
augmented_texts = aug.augment(text, n=1)

print("Original Text:")
print(text)
print("Augmented Text:")
print(augmented_texts[0])
