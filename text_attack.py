import sys
from textattack.augmentation import EmbeddingAugmenter

augmenter = EmbeddingAugmenter()
text = sys.argv[1]
augmented_texts=augmenter.augment(text)

print("Original Text:")
print(text)
print("Augmented Text:")
print(augmented_texts[0])
