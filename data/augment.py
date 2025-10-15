import json
import random
import nltk
from nltk.corpus import wordnet
from nltk import pos_tag, word_tokenize

# Download needed NLTK data
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('averaged_perceptron_tagger')

def get_synonyms(word, pos_tag):
    synonyms = set()
    for syn in wordnet.synsets(word, pos=pos_tag):
        if syn is not None:
            for lemma in syn.lemmas():
                synonym = lemma.name().replace('_', ' ').lower()
                if synonym != word.lower():
                    synonyms.add(synonym)
    return list(synonyms)

def pos_tag_to_wordnet(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return None

def synonym_replacement(sentence, n=1):
    words = word_tokenize(sentence)
    tagged = pos_tag(words)
    new_words = words.copy()

    eligible_indices = []
    for i, (word, tag) in enumerate(tagged):
        wn_tag = pos_tag_to_wordnet(tag)
        if wn_tag is not None:
            synonyms = get_synonyms(word, wn_tag)
            if synonyms:
                eligible_indices.append((i, synonyms))

    if not eligible_indices:
        return sentence  # no synonym replacement possible

    n = min(n, len(eligible_indices))
    indices_to_replace = random.sample(eligible_indices, n)

    for i, synonyms in indices_to_replace:
        synonym = random.choice(synonyms)
        new_words[i] = synonym

    return ' '.join(new_words)

def augment_json_file(input_file_path, output_file_path, n_aug_per_sample=1):
    with open(input_file_path, 'r', encoding='utf-8') as f:
        content = f.read()
        print("File content preview:", content[:100])  # prints first 100 chars
        data = json.loads(content)


    augmented_data = []
    for entry in data:
        text = entry.get("text")
        if not text:
            continue

        # Add original entry
        augmented_data.append(entry)

        # Generate augmentations
        for _ in range(n_aug_per_sample):
            augmented_text = synonym_replacement(text, n=2)  # change 2 to any number of replaced words
            new_entry = entry.copy()
            new_entry["text"] = augmented_text
            augmented_data.append(new_entry)

    # Save augmented data
    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(augmented_data, f, indent=2, ensure_ascii=False)

    print(f"Augmented data saved to {output_file_path}")

# Example usage:
augment_json_file(r"D:\Finrapt\Intent_classification\data\intent_each.json", "augmented_data.json", n_aug_per_sample=2)
