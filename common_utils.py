from spellchecker import SpellChecker


spell = SpellChecker()

# Attempt to use Python's Spell checking library for 'fake' modal checks since this is not passed to GPT which is more forgiving with spelling errors.


def check_for_image_generation(message, trigger_words, threshold):
    corrected_message_text = correct_spelling(message)
    # Convert to a set to avoid substring matches (e.g. 'create' triggering from 'created')
    message_words = set(corrected_message_text.lower().split())
    trigger_count = sum(word in message_words for word in trigger_words)
    return trigger_count >= threshold, corrected_message_text

# Check spelling and maintain capitalization of original message


def correct_spelling(text):
    corrected_words = []
    words = text.split()

    for word in words:
        if word.lower() in spell.unknown([word.lower()]):
            # Attempt to correct the word
            corrected_word = spell.correction(word.lower())

            # If correction returns None, use the original word
            if corrected_word is None:
                corrected_word = word

            # Match the case of the original word
            if word.isupper():
                corrected_word = corrected_word.upper()
            elif word[0].isupper():
                corrected_word = corrected_word.capitalize()

            corrected_words.append(corrected_word)
        else:
            corrected_words.append(word)

    return ' '.join(corrected_words)


# Read the trigger_words txt file
def read_trigger_words(file_path):
    with open(file_path, 'r') as file:

        return [line.strip() for line in file if line.strip()]


def handle_error(say, error):
    say(
        f':no_entry: `Sorry, I ran into an error. The raw error details are as follows:` :no_entry:\n```{error}```')
