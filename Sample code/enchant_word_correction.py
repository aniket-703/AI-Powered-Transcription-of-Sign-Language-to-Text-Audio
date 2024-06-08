import enchant

# Define custom word list with names
custom_words = {
    'ANIKET', 'SHARMA', 'VISHAL', 'DIVYANSH', 'GUPTA',  # Example names
    # Add more custom words here
}

def correct_sentence(user_input):
    # Create an English dictionary object
    english_dict = enchant.Dict("en_US")

    # Correct the sentence
    corrected_words = []
    for word in user_input.split():
        # Convert word to uppercase
        upper_word = word.upper()
        if upper_word in custom_words:
            # If match found in custom_words, convert to lowercase and append
            corrected_words.append(word.lower())
        elif english_dict.check(word):
            corrected_words.append(word)
        else:
            suggestions = english_dict.suggest(word)
            if suggestions:
                corrected_words.append(suggestions[0])
            else:
                corrected_words.append(word)
    corrected_sentence = ' '.join(corrected_words)

    # Print the corrected sentence
    print("Corrected sentence:", corrected_sentence)

    # Write the corrected sentence to a file
    with open("D:/Sign Language Convertor/Output/corrected_sentence.txt", "w") as file:
        file.write(corrected_sentence)

if __name__ == "__main__":
    user_input = input("Enter a sentence: ")
    correct_sentence(user_input)
