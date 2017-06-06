from collections import Counter
import numpy as np
import random

def preprocess(text):
    # Replace punctuation with tokens so we can use them in our model
    text = text.lower()
    text = text.replace('.', ' <PERIOD> ')
    text = text.replace(',', ' <COMMA> ')
    text = text.replace('"', ' <QUOTATION_MARK> ')
    text = text.replace(';', ' <SEMICOLON> ')
    text = text.replace('!', ' <EXCLAMATION_MARK> ')
    text = text.replace('?', ' <QUESTION_MARK> ')
    text = text.replace('(', ' <LEFT_PAREN> ')
    text = text.replace(')', ' <RIGHT_PAREN> ')
    text = text.replace('--', ' <HYPHENS> ')
    text = text.replace('?', ' <QUESTION_MARK> ')
    # text = text.replace('\n', ' <NEW_LINE> ')
    text = text.replace(':', ' <COLON> ')
    words = text.split()
    # Remove all words with  5 or fewer occurences
    word_counts = Counter(words)
    trimmed_words = [word for word in words if word_counts[word] > 1]

    return trimmed_words

t = "Jacob Israel Ben Ẓebi Ashkenazi Emden (* 4. Juni 1697 in Altona; † 19. April 1776 ebenda) – in nichtjüdischen Quellen als Jacob Hertzel oder Jacob Hirschel bezeichnet – war ein Rabbiner, Talmudgelehrter und Gegner der Bewegung des „falschen Messias“ Shabbetaj Zvi. Für das 18. Jahrhundert kann Jacob Emden als der große jüdische Gelehrte Norddeutschlands gelten. Er verkörpert den Übergang von der Vormoderne in die Moderne, obwohl er selbst auf der Schwelle der neuen Zeit stehen blieb und zeitlebens Verfechter eines strengen Traditionalismus war. Bis zu seinem 17. Lebensjahr studierte Jacob Emden den Talmud in Altona bei seinem Vater Zvi Ashkenazi, der als eine der größten rabbinischen Autoritäten seiner Zeit gilt und Rabbiner der großen Dreigemeinde Altona-Hamburg-Wandsbek (in der jüdischen Geschichte nach den hebräischen Initialen der Gemeinschaften auch als „Kehiloth AHU“ bekannt) war. Danach studierte er von 1710 bis 1714 in Amsterdam. Bis zu seinem 13. Lebensjahr studierte Jacob Emden den Talmud in Altona bei seinem Vater Zvi Ashkenazi, der als eine der größten rabbinischen Autoritäten seiner Zeit gilt und Rabbiner der großen Dreigemeinde Altona-Hamburg-Wandsbek (in der jüdischen Geschichte nach den hebräischen Initialen der Gemeinschaften auch als „Kehiloth AHU“ bekannt) war. Nach seiner Bar Mitzwa studierte er von 1710 bis 1714 in Amsterdam. 1715 heiratete er die Tochter des Mordecai ben Naphtali Kohen, Rabbi von Uherský Brod in Mähren, und setzte seine Studien an der Jeschiwa (Talmudhochschule) seines Schwiegervaters fort. Er wurde zu einem großen Kenner talmudischer Literatur, später studierte er Philosophie, Kabbala sowie hebräische Grammatik und versuchte, Latein und Niederländisch zu erlernen, was jedoch durch seinen Glauben erschwert wurde, demgemäß ein Jude sich mit weltlichen Wissenschaften nur während der Stunde der Dämmerung befassen sollte. Dieser Glaube leitet sich von dem biblischen Vers ab (Jos 1,8 EU): „Du studierst [die Tora] Tag und Nacht.“ Eine intensivere Beschäftigung mit fremdem Wissensgut lehnte Jacob Emden jedoch ab und befürwortete dessen Kenntnis nur insoweit, als er es bei Angriffen auf die jüdische Religion oder Kultur zur Verteidigung für notwendig erachtete. Nach drei Jahren intensiver Studien verließ er das Haus seines Schwiegervaters und wurde ein reisender Verkäufer für Schmuck."

threshold = 1e-5
words = preprocess(t)
total_count = len(words)
wc = Counter(words)
print(total_count)
freqs = {word: count/total_count for word, count in wc.items()}
p_drop = {word: 1 - np.sqrt(threshold/freqs[word]) for word in wc}
train_words = [word for word in words if p_drop[word] < random.random()]
print(train_words)
