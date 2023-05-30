import re

def pre_caption(text: str, max_words=50):
    '''
    '''
    text = re.sub(r"([.!\"()*#:;~])", ' ', text.lower())
    text = re.sub(r'\s{2,}', ' ', text)
    text = text.rstrip('\n')
    text = text.strip(' ')

    words = text.split(' ')
    if len(words) > max_words:
        text = ' '.join(words[:max_words])
    
    return text

