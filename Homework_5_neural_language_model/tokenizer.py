import torch 
import pandas as pd 
import numpy as np 
import re

class Tokenizer:
    def __init__(self, data:pd.DataFrame):
        self.vocab, self.word_to_idx, self.idx_to_word = self.create_vocab(data)
        
    def create_vocab(self, df: pd.DataFrame):
        df["text"] = df["text"].apply(self.normalizer)
        vocab = set()
        for i, row in df.iterrows():
            vocab.update(row["text"].split())
        vocab = sorted(list(vocab))
        vocab.append("<pad>")
        vocab.append("<unk>")
        
        word_to_idx = {word: idx for idx, word in enumerate(vocab)}
        idx_to_word = {idx: word for idx, word in enumerate(vocab)}
        
        return vocab, word_to_idx, idx_to_word
    
    def get_vocab(self):
        return self.vocab
    
    def get_word_to_idx(self):
        return self.word_to_idx
    
    def get_idx_to_word(self):
        return self.idx_to_word
    
    def get_vocab_size(self):
        return len(self.vocab)
    
    def normalizer(self, s: str):
        s = s.lower()
        s = re.sub(r'[^a-z\s]', '', s)
        s = re.sub(r'\s+', ' ', s)
        s = s.strip()
        return s
    
    def encode(self, text):
        if isinstance(text, list):
            return [self.encode(t) for t in text]
        
        elif isinstance(text, str):
            text = self.normalizer(text)
            return [self.word_to_idx.get(word, len(self.vocab) - 1)  for word in text.split()]
        
        else:
            raise ValueError("text should be a string or a list of strings")
    
    def decode(self, idxs):
        return " ".join([self.idx_to_word[idx] for idx in idxs])
    
        
if __name__ == "__main__":
    text = """A closed-loop controller or feedback controller is a control 
loop which incorporates feedback, in contrast to an open-loop controller or non-feedback controller. 
A closed-loop controller uses feedback to control states or outputs of a dynamical system. Its name comes 
rom the information path in the system: process inputs (e.g., voltage applied to an electric motor) have 
an effect on the process outputs (e.g., speed or torque of the motor), which is measured with sensors and 
processed by the controller; the result (the control signal) is "fed back" as input to the process, closing 
the loop"""
    data = pd.DataFrame(data = [text], columns = ["text"])
    tokenizer = Tokenizer(data)
    # test 1
    data_test1 = "A closed-loop controller or feedback controller is a control loop which incorporates feedback"
    encoded = tokenizer.encode(data_test1)
    print(encoded)
    decoded = tokenizer.decode(encoded)
    print(decoded)
    
    # #test2
    # print("-"*50)
    # data_test2 = "today is sunday and i am going to the market"
    # encoded = tokenizer.encode(data_test2)
    # print(encoded)
    # decoded = tokenizer.decode(encoded)
    # print(decoded)
    
    #results
    # [0, 7, 28, 12, 36, 18, 12, 26, 0, 11, 28, 53, 22, 18]
    # a closed loop controller or feedback controller is a control loop which incorporates feedback
    # --------------------------------------------------
    # [56, 26, 56, 2, 56, 56, 56, 49, 48, 56]
    # <unk> is <unk> and <unk> <unk> <unk> to the <unk>
    