import sentencepiece

sp = sentencepiece.SentencePieceProcessor()
sp.load('../weights/llama-7b-hf/tokenizer.model')
input_ids = sp.encode('Once upon a time, ')
print(input_ids)
# tensor([[    0,  9038,  2501,   263,   931, 29892, 29871]])
# [9038, 2501, 263, 931, 29892, 29871]