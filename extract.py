import tiktoken
import json

encoding = tiktoken.get_encoding("cl100k_base")

vocabulary = {}

for i in range(5_000_000):
    try:
        vocabulary[i] = encoding.decode_single_token_bytes(i)
    except:
        print(f"stopped at idx {i}")
        break


for k, v in vocabulary.items():
    try:
        vocabulary[k] = v.decode("utf-8")
    except:
        vocabulary[k] = str(v)
        print(f"issue encoding idx {k}, value {v}")


with open("./decoder_vocab.json", "w") as log_file:
    json.dump(vocabulary, log_file, indent=4)

with open("./encoder_vocab.json", "w") as log_file:
    json.dump(dict(sorted({v: k for k, v in vocabulary.items()}.items())), log_file, indent=4)
