from transformers import T5Tokenizer, T5Model
import scipy.spatial.distance
import torch
import pandas as pd


def calculate_words_vec(words):
    tokenizer = T5Tokenizer.from_pretrained("t5-base")
    model = T5Model.from_pretrained("t5-base")
    tokenized_text = tokenizer(words, return_tensors="pt")

    with torch.no_grad():
        output = model(**tokenized_text, decoder_input_ids=torch.tensor([[1]]))

    vector = output.last_hidden_state.mean(dim=1).squeeze().numpy()
    return vector


def get_yani(post_data="タバコ最高"):
    df = pd.read_csv("./static/yani-data.csv", index_col=0)
    data_vec = calculate_words_vec(post_data)

    max_brand, max_score = "", 0
    for i, feature in enumerate(df["特徴"]):
        vec = calculate_words_vec(feature)
        score = 1 - scipy.spatial.distance.cosine(data_vec, vec)
        if score > max_score:
            max_brand = df.loc[i, "銘柄"]
            max_score = score
    return {"brand": max_brand, "score": max_score}


if __name__ == "__main__":
    print(get_yani("タバコってなんか気づいたら吸っちゃってるんですよね。世界七不思議くらい不思議です"))
