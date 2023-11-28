from transformers import BertTokenizer, BertModel
import torch
import scipy.spatial.distance
import pandas as pd


def load_yani_data(file_path="./static/yani-data.csv"):
    return pd.read_csv(file_path, index_col=0)


def tokenize_and_embed(text, model):
    tokenizer = BertTokenizer.from_pretrained(model)
    model = BertModel.from_pretrained(model)

    encoded_input = tokenizer(text, return_tensors="pt")

    with torch.no_grad():
        output = model(**encoded_input)

    embeddings = output.last_hidden_state[:, 0, :]
    vector = embeddings.tolist()[0]

    return vector


def calculate_similarity(vec1, vec2):
    return 1 - scipy.spatial.distance.cosine(vec1, vec2)


def find_most_similar_brand(post_data, df):
    data_vec = tokenize_and_embed(post_data, "bert-base-uncased")

    max_brand, max_score = "", 0
    for i, feature in enumerate(df["特徴"]):
        vec = tokenize_and_embed(feature, "bert-base-uncased")
        score = calculate_similarity(data_vec, vec)
        if score > max_score:
            max_brand = df.loc[i, "銘柄"]
            max_score = score

    return {"brand": max_brand, "score": max_score}


def get_yani(post_data="タバコ最高"):
    df = load_yani_data()
    result = find_most_similar_brand(post_data, df)
    return result


if __name__ == "__main__":
    sample_text = "タバコってなんか気づいたら吸っちゃってるんですよね。世界七不思議くらい不思議です"
    print(get_yani(sample_text))
