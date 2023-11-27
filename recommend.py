import MeCab
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.test.utils import common_texts
import scipy.spatial.distance
import pandas as pd


def calculate_vec(text):
    mecab = MeCab.Tagger("-Owakati")
    tokens = mecab.parse(text).split()
    documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(common_texts)]
    model = Doc2Vec(documents, vector_size=100, window=5, min_count=1, workers=4)
    vector = model.infer_vector(tokens)
    return vector


def get_yani(post_data="タバコ最高"):
    df = pd.read_csv("./static/yani-data.csv", index_col=0)
    data_vec = calculate_vec(post_data)

    max_brand, max_score = "", 0
    for i, feature in enumerate(df["特徴"]):
        vec = calculate_vec(feature)
        score = 1 - scipy.spatial.distance.cosine(data_vec, vec)
        print(df.loc[i, "銘柄"], score)
        if score > max_score:
            max_brand = df.loc[i, "銘柄"]
            max_score = score
    return {"brand": max_brand, "score": max_score}


print(get_yani(post_data="タバコってなんか気づいたら吸っちゃってるんですよね。世界七不思議くらい不思議です"))
