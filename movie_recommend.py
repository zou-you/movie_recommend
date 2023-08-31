from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel, pipeline
import torch
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import os
import pandas as pd


BERT_BATCH_SIZE = 10
MODEL_NAME = 'sentence-transformers/paraphrase-MiniLM-L6-v2'



def extract_best_indices(m, topk, mask=None):
    """
    Use sum of the cosine distance over all tokens.
    m (np.array): cos matrix of shape (nb_in_tokens, nb_dict_tokens)
    topk (int): number of indices to return (from high to lowest in order)
    """
    # return the sum on all tokens of cosinus for each sentence
    if len(m.shape) > 1:
        cos_sim = np.mean(m, axis=0) 
    else: 
        cos_sim = m
    index = np.argsort(cos_sim)[::-1] # from highest idx to smallest score 
    if mask is not None:
        assert mask.shape == m.shape
        mask = mask[index]
    else:
        mask = np.ones(len(cos_sim))
    mask = np.logical_or(cos_sim[index] != 0, mask) #eliminate 0 cosine distance
    best_index = index[mask][:topk]  
    return best_index




class BertModel:
    def __init__(self, model_name, device=-1, small_memory=True, batch_size=BERT_BATCH_SIZE):
        self.model_name = model_name
        self._set_device(device)
        self.small_device = 'cpu' if small_memory else self.device
        self.batch_size = batch_size
        self.model = None
        self.embed_mat = None
        self.load_pretrained_model()

    def _set_device(self, device):
        if device == -1 or device == 'cpu':
            self.device = 'cpu'
        elif device == 'cuda' or device == 'gpu':
            self.device = 'cuda'
        elif isinstance(device, int) or isinstance(device, float):
            self.device = 'cuda'
        else:  # default
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu")

    def load_pretrained_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        device = -1 if self.device == 'cpu' else 0
        self.pipeline = pipeline('feature-extraction',
                                 model=self.model, tokenizer=self.tokenizer, device=device)

    def embed(self, data):
        """ Create the embedded matrice from original sentences """
        nb_batchs = 1 if (len(data) < self.batch_size) else len(
            data) // self.batch_size
        batchs = np.array_split(data, nb_batchs)
        mean_pooled = []
        for batch in tqdm(batchs, total=len(batchs), desc='Training...'):
            mean_pooled.append(self.transform(batch))
        mean_pooled_tensor = torch.tensor(
            len(data), dtype=float).to(self.small_device)
        mean_pooled = torch.cat(mean_pooled, out=mean_pooled_tensor)
        self.embed_mat = mean_pooled

    @staticmethod
    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(
            -1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def transform(self, data):
        if 'str' in data.__class__.__name__:
            data = [data]
        data = list(data)
        token_dict = self.tokenizer(
            data,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt")
        token_dict = self.to(token_dict, self.device)
        with torch.no_grad():
            token_embed = self.model(**token_dict)
        # each of the 512 token has a 768 or 384-d vector depends on model)
        attention_mask = token_dict['attention_mask']
        # average pooling of masked embeddings
        mean_pooled = self.mean_pooling(
            token_embed, attention_mask)
        mean_pooled = mean_pooled.to(self.small_device)
        return mean_pooled
    
    def to(self, data: dict, device: str):
        """Send all values to device by calling v.to(device)"""
        data = {k: v.to(device) for k, v in data.items()}
        return data

    def predict(self, in_sentence, topk=3):
        input_vec = self.transform(in_sentence)
        mat = cosine_similarity(input_vec, self.embed_mat)
        # best cos sim for each token independantly
        best_index = extract_best_indices(mat, topk=topk)
        return best_index
    

class Recommend:

    def __init__(self) -> None:
        #初始化模型
        self.model = BertModel(model_name=MODEL_NAME, batch_size=BERT_BATCH_SIZE)
        #拿到保存的参数
        self.model_type = "add_actor"
        if self.model_type == "add_actor":
            model_static_dict = torch.load("model_add_actor.pt")
        else:
            model_static_dict = torch.load("model.pt")
        #把参数加载到模型中
        self.model.model.load_state_dict(model_static_dict['model'])
        self.model.embed_mat = model_static_dict['embed_mat']

        # 读取/处理数据
        DATAPATH = 'data/豆瓣电影数据集(2019.3).xlsx' 
        self.columns = ['电影名称', '评分', '类型', '导演', '主演', '剧情简介', '上映日期']
        self.show_columns = ['电影名称', '评分', '类型', '导演', '主演', '剧情简介']
        df = pd.read_excel(DATAPATH, usecols=self.columns)
        self.movie_data = self.process_data(df)
        # 推荐数
        self.num = 100

    def process_data(self, df):
        df = df[~df.剧情简介.isna()]
        df = df[~df.主演.isna()]
        df = df[~df.类型.isna()]
        df = df[~df.评分.isna()]
        df = df.drop_duplicates(subset=['剧情简介'], keep=False)
        df = df[df['剧情简介'].str.len() > 20]

        return df

    @staticmethod
    def comment(series, actors, types):
        items1 = series['主演'].split('/')
        items2 = series['类型'].split('/')

        same_actors, same_types = 0, 0
        # if items1[0] in actors:
        same_actors = len(set(actors) & set(items1))
        same_actor_lst = ",".join(list(set(actors) & set(items1)))
        if same_actors < 3:
            same_actors = len(set(actors[:5]) & set(items1[:5]))
            same_actor_lst = ",".join(list(set(actors[:5]) & set(items1[:5])))

        if items2[0] in types:
            same_types = len(types & set(items2))

        return same_actors, same_types, same_actor_lst
    
    def recommend_sorted(self, target_movie):
        query_sentence = target_movie['剧情简介'].values[0]
        if self.model_type == "add_actor":
            query_sentence = ','.join(target_movie['主演'].values[0].split('/')) + query_sentence

        print("剧情简介：\n", query_sentence, '\n')
        
        # 模型搜索
        indices = self.model.predict(query_sentence, self.num)
        recommend_movies = self.movie_data[self.columns].iloc[indices]

        # 推荐排序
        actors = target_movie['主演'].values[0].split('/')
        types = set(target_movie['类型'].values[0].split('/')[:3])

        recommend_movies['same_actors'], recommend_movies['same_types'], recommend_movies['same_actor_list'] = zip(*recommend_movies.apply(Recommend.comment, axis=1, args=(actors, types)))

        recommend_movies.sort_values(["same_actors", "same_types", "评分"], inplace=True, ascending=False) 

        return recommend_movies


    def recommend(self, movie):

        target_movies = self.movie_data[self.movie_data['电影名称'].str.contains(movie)]
        if len(target_movies) == 0:
            df1 = pd.DataFrame({"ERROR": ["未找到相关电影"]})
            df2 = pd.DataFrame(columns=self.columns)
            return df1, df2
        if target_movies[target_movies['电影名称'] == movie].empty:
            # target_movies.sort_values(by="上映日期", inplace=True) 
            # target_movie = target_movies.iloc[:1]
            min_len = np.argmin(target_movies['电影名称'].str.len())
            target_movie = target_movies.iloc[min_len: min_len+1]
        else:
            target_movie = target_movies[target_movies['电影名称'] == movie]
        
        movie_name = target_movie['电影名称'].values[0]

        res_movies = self.recommend_sorted(target_movie)
        res_movies = res_movies[~(res_movies['电影名称'] == movie_name)][self.show_columns].iloc[:10]
        # res_movies = res_movies[['电影名称', '评分', '类型', '主演', "same_actors", "same_types", 'same_actor_list']]
        res_movies.insert(0, "序号", range(1, 11))

        return target_movie, res_movies