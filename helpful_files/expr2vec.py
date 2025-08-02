# 只有一个函数的实现，也就是expr2vec。实现了从表达式到向量的转换。
# 这个是使用了cache的那个版本啊
import os
import sys
# 如果再服务器上需要修改为使用5比较好。
#os.environ["CUDA_VISIBLE_DEVICES"] = "5"  # 或 "6" 使用 GPU 6
import numpy as np
from sentence_transformers import SentenceTransformer
import torch

# 全局模型缓存（避免重复加载）
_MODEL = None

def load_model(model_path):
    """加载本地v2模型
    注意modelpath是完整路径
    """
    global _MODEL
    if _MODEL is None:
        _MODEL = SentenceTransformer(
            model_path,
            device = "cuda" if torch.cuda.is_available() else "cpu"
        )
    return _MODEL

def Expr2Vec(
    ExprStrList: list = None,
    model: str = "all-MiniLM-L12-v2", # "bge-m3"
    JFpath: str = "/home/yxd/yyj/learntoconverse-master/main/helpfiles"
) -> list:
    """
    输入: ExprStrList (表达式字符串列表)
    输出: vecs (每个表达式的向量，顺序与输入一致)
    """
    global _MODEL

    if ExprStrList is None:
        ExprStrList = ["Empty Expression"]

    model_path = r"../all-MiniLM-L12-v2-model"  # 替换为实际模型路径
    local_model = _MODEL if _MODEL is not None else load_model(model_path)
    # 批量编码，使用较大batch_size和低精度加速
    with torch.cuda.amp.autocast(enabled=torch.cuda.is_available(), dtype=torch.float16):
        vecs = local_model.encode(
            ExprStrList,
            batch_size=min(100, len(ExprStrList)),  # 可根据显存调整
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_numpy=True
        )
    return vecs

if __name__ == '__main__':
    # 测试用例
    ExprStrList = ["1+2+3", "1-2-3", "1+2+3"]  # 测试重复和新表达式
    vecs = Expr2Vec(ExprStrList)
    print(vecs)
    print(abs(vecs[0] - vecs[1]))