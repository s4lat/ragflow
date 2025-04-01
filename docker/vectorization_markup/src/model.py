import torch
import torch
from transformers import AutoTokenizer, AutoModel
from typing import List
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler = logging.FileHandler('ners.log')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(console_handler)

class Vectorizer(object):
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("cointegrated/LaBSE-en-ru")
        self.model = AutoModel.from_pretrained("cointegrated/LaBSE-en-ru")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.debug("init success " + str(self.device))

    def set_device(self, device: str):
        self.model = self.model.to(device)
        self.device = device

    @staticmethod
    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def __call__(self, texts: List[str]) -> List[float]:
        if len(texts) == 0:
            return []
        logger.debug("texts number came is " + str(len(texts)))
        encoded_input = self.tokenizer(texts, padding=True, truncation=True, max_length=512,
                                       return_tensors='pt').to(self.device)
        with torch.no_grad():
            model_output = self.model(**encoded_input)
        logger.debug("texts done")
        return torch.nn.functional.normalize(model_output.pooler_output).tolist()
