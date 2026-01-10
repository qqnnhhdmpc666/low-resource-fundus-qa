import torch
from transformers import MarianMTModel, MarianTokenizer

class LocalTranslator:
    """
    本地中英文翻译模块（MarianMT）
    强制使用 CPU，避免占用主模型显存
    """

    def __init__(self, device="cpu", cache_dir=None):
        self.device = torch.device(device)
        self.cache_dir = cache_dir

        self.zh_en_name = "Helsinki-NLP/opus-mt-zh-en"
        self.en_zh_name = "Helsinki-NLP/opus-mt-en-zh"

        # 中文 -> 英文
        self.zh_en_tokenizer = MarianTokenizer.from_pretrained(
            self.zh_en_name, cache_dir=self.cache_dir
        )
        self.zh_en_model = (
            MarianMTModel.from_pretrained(
                self.zh_en_name, cache_dir=self.cache_dir
            )
            .to(self.device)
            .eval()
        )

        # 英文 -> 中文
        self.en_zh_tokenizer = MarianTokenizer.from_pretrained(
            self.en_zh_name, cache_dir=self.cache_dir
        )
        self.en_zh_model = (
            MarianMTModel.from_pretrained(
                self.en_zh_name, cache_dir=self.cache_dir
            )
            .to(self.device)
            .eval()
        )

    @torch.inference_mode()
    def zh_to_en(self, text: str) -> str:
        inputs = self.zh_en_tokenizer(
            text, return_tensors="pt", truncation=True
        ).to(self.device)
        outputs = self.zh_en_model.generate(**inputs, max_new_tokens=256, num_beams=4)
        return self.zh_en_tokenizer.decode(outputs[0], skip_special_tokens=True)

    @torch.inference_mode()
    def en_to_zh(self, text: str) -> str:
        inputs = self.en_zh_tokenizer(
            text, return_tensors="pt", truncation=True
        ).to(self.device)
        outputs = self.en_zh_model.generate(**inputs, max_new_tokens=256, num_beams=4)
        return self.en_zh_tokenizer.decode(outputs[0], skip_special_tokens=True)
