from typing import Tuple, List
from googletrans import Translator


class My_Translator():
    def __init__(self, src_lang: str = "en", tar_lang: str = "zh-cn") -> None:
        self.src_lang = src_lang
        self.tar_lang = tar_lang
        self.translator = Translator()

    def is_fullwidth(self, text: str) -> bool:
        return any('\u4e00' <= char <= '\u9fff' for char in text)

    def split_text_into_lines(self, text: str, char_ratio: list) -> list:
        lines = []
        current_length = 0
        last_length = current_length
        
        for ratio in char_ratio:
            target_length = int(ratio * len(text))

            if target_length < 1: target_length = 1
            else: target_length = round(target_length)

            # 确保每行至少有一个字符
            if current_length + target_length > len(text):
                target_length = len(text) - current_length

            current_length += target_length
            lines.append(text[last_length:current_length])
            last_length = current_length

        # 确保最后一行至少有一个字符
        if last_length < len(text):
            lines[-1] += text[last_length:]

        return lines

    def center_align_lines(self, lines: list, is_fullwidth: bool) -> list:
        max_length = max(len(line) for line in lines)
        
        if is_fullwidth:
            space_char = ' '
        else:
            space_char = ' ' * 2
        
        centered_lines = []
        for line in lines:
            l = len(line)
            padding_length = (max_length - l) // 2
            centered_line = padding_length * space_char + line + (max_length - l - padding_length) * space_char
            centered_lines.append(centered_line)
        
        return centered_lines

    def handle_tgt_text(self, text: str, char_ratio: list, centered: bool):
        """
        1. 判断输入的文本是全角字符文本（如中文）还是半角字符
        2. 根据char_ratio将这段文本划分为len(char_ratio)行 （使用某种合适的方案）
        3. 如果centered为True 则将这几行文本做居中对齐,即在前后填充空格让每行的长度与长度最大行相等(想要的是在使用pygame渲染字体后视觉上相等因此需要注意全角字符的填充方式)
        """
        if len(char_ratio) == 1:
            return text
        is_fullwidth = self.is_fullwidth(text)
        lines = self.split_text_into_lines(text, char_ratio)
        if centered:
            lines = self.center_align_lines(lines, is_fullwidth)
        return '\n'.join(lines)

    def handle_src_text(self, text: str) -> Tuple[str, List[float], bool]:
        lines = text.split('\n')
        if len(lines) == 1:
            return text.strip(), [1], False
        else:
            centered = all(len(line) == len(lines[0]) for line in lines)
            char_ratio = []
            total_len = sum(len(line.strip()) if centered else len(line) for line in lines)
            ret_text = " ".join(line.strip() for line in lines)
            char_ratio = [len(line.strip()) / total_len if centered else len(line) / total_len for line in lines]

            return ret_text.rstrip(), char_ratio, centered


    def translate(self, src_text: str) -> str:
        tgt_text = self.translator.translate(src_text, self.tar_lang, self.src_lang).text
        return tgt_text



if "__main__" == __name__:
    text = r"   I   \n want to go  \n to beijing"
