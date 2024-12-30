from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import os.path as osp
import random, os
import cv2

# import cPickle as cp
import _pickle as cp
import scipy.signal as ssig
import scipy.stats as sstat
import pygame, pygame.locals
from pygame import freetype

# import Image
from PIL import Image
import math
from common import *
import pickle
from trans_utils import My_Translator
import unicodedata


def sample_weighted(p_dict):
    ps = list(p_dict.keys())
    return p_dict[np.random.choice(ps, p=ps)]


def move_bb(bbs, t):
    """
    Translate the bounding-boxes in by t_x,t_y.
    BB : 2x4xn
    T  : 2-long np.array
    """
    return bbs + t[:, None, None]


def crop_safe(arr, rect, bbs=[], pad=0):
    """
    ARR : arr to crop
    RECT: (x,y,w,h) : area to crop to
    BBS : nx4 xywh format bounding-boxes
    PAD : percentage to pad

    Does safe cropping. Returns the cropped rectangle and
    the adjusted bounding-boxes
    """
    rect = np.array(rect)
    rect[:2] -= pad
    rect[2:] += 2 * pad
    v0 = [max(0, rect[0]), max(0, rect[1])]
    v1 = [min(arr.shape[0], rect[0] + rect[2]), min(arr.shape[1], rect[1] + rect[3])]
    arr = arr[v0[0] : v1[0], v0[1] : v1[1], ...]
    if len(bbs) > 0:
        for i in range(len(bbs)):
            bbs[i, 0] -= v0[0]
            bbs[i, 1] -= v0[1]
        return arr, bbs
    else:
        return arr


class BaselineState(object):
    curve = lambda this, a: lambda x: a * x * x
    differential = lambda this, a: lambda x: 2 * a * x
    a = [0.50, 0.05]

    def get_sample(self):
        """
        Returns the functions for the curve and differential for a and b
        """
        sgn = 1.0
        if np.random.rand() < 0.5:
            sgn = -1

        a = self.a[1] * np.random.randn() + sgn * self.a[0]
        return {
            "curve": self.curve(a),
            "diff": self.differential(a),
        }


class RenderFont(object):
    """
    Outputs a rasterized font sample.
        Output is a binary mask matrix cropped closesly with the font.
        Also, outputs ground-truth bounding boxes and text string
    """

    def __init__(self, data_dir="data", paired_text=False):
        # distribution over the type of text:
        # whether to get a single word, paragraph or a line:
        self.p_text = {0.0: "WORD", 0.0: "LINE", 1.0: "PARA"}

        ## TEXT PLACEMENT PARAMETERS:
        self.f_shrink = 0.90
        self.max_shrink_trials = 5  # 0.9^5 ~= 0.6
        # the minimum number of characters that should fit in a mask
        # to define the maximum font height.
        self.min_nchar = 2
        self.min_font_h = 16  # px : 0.6*12 ~ 7px <= actual minimum height
        self.max_font_h = 120  # px
        self.p_flat = 0.10

        # curved baseline:
        self.p_curved = 1.0
        self.baselinestate = BaselineState()

        # text-source : gets english text:
        if paired_text:
            self.text_source = PairedTextSource(
                min_nchar=self.min_nchar, fn=osp.join(data_dir, "pairedtext/en-zh.txt")
            )
        else:
            self.text_source = TextSource(
                min_nchar=self.min_nchar,
                fn=osp.join(data_dir, "newsgroup/newsgroup.txt"),
            )

        # get font-state object:
        self.font_state = FontState(data_dir)
        self.translator = My_Translator("en", "zh-cn")

        pygame.init()

    def render_multiline(self, font, text):
        """
        renders multiline TEXT on the pygame surface SURF with the
        font style FONT.
        A new line in text is denoted by \n, no other characters are
        escaped. Other forms of white-spaces should be converted to space.

        returns the updated surface, words and the character bounding boxes.
        """
        # get the number of lines
        lines = text.split("\n")
        lengths = [len(l) for l in lines]

        # font parameters:
        line_spacing = font.get_sized_height() + 1

        # initialize the surface to proper size:
        line_bounds = font.get_rect(lines[np.argmax(lengths)])
        fsize = (
            round(2.0 * line_bounds.width),
            round(1.25 * line_spacing * len(lines)),
        )
        surf = pygame.Surface(fsize, pygame.locals.SRCALPHA, 32)

        bbs = []
        space = font.get_rect("O")
        x, y = 0, 0
        for l in lines:
            x = 0  # carriage-return
            y += line_spacing  # line-feed

            for ch in l:  # render each character
                if ch.isspace():  # just shift
                    x += space.width
                else:
                    # render the character
                    ch_bounds = font.render_to(surf, (x, y), ch)
                    ch_bounds.x = x + ch_bounds.x
                    ch_bounds.y = y - ch_bounds.y
                    x += ch_bounds.width
                    bbs.append(np.array(ch_bounds))

        # get the union of characters for cropping:
        r0 = pygame.Rect(bbs[0])
        rect_union = r0.unionall(bbs)

        # get the words:
        words = " ".join(text.split())

        # crop the surface to fit the text:
        bbs = np.array(bbs)
        surf_arr, bbs = crop_safe(
            pygame.surfarray.pixels_alpha(surf), rect_union, bbs, pad=5
        )
        surf_arr = surf_arr.swapaxes(0, 1)
        # self.visualize_bb(surf_arr,bbs)
        return surf_arr, words, bbs

    def is_latin(self, text):
        """
        判断字符串是否为拉丁语。
        :param text: 输入字符串
        :return: 如果字符串中的所有字母字符都是拉丁字母，返回 True；否则返回 False。
        """
        for char in text:
            if char.isalpha():  # 只检查字母字符
                if "LATIN" not in unicodedata.name(char):
                    return False
        return True

    def render_curved_(self, font, text_a, text_b):
        """
        use curved baseline for rendering word
        """
        # print("text_a: \n", text_a, "\n=========")
        # print("text_b: \n", text_b, "\n=========")

        # "thank you" len(text_a.split()) = 2
        # "谢谢你" len(text_b.split()) = 1
        # 因此，如果有任一语言为拉丁语则以该text为基准决定是否使用曲线
        if self.is_latin(text_a):
            wl = len(text_a)
            wl_ = len(text_b)
            isword = len(text_a.split()) == 1
        elif self.is_latin(text_b):
            wl = len(text_b)
            wl_ = len(text_a)
            isword = len(text_b.split()) == 1
        else:
            wl = round(len(text_a) * 1.5)
            wl_ = len(text_b)
            isword = len(text_a.split()) == 1

        # do curved iff, the length of the word <= 10
        if (
            not isword
            or wl > 10
            or wl == 1
            or wl_ == 1
            or np.random.rand() > self.p_curved
        ):
            result_a = self.render_multiline(font, text_a)
            result_b = self.render_multiline(font, text_b)
            if result_a is None or result_b is None:
                print("result_a is None or result_b is None")
                return None

            surf_arr_a, text_a, bb_a = result_a
            surf_arr_b, text_b, bb_b = result_b

            return surf_arr_a, text_a, bb_a, surf_arr_b, text_b, bb_b

        # create the surface:
        lspace = font.get_sized_height() + 1
        lbound = font.get_rect(text_a)
        fsize = (round(2.0 * lbound.width), round(3 * lspace))
        surf = pygame.Surface(fsize, pygame.locals.SRCALPHA, 32)
        surf_ = pygame.Surface(fsize, pygame.locals.SRCALPHA, 32)

        # baseline state
        mid_idx = wl // 2
        mid_idx_ = wl_ // 2

        BS = self.baselinestate.get_sample()

        curve = [BS["curve"](i - mid_idx) for i in range(wl)]
        curve_ = [BS["curve"](i - mid_idx_) for i in range(wl_)]

        curve[mid_idx] = -np.sum(curve) / (wl - 1)
        curve_[mid_idx_] = -np.sum(curve_) / (wl_ - 1)

        rots = [
            -int(math.degrees(math.atan(BS["diff"](i - mid_idx) / (font.size / 2))))
            for i in range(wl)
        ]
        rots_ = [
            -int(math.degrees(math.atan(BS["diff"](i - mid_idx_) / (font.size / 2))))
            for i in range(wl_)
        ]

        bbs = []
        bbs_ = []

        # place middle char
        rect = font.get_rect(text_a[mid_idx])
        rect.centerx = surf.get_rect().centerx
        rect.centery = surf.get_rect().centery + rect.height
        rect.centery += curve[mid_idx]
        ch_bounds = font.render_to(surf, rect, text_a[mid_idx], rotation=rots[mid_idx])
        ch_bounds.x = rect.x + ch_bounds.x
        ch_bounds.y = rect.y - ch_bounds.y
        mid_ch_bb = np.array(ch_bounds)

        rect_ = font.get_rect(text_b[mid_idx_])
        rect_.centerx = surf_.get_rect().centerx
        rect_.centery = surf_.get_rect().centery + rect_.height
        rect_.centery += curve_[mid_idx_]
        ch_bounds_ = font.render_to(
            surf_, rect_, text_b[mid_idx_], rotation=rots_[mid_idx_]
        )
        ch_bounds_.x = rect_.x + ch_bounds_.x
        ch_bounds_.y = rect_.y - ch_bounds_.y
        mid_ch_bb_ = np.array(ch_bounds_)

        # render chars to the left and right:
        last_rect = rect
        ch_idx = []
        for i in range(wl):
            # skip the middle character
            if i == mid_idx:
                bbs.append(mid_ch_bb)
                ch_idx.append(i)
                continue

            if i < mid_idx:  # left-chars
                i = mid_idx - 1 - i
            elif i == mid_idx + 1:  # right-chars begin
                last_rect = rect

            ch_idx.append(i)
            ch = text_a[i]

            newrect = font.get_rect(ch)
            newrect.y = last_rect.y
            if i > mid_idx:
                newrect.topleft = (last_rect.topright[0] + 2, newrect.topleft[1])
            else:
                newrect.topright = (last_rect.topleft[0] - 2, newrect.topleft[1])
            newrect.centery = max(
                newrect.height,
                min(fsize[1] - newrect.height, newrect.centery + curve[i]),
            )
            try:
                bbrect = font.render_to(surf, newrect, ch, rotation=rots[i])
            except ValueError:
                bbrect = font.render_to(surf, newrect, ch)
            bbrect.x = newrect.x + bbrect.x
            bbrect.y = newrect.y - bbrect.y
            bbs.append(np.array(bbrect))
            last_rect = newrect

        last_rect_ = rect_
        ch_idx_ = []
        for i in range(wl_):
            # skip the middle character
            if i == mid_idx_:
                bbs_.append(mid_ch_bb_)
                ch_idx_.append(i)
                continue

            if i < mid_idx_:  # left-chars
                i = mid_idx_ - 1 - i
            elif i == mid_idx_ + 1:  # right-chars begin
                last_rect_ = rect_

            ch_idx_.append(i)
            ch = text_b[i]

            newrect_ = font.get_rect(ch)
            newrect_.y = last_rect_.y
            if i > mid_idx_:
                newrect_.topleft = (last_rect_.topright[0] + 2, newrect_.topleft[1])
            else:
                newrect_.topright = (last_rect_.topleft[0] - 2, newrect_.topleft[1])
            newrect_.centery = max(
                newrect_.height,
                min(fsize[1] - newrect_.height, newrect_.centery + curve_[i]),
            )
            try:
                bbrect_ = font.render_to(surf_, newrect_, ch, rotation=rots_[i])
            except ValueError:
                bbrect_ = font.render_to(surf_, newrect_, ch)
            bbrect_.x = newrect_.x + bbrect_.x
            bbrect_.y = newrect_.y - bbrect_.y
            bbs_.append(np.array(bbrect_))
            last_rect_ = newrect_

        # correct the bounding-box order:
        bbs_sequence_order = [None for i in ch_idx]
        for idx, i in enumerate(ch_idx):
            bbs_sequence_order[i] = bbs[idx]
        bbs = bbs_sequence_order

        bbs_sequence_order_ = [None for i in ch_idx_]
        for idx, i in enumerate(ch_idx_):
            bbs_sequence_order_[i] = bbs_[idx]
        bbs_ = bbs_sequence_order_

        # get the union of characters for cropping:
        r0 = pygame.Rect(bbs[0])
        rect_union = r0.unionall(bbs)

        # crop the surface to fit the text:
        bbs = np.array(bbs)
        surf_arr, bbs = crop_safe(
            pygame.surfarray.pixels_alpha(surf), rect_union, bbs, pad=5
        )
        surf_arr = surf_arr.swapaxes(0, 1)

        r0_ = pygame.Rect(bbs_[0])
        rect_union_ = r0_.unionall(bbs_)
        bbs_ = np.array(bbs_)
        surf_arr_, bbs_ = crop_safe(
            pygame.surfarray.pixels_alpha(surf_), rect_union_, bbs_, pad=5
        )
        surf_arr_ = surf_arr_.swapaxes(0, 1)

        return surf_arr, text_a, bbs, surf_arr_, text_b, bbs_

    def render_curved(self, font, word_text):
        """
        use curved baseline for rendering word
        """
        wl = len(word_text)
        isword = len(word_text.split()) == 1

        # do curved iff, the length of the word <= 10
        if not isword or wl > 10 or np.random.rand() > self.p_curved:
            return self.render_multiline(font, word_text)

        # create the surface:
        lspace = font.get_sized_height() + 1
        lbound = font.get_rect(word_text)
        fsize = (round(2.0 * lbound.width), round(3 * lspace))

        surf = pygame.Surface(fsize, pygame.locals.SRCALPHA, 32)

        # baseline state
        mid_idx = wl // 2
        BS = self.baselinestate.get_sample()
        curve = [BS["curve"](i - mid_idx) for i in range(wl)]
        curve[mid_idx] = -np.sum(curve) / (wl - 1)
        rots = [
            -int(math.degrees(math.atan(BS["diff"](i - mid_idx) / (font.size / 2))))
            for i in range(wl)
        ]

        bbs = []
        # place middle char
        rect = font.get_rect(word_text[mid_idx])
        rect.centerx = surf.get_rect().centerx
        rect.centery = surf.get_rect().centery + rect.height
        rect.centery += curve[mid_idx]
        ch_bounds = font.render_to(
            surf, rect, word_text[mid_idx], rotation=rots[mid_idx]
        )
        ch_bounds.x = rect.x + ch_bounds.x
        ch_bounds.y = rect.y - ch_bounds.y
        mid_ch_bb = np.array(ch_bounds)

        # render chars to the left and right:
        last_rect = rect
        ch_idx = []
        for i in range(wl):
            # skip the middle character
            if i == mid_idx:
                bbs.append(mid_ch_bb)
                ch_idx.append(i)
                continue

            if i < mid_idx:  # left-chars
                i = mid_idx - 1 - i
            elif i == mid_idx + 1:  # right-chars begin
                last_rect = rect

            ch_idx.append(i)
            ch = word_text[i]

            newrect = font.get_rect(ch)
            newrect.y = last_rect.y
            if i > mid_idx:
                newrect.topleft = (last_rect.topright[0] + 2, newrect.topleft[1])
            else:
                newrect.topright = (last_rect.topleft[0] - 2, newrect.topleft[1])
            newrect.centery = max(
                newrect.height,
                min(fsize[1] - newrect.height, newrect.centery + curve[i]),
            )
            try:
                bbrect = font.render_to(surf, newrect, ch, rotation=rots[i])
            except ValueError:
                bbrect = font.render_to(surf, newrect, ch)
            bbrect.x = newrect.x + bbrect.x
            bbrect.y = newrect.y - bbrect.y
            bbs.append(np.array(bbrect))
            last_rect = newrect

        # correct the bounding-box order:
        bbs_sequence_order = [None for i in ch_idx]
        for idx, i in enumerate(ch_idx):
            bbs_sequence_order[i] = bbs[idx]
        bbs = bbs_sequence_order

        # get the union of characters for cropping:
        r0 = pygame.Rect(bbs[0])
        rect_union = r0.unionall(bbs)

        # crop the surface to fit the text:
        bbs = np.array(bbs)
        surf_arr, bbs = crop_safe(
            pygame.surfarray.pixels_alpha(surf), rect_union, bbs, pad=5
        )
        surf_arr = surf_arr.swapaxes(0, 1)
        return surf_arr, word_text, bbs

    def get_nline_nchar(self, mask_size, font_height, font_width):
        """
        Returns the maximum number of lines and characters which can fit
        in the MASK_SIZED image.
        """
        H, W = mask_size
        nline = int(np.ceil(H / (2 * font_height)))
        nchar = int(np.floor(W / font_width))
        return nline, nchar

    def place_text(self, text_arrs, back_arr, bbs):
        areas = [-np.prod(ta.shape) for ta in text_arrs]
        order = np.argsort(areas)

        locs = [None for i in range(len(text_arrs))]
        out_arr = np.zeros_like(back_arr)
        for i in order:
            ba = np.clip(back_arr.copy().astype(np.float), 0, 255)
            ta = np.clip(text_arrs[i].copy().astype(np.float), 0, 255)
            ba[ba > 127] = 1e8
            intersect = ssig.fftconvolve(ba, ta[::-1, ::-1], mode="valid")
            safemask = intersect < 1e8

            if not np.any(safemask):  # no collision-free position:
                # warn("COLLISION!!!")
                return back_arr, locs[:i], bbs[:i], order[:i]

            minloc = np.transpose(np.nonzero(safemask))
            loc = minloc[np.random.choice(minloc.shape[0]), :]
            locs[i] = loc

            # update the bounding-boxes:
            bbs[i] = move_bb(bbs[i], loc[::-1])

            # blit the text onto the canvas
            w, h = text_arrs[i].shape
            out_arr[loc[0] : loc[0] + w, loc[1] : loc[1] + h] += text_arrs[i]

        return out_arr, locs, bbs, order

    def place_text_with_mask(self, text_arrs, back_arr, bbs, text_arrs_, back_arr_):
        areas = [-np.prod(ta.shape) for ta in text_arrs]
        order = np.argsort(areas)

        locs = [None for i in range(len(text_arrs))]
        out_arr = np.zeros_like(back_arr)

        for i in order:
            # 获取两个文本的尺寸
            h1, w1 = text_arrs[i].shape
            h2, w2 = text_arrs_[i].shape

            # 计算需要的总空间大小
            total_h = max(h1, h2)
            total_w = max(w1, w2)

            # 检查第一个掩码的安全区域
            ba = np.clip(back_arr.copy().astype(np.float), 0, 255)
            ba[ba > 127] = 1e8
            # 使用较大的尺寸进行卷积
            ta = np.zeros((total_h, total_w), dtype=np.float)
            y_offset = (total_h - h1) // 2
            x_offset = (total_w - w1) // 2
            ta[y_offset : y_offset + h1, x_offset : x_offset + w1] = np.clip(
                text_arrs[i].copy(), 0, 255
            )
            intersect = ssig.fftconvolve(ba, ta[::-1, ::-1], mode="valid")
            safemask = intersect < 1e8

            # 检查第二个掩码的安全区域
            ba_ = np.clip(back_arr_.copy().astype(np.float), 0, 255)
            ba_[ba_ > 127] = 1e8
            ta_ = np.zeros((total_h, total_w), dtype=np.float)
            y_offset_ = (total_h - h2) // 2
            x_offset_ = (total_w - w2) // 2
            ta_[y_offset_ : y_offset_ + h2, x_offset_ : x_offset_ + w2] = np.clip(
                text_arrs_[i].copy(), 0, 255
            )
            intersect_ = ssig.fftconvolve(ba_, ta_[::-1, ::-1], mode="valid")
            safemask_ = intersect_ < 1e8

            # 找到两个掩码都安全的区域
            safemask = safemask & safemask_

            if not np.any(safemask):
                return back_arr, locs[:i], bbs[:i], order[:i]

            # 随机选择一个安全位置
            minloc = np.transpose(np.nonzero(safemask))
            loc = minloc[np.random.choice(minloc.shape[0]), :]
            locs[i] = loc

            # 更新边界框
            bbs[i] = move_bb(bbs[i], loc[::-1])

            # 将文本绘制到画布上
            w, h = text_arrs[i].shape
            out_arr[loc[0] : loc[0] + w, loc[1] : loc[1] + h] += text_arrs[i]

        return out_arr, locs, bbs, order

    def robust_HW(self, mask):
        m = mask.copy()
        m = (~mask).astype("float") / 255
        rH = np.median(np.sum(m, axis=0))
        rW = np.median(np.sum(m, axis=1))
        return rH, rW

    def sample_font_height_px(self, h_min, h_max):
        if np.random.rand() < self.p_flat:
            rnd = np.random.rand()
        else:
            rnd = np.random.beta(2.0, 2.0)

        h_range = h_max - h_min
        f_h = np.floor(h_min + h_range * rnd)
        return f_h

    def bb_xywh2coords(self, bbs):
        """
        Takes an nx4 bounding-box matrix specified in x,y,w,h
        format and outputs a 2x4xn bb-matrix, (4 vertices per bb).
        """
        n, _ = bbs.shape
        coords = np.zeros((2, 4, n))
        for i in range(n):
            coords[:, :, i] = bbs[i, :2][:, None]
            coords[0, 1, i] += bbs[i, 2]
            coords[:, 2, i] += bbs[i, 2:4]
            coords[1, 3, i] += bbs[i, 3]
        return coords

    def render_sample(self, font, mask, mask_):
        """
        Places text in the "collision-free" region as indicated
        in the mask -- 255 for unsafe, 0 for safe.
        The text is rendered using FONT, the text content is TEXT.
        """
        # H,W = mask.shape
        H, W = self.robust_HW(mask)
        f_asp = self.font_state.get_aspect_ratio(font)
        # print(f_asp)

        # find the maximum height in pixels:
        # 考虑三个限制因素：
        # 1. 字符高度小于等于0.9*H
        # 2. 字符高度小于等于(1/f_asp)*W/(self.min_nchar+1)
        #    其中W/(self.min_nchar+1) 是字符宽度的最大值
        #    则(1/f_asp)*W/(self.min_nchar+1) 是字符高度的最大值
        # 3. 字符高度小于等于self.max_font_h
        max_font_h = min(0.9 * H, (1 / f_asp) * W / (self.min_nchar + 1))
        max_font_h = min(max_font_h, self.max_font_h)
        if max_font_h < self.min_font_h:  # not possible to place any text here
            return  # None

        # let's just place one text-instance for now
        ## TODO : change this to allow multiple text instances?
        i = 0
        while i < self.max_shrink_trials and max_font_h > self.min_font_h:
            # if i > 0:
            #     print colorize(Color.BLUE, "shrinkage trial : %d"%i, True)

            # sample a random font-height:
            f_h_px = self.sample_font_height_px(self.min_font_h, max_font_h)
            # print "font-height : %.2f (min: %.2f, max: %.2f)"%(f_h_px, self.min_font_h,max_font_h)
            # convert from pixel-height to font-point-size:
            f_h = self.font_state.get_font_size(font, f_h_px)

            # update for the loop
            max_font_h = f_h_px
            i += 1

            font.size = f_h  # set the font-size

            # compute the max-number of lines/chars-per-line:
            nline, nchar = self.get_nline_nchar(mask.shape[:2], f_h, f_h * f_asp)
            # print "  > nline = %d, nchar = %d"%(nline, nchar)

            assert nline >= 1 and nchar >= self.min_nchar

            # sample text:
            text_type = sample_weighted(self.p_text)
            if isinstance(self.text_source, PairedTextSource):
                text_a, text_b = self.text_source.sample(nline, nchar)
                if (
                    text_a is None
                    or text_b is None
                    or len(text_a) == 0
                    or np.any([len(line) == 0 for line in text_a])
                    or len(text_b) == 0
                    or np.any([len(line) == 0 for line in text_b])
                ):
                    continue
                # print("text_a: \n" + text_a + "\n=========")
                # print("text_b: \n" + text_b + "\n=========")
            else:
                text_a = self.text_source.sample(nline, nchar, text_type)
                if len(text_a) == 0 or np.any([len(line) == 0 for line in text_a]):
                    continue

                handled_text, ratio, centered = self.translator.handle_src_text(text_a)
                text_b = self.translator.translate(handled_text)
                text_b = self.translator.handle_tgt_text(text_b, ratio, centered)
            # print("text_a: \n", text_a , "\n=========")
            # print("text_b: \n", text_b, "\n=========")

            # render the text:
            txt_arr, txt, bb, txt_arr_, txt_, bb_ = self.render_curved_(
                font, text_a, text_b
            )
            # txt_arr,txt,bb = self.render_curved(font, text_a)
            bb = self.bb_xywh2coords(bb)
            bb_ = self.bb_xywh2coords(bb_)
            # make sure that the text-array is not bigger than mask array:
            if np.any(np.r_[txt_arr.shape[:2]] > np.r_[mask.shape[:2]]) or np.any(
                np.r_[txt_arr_.shape[:2]] > np.r_[mask.shape[:2]]
            ):
                # warn("text-array is bigger than mask")
                continue

            # position the text within the mask:
            # print("txt_arr", txt_arr.shape)
            # print("mask", mask.shape)
            # print("bb", bb.shape)

            text_mask, loc, bb, _ = self.place_text_with_mask(
                [txt_arr], mask, [bb], [txt_arr_], mask_
            )
            # text_mask, loc, bb, _ = self.place_text([txt_arr], mask, [bb])

            # visualize debug
            # src_img = Image.fromarray(txt_arr)
            # tgt_img = Image.fromarray(txt_arr_)
            # src_img.save('src_text.png')
            # tgt_img.save('tgt_text.png')
            # print("loc", loc)

            if len(loc) > 0:
                a_center = loc[0] + np.array(
                    [txt_arr.shape[0] // 2, txt_arr.shape[1] // 2]
                )
                b_offset = a_center - np.array(
                    [txt_arr_.shape[0] // 2, txt_arr_.shape[1] // 2]
                )

                text_mask_ = np.zeros_like(mask)
                h, w = txt_arr_.shape
                y, x = b_offset

                if (
                    y >= 0
                    and x >= 0
                    and y + h <= mask.shape[0]
                    and x + w <= mask.shape[1]
                ):
                    text_mask_[y : y + h, x : x + w] = txt_arr_

                    if isinstance(bb_, list):
                        bb_ = np.array(bb_)
                    if len(bb_.shape) == 2 and bb_.shape[1] == 4:
                        bb_ = self.bb_xywh2coords(bb_)

                    bb_ = bb_.copy()
                    for i in range(bb_.shape[-1]):
                        bb_[:, :, i] += b_offset[::-1][:, None]  # 注意坐标转换

                    if isinstance(bb, list):
                        bb = np.array(bb[0])
                    if len(bb.shape) == 2 and bb.shape[1] == 4:
                        bb = self.bb_xywh2coords(bb)

                    if (
                        len(bb.shape) == 3
                        and bb.shape[0] == 2
                        and bb.shape[1] == 4
                        and len(bb_.shape) == 3
                        and bb_.shape[0] == 2
                        and bb_.shape[1] == 4
                    ):

                        try:
                            return {
                                "text_a": [text_mask, loc[0], bb, text_a],
                                "text_b": [text_mask_, b_offset, bb_, text_b],
                            }
                        except Exception as e:
                            print(e)
                            return None

        return None

    def visualize_bb(self, text_arr, bbs):
        ta = text_arr.copy()
        for r in bbs:
            cv2.rectangle(
                ta, (r[0], r[1]), (r[0] + r[2], r[1] + r[3]), color=128, thickness=1
            )
        plt.imshow(ta, cmap="gray")
        plt.show()


class FontState(object):
    """
    Defines the random state of the font rendering
    """

    size = [50, 10]  # normal dist mean, std
    underline = 0.05
    strong = 0.5
    oblique = 0.2
    wide = 0.5
    strength = [0.05, 0.1]  # uniform dist in this interval
    underline_adjustment = [1.0, 2.0]  # normal dist mean, std
    kerning = [
        2,
        5,
        0,
        20,
    ]  # beta distribution alpha, beta, offset, range (mean is a/(a+b))
    border = 0.25
    random_caps = -1  ## don't recapitalize : retain the capitalization of the lexicon
    capsmode = [
        str.lower,
        str.upper,
        str.capitalize,
    ]  # lower case, upper case, proper noun
    curved = 0.2
    random_kerning = 0.2
    random_kerning_amount = 0.1

    def __init__(self, data_dir="data"):

        char_freq_path = osp.join(data_dir, "models/char_freq.cp")
        font_model_path = osp.join(data_dir, "models/font_px2pt.cp")

        # get character-frequencies in the English language:
        with open(char_freq_path, "rb") as f:
            # self.char_freq = cp.load(f)
            u = pickle._Unpickler(f)
            u.encoding = "latin1"
            p = u.load()
            self.char_freq = p

        # get the model to convert from pixel to font pt size:
        with open(font_model_path, "rb") as f:
            # self.font_model = cp.load(f)
            u = pickle._Unpickler(f)
            u.encoding = "latin1"
            p = u.load()
            self.font_model = p

        # get the names of fonts to use:
        self.FONT_LIST = osp.join(data_dir, "fonts/fontlist.txt")
        self.fonts = [
            os.path.join(data_dir, "fonts", f.strip()) for f in open(self.FONT_LIST)
        ]

    def get_aspect_ratio(self, font, size=None):
        """
        Returns the median aspect ratio of each character of the font.
        """
        if size is None:
            size = 12  # doesn't matter as we take the RATIO
        chars = "".join(self.char_freq.keys())
        # print(chars)
        w = np.array(list(self.char_freq.values()))

        # get the [height,width] of each character:
        try:
            sizes = font.get_metrics(chars, size)
            good_idx = [i for i in range(len(sizes)) if sizes[i] is not None]
            # sizes,w = [sizes[i] for i in good_idx], w[good_idx]

            sizes = [sizes[i] for i in good_idx]
            w = np.array([w[i] for i in good_idx])
            sizes = np.array(sizes).astype("float")[:, [3, 4]]

            non_zero_indices = sizes[:, 0] != 0
            r = np.zeros(sizes.shape[0])  # 初始化 r
            r[non_zero_indices] = np.abs(
                sizes[non_zero_indices, 1] / sizes[non_zero_indices, 0]
            )  # 仅计算非零部分
            # r = np.abs(sizes[:,1]/sizes[:,0]) # width/height
            good = np.isfinite(r)
            r = r[good]
            w = w[good]
            w /= np.sum(w)
            r_avg = np.sum(w * r)
            return r_avg
        except Exception as e:
            print(e)
            return 1.0

    def get_font_size(self, font, font_size_px):
        """
        Returns the font-size which corresponds to FONT_SIZE_PX pixels font height.
        """
        m = self.font_model[font.name]
        return m[0] * font_size_px + m[1]  # linear model

    def sample(self):
        """
        Samples from the font state distribution
        """
        return {
            "font": self.fonts[int(np.random.randint(0, len(self.fonts)))],
            "size": self.size[1] * np.random.randn() + self.size[0],
            "underline": np.random.rand() < self.underline,
            "underline_adjustment": max(
                2.0,
                min(
                    -2.0,
                    self.underline_adjustment[1] * np.random.randn()
                    + self.underline_adjustment[0],
                ),
            ),
            "strong": np.random.rand() < self.strong,
            "oblique": np.random.rand() < self.oblique,
            "strength": (self.strength[1] - self.strength[0]) * np.random.rand()
            + self.strength[0],
            "char_spacing": int(
                self.kerning[3] * (np.random.beta(self.kerning[0], self.kerning[1]))
                + self.kerning[2]
            ),
            "border": np.random.rand() < self.border,
            "random_caps": np.random.rand() < self.random_caps,
            "capsmode": random.choice(self.capsmode),
            "curved": np.random.rand() < self.curved,
            "random_kerning": np.random.rand() < self.random_kerning,
            "random_kerning_amount": self.random_kerning_amount,
        }

    def init_font(self, fs):
        """
        Initializes a pygame font.
        FS : font-state sample
        """
        font = freetype.Font(fs["font"], size=fs["size"])
        font.underline = fs["underline"]
        font.underline_adjustment = fs["underline_adjustment"]
        font.strong = fs["strong"]
        font.oblique = fs["oblique"]
        font.strength = fs["strength"]
        char_spacing = fs["char_spacing"]
        font.antialiased = True
        font.origin = True
        return font


class TextSource(object):
    """
    Provides text for words, paragraphs, sentences.
    """

    def __init__(self, min_nchar, fn):
        """
        TXT_FN : path to file containing text data.
        """
        self.min_nchar = min_nchar
        self.fdict = {
            "WORD": self.sample_word,
            "LINE": self.sample_line,
            "PARA": self.sample_para,
        }

        with open(fn, "r") as f:
            self.txt = [l.strip() for l in f.readlines()]

        # distribution over line/words for LINE/PARA:
        self.p_line_nline = np.array([0.85, 0.10, 0.05])
        self.p_line_nword = [4, 3, 12]  # normal: (mu, std)
        self.p_para_nline = [1.0, 1.0]  # [1.7,3.0] # beta: (a, b), max_nline
        self.p_para_nword = [1.7, 3.0, 10]  # beta: (a,b), max_nword

        # probability to center-align a paragraph:
        self.center_para = 0.5

    def check_symb_frac(self, txt, f=0.35):
        """
        T/F return : T iff fraction of symbol/special-charcters in
                     txt is less than or equal to f (default=0.25).
        """
        return np.sum([not ch.isalnum() for ch in txt]) / (len(txt) + 0.0) <= f

    def is_good(self, txt, f=0.35):
        """
        T/F return : T iff the lines in txt (a list of txt lines)
                     are "valid".
                     A given line l is valid iff:
                         1. It is not empty.
                         2. symbol_fraction > f
                         3. Has at-least self.min_nchar characters
                         4. Not all characters are i,x,0,O,-
        """

        def is_txt(l):
            char_ex = ["i", "I", "o", "O", "0", "-"]
            chs = [ch in char_ex for ch in l]
            return not np.all(chs)

        return [
            (len(l) > self.min_nchar and self.check_symb_frac(l, f) and is_txt(l))
            for l in txt
        ]

    def center_align(self, lines):
        """
        PADS lines with space to center align them
        lines : list of text-lines.
        """
        ls = [len(l) for l in lines]
        max_l = max(ls)
        for i in range(len(lines)):
            l = lines[i].strip()
            dl = max_l - ls[i]
            lspace = dl // 2
            rspace = dl - lspace
            lines[i] = " " * lspace + l + " " * rspace
        return lines

    def get_lines(self, nline, nword, nchar_max, f=0.35, niter=100):
        def h_lines(niter=100):
            lines = [""]
            iter = 0
            while not np.all(self.is_good(lines, f)) and iter < niter:
                iter += 1
                line_start = np.random.choice(len(self.txt) - nline)
                lines = [self.txt[line_start + i] for i in range(nline)]
            return lines

        lines = [""]
        iter = 0
        while not np.all(self.is_good(lines, f)) and iter < niter:
            iter += 1
            lines = h_lines(niter=100)
            # get words per line:
            nline = len(lines)
            for i in range(nline):
                words = lines[i].split()
                dw = len(words) - nword[i]
                if dw > 0:
                    first_word_index = random.choice(range(dw + 1))
                    lines[i] = " ".join(
                        words[first_word_index : first_word_index + nword[i]]
                    )

                while len(lines[i]) > nchar_max:  # chop-off characters from end:
                    if not np.any([ch.isspace() for ch in lines[i]]):
                        lines[i] = ""
                    else:
                        lines[i] = lines[i][
                            : len(lines[i]) - lines[i][::-1].find(" ")
                        ].strip()

        if not np.all(self.is_good(lines, f)):
            return  # None
        else:
            return lines

    def sample(self, nline_max, nchar_max, kind="WORD"):
        return self.fdict[kind](nline_max, nchar_max)

    def sample_word(self, nline_max, nchar_max, niter=100):
        rand_line = self.txt[np.random.choice(len(self.txt))]
        words = rand_line.split()
        rand_word = random.choice(words)

        iter = 0
        while iter < niter and (
            not self.is_good([rand_word])[0] or len(rand_word) > nchar_max
        ):
            rand_line = self.txt[np.random.choice(len(self.txt))]
            words = rand_line.split()
            rand_word = random.choice(words)
            iter += 1

        if not self.is_good([rand_word])[0] or len(rand_word) > nchar_max:
            return []
        else:
            return rand_word

    def sample_line(self, nline_max, nchar_max):
        nline = nline_max + 1
        while nline > nline_max:
            nline = np.random.choice([1, 2, 3], p=self.p_line_nline)

        # get number of words:
        nword = [
            self.p_line_nword[2]
            * sstat.beta.rvs(a=self.p_line_nword[0], b=self.p_line_nword[1])
            for _ in range(nline)
        ]
        nword = [max(1, int(np.ceil(n))) for n in nword]

        lines = self.get_lines(nline, nword, nchar_max, f=0.35)
        if lines is not None:
            return "\n".join(lines)
        else:
            return []

    def sample_para(self, nline_max, nchar_max):
        # get number of lines in the paragraph:
        nline = nline_max * sstat.beta.rvs(
            a=self.p_para_nline[0], b=self.p_para_nline[1]
        )
        nline = max(1, int(np.ceil(nline)))

        # get number of words:
        nword = [
            self.p_para_nword[2]
            * sstat.beta.rvs(a=self.p_para_nword[0], b=self.p_para_nword[1])
            for _ in range(nline)
        ]
        nword = [max(1, int(np.ceil(n))) for n in nword]

        lines = self.get_lines(nline, nword, nchar_max, f=0.35)
        if lines is not None:
            # center align the paragraph-text:
            if np.random.rand() < self.center_para:
                lines = self.center_align(lines)
            return "\n".join(lines)
        else:
            return []


class PairedTextSource(object):
    """
    从双语语料中提供配对的文本。
    """

    def __init__(self, min_nchar, fn):
        """
        min_nchar : 最小字符数
        fn : 双语语料文件路径
        """
        self.min_nchar = min_nchar
        self.translator = None

        # 读取并解析双语语料
        self.text_pairs = []
        with open(fn, "r", encoding="utf-8") as f:
            lines = f.readlines()
            for i in range(0, len(lines) - 1, 2):
                src = lines[i].strip()
                tgt = lines[i + 1].strip()
                if src and tgt:  # 确保两行都不为空
                    self.text_pairs.append((src, tgt))

        # 概率分布参数
        self.center_para = 0.5  # 居中对齐概率

    def is_cjk(self, text):
        """
        检查文本是否包含 CJK 字符
        """
        # CJK 统一表意文字的范围
        ranges = [
            (0x4E00, 0x9FFF),  # CJK 统一表意文字
            (0x3400, 0x4DBF),  # CJK 统一表意文字扩展 A
            (0x20000, 0x2A6DF),  # CJK 统一表意文字扩展 B
            (0x2A700, 0x2B73F),  # CJK 统一表意文字扩展 C
            (0x2B740, 0x2B81F),  # CJK 统一表意文字扩展 D
            (0x2B820, 0x2CEAF),  # CJK 统一表意文字扩展 E
            (0x3000, 0x303F),  # CJK 符号和标点
        ]

        for char in text:
            for bottom, top in ranges:
                if bottom <= ord(char) <= top:
                    return True
        return False

    def get_text_width(self, text):
        """
        计算文本的显示宽度，考虑全角和半角字符
        """
        width = 0
        for char in text:
            if self.is_cjk(char) or ord(char) == 0x3000:  # CJK字符或全角空格
                width += 2
            else:
                width += 1
        return width

    def check_symb_frac(self, txt, f=0.35):
        """
        检查特殊字符比例是否合适
        """
        return np.sum([not ch.isalnum() for ch in txt]) / (len(txt) + 0.0) <= f

    def is_good(self, txt_pair, f=0.35):
        """
        检查文本对是否合适:
        1. 不为空
        2. 特殊字符比例合适
        3. 长度大于最小字符数
        4. 不全是特殊字符
        """
        src, tgt = txt_pair

        def is_txt(l):
            char_ex = ["i", "I", "o", "O", "0", "-"]
            chs = [ch in char_ex for ch in l]
            return not np.all(chs)

        return (
            len(src) > self.min_nchar
            and len(tgt) > self.min_nchar
            and self.check_symb_frac(src, f)
            and self.check_symb_frac(tgt, f)
            and is_txt(src)
            and is_txt(tgt)
        )

    # def center_align(self, lines):
    #     """
    #     对文本行进行居中对齐
    #     """
    #     ls = [len(l) for l in lines]
    #     max_l = max(ls)
    #     for i in range(len(lines)):
    #         l = lines[i].strip()
    #         dl = max_l-ls[i]
    #         lspace = dl//2
    #         rspace = dl-lspace
    #         lines[i] = ' '*lspace+l+' '*rspace
    #     return lines

    def center_align(self, lines):
        """
        对文本行进行居中对齐，考虑 CJK 字符的全角特性
        """
        # 计算每行的显示宽度
        widths = [self.get_text_width(l.strip()) for l in lines]
        max_width = max(widths)

        aligned_lines = []
        for i, line in enumerate(lines):
            line = line.strip()
            padding_width = max_width - widths[i]

            if self.is_cjk(line):
                # padding_width *= 2
                left_pad = padding_width // 2
                right_pad = padding_width - left_pad
            else:
                left_pad = padding_width // 2
                right_pad = padding_width - left_pad
            pad_char = " "

            aligned_line = pad_char * left_pad + line + pad_char * right_pad
            aligned_lines.append(aligned_line)

        return aligned_lines

    def sample_pair(self, nline_max, nchar_max, niter=100):
        """
        采样一对合适的文本
        nline_max: 最大行数
        nchar_max: 每行最大字符数
        niter: 最大尝试次数
        """
        for _ in range(niter):
            # 随机选择一对文本
            idx = np.random.randint(len(self.text_pairs))
            src, tgt = self.text_pairs[idx]

            # 检查行数
            src_lines = src.split("\\n")
            tgt_lines = tgt.split("\\n")
            if len(src_lines) > nline_max or len(tgt_lines) > nline_max:
                continue

            # 检查每行的长度
            if any(len(line) > nchar_max for line in src_lines) or any(
                len(line) > nchar_max for line in tgt_lines
            ):
                continue

            # 检查文本质量
            if self.is_good((src, tgt)):
                # 如果需要居中对齐
                if np.random.rand() < self.center_para:
                    src_lines = self.center_align(src_lines)
                    tgt_lines = self.center_align(tgt_lines)
                    # print("center_align")
                    # print(src_lines)
                    # print(tgt_lines)

                return "\n".join(src_lines), "\n".join(tgt_lines)

        return None, None

    def sample(self, nline_max, nchar_max):
        """
        采样接口函数
        nline_max: 最大行数
        nchar_max: 每行最大字符数
        """
        return self.sample_pair(nline_max, nchar_max)
