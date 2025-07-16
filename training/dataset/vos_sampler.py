# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import random
from dataclasses import dataclass
from typing import List

from training.dataset.vos_segment_loader import LazySegments

MAX_RETRIES = 1000


@dataclass
class SampledFramesAndObjects:
    frames: List[int] # 一组采样的视频帧
    object_ids: List[int] # 要追踪的目标对象ID


class VOSSampler: # 整个video object segmentation 采样器系统的接口设计
    def __init__(self, sort_frames=True):
        # frames are ordered by frame id when sort_frames is True
        self.sort_frames = sort_frames

    def sample(self, video):
        raise NotImplementedError()


class RandomUniformSampler(VOSSampler):
    # RandomUniformSampler 继承自 VOSSampler，用于训练阶段的帧采样逻辑。
    def __init__(
        self,
        num_frames,
        max_num_objects,
        reverse_time_prob=0.0,
    ):
        # num_frames: 从一个视频中采样多少帧
        # max_num_objects: 最多采样多少个目标对象（每个对象代表一个 mask）
        # reverse_time_prob: 反转帧序列的概率（即做时间顺序反转，增强模型的鲁棒性）
        self.num_frames = num_frames
        self.max_num_objects = max_num_objects
        self.reverse_time_prob = reverse_time_prob

    def sample(self, video, segment_loader, epoch=None):
        # 从 video 中采样帧和对象，返回一个 SampledFramesAndObjects 对象。
        for retry in range(MAX_RETRIES):
            if len(video.frames) < self.num_frames:
                raise Exception(
                    f"Cannot sample {self.num_frames} frames from video {video.video_name} as it only has {len(video.frames)} annotated frames."
                )
            start = random.randrange(0, len(video.frames) - self.num_frames + 1)
            frames = [video.frames[start + step] for step in range(self.num_frames)]
            if random.uniform(0, 1) < self.reverse_time_prob:
                # Reverse time
                frames = frames[::-1]

            # Get first frame object ids
            visible_object_ids = []
            # 调用 segment_loader 加载第一帧的分割 mask（每个 object 有一个 mask）
            # segment_loader.load() 会返回一个 dict，键是 object ID，值是这个 object 的 mask（掩码图）
            loaded_segms = segment_loader.load(frames[0].frame_idx)
            if isinstance(loaded_segms, LazySegments):
                # 如果返回的是 LazySegments 类型（即“懒加载”的 mask 对象），就直接取出所有可见 object 的 ID
                # LazySegments for SA1BRawDataset
                visible_object_ids = list(loaded_segms.keys()) # loaded_segms.keys() 是所有 object 的 ID 列表
            else:
                # 对于普通的 dict 类型的分割数据：
                # 遍历所有 object_id, segment（每个对象和它的 mask）
                # segment.sum() 表示该 mask 是否有前景（非空），也就是该 object 是否真正出现在图中。
                # 只有 segment.sum() > 0，才认为是“可见对象”并加入 visible_object_ids
                for object_id, segment in segment_loader.load(
                    frames[0].frame_idx
                ).items():
                    if segment.sum():
                        visible_object_ids.append(object_id)

            # First frame needs to have at least a target to track
            if len(visible_object_ids) > 0:
                break
            if retry >= MAX_RETRIES - 1:
                raise Exception("No visible objects")

        object_ids = random.sample(
            visible_object_ids,
            min(len(visible_object_ids), self.max_num_objects),
        )
        return SampledFramesAndObjects(frames=frames, object_ids=object_ids)


class EvalSampler(VOSSampler):
    """
    VOS Sampler for evaluation: sampling all the frames and all the objects in a video
    """

    def __init__(
        self,
    ):
        super().__init__()

    def sample(self, video, segment_loader, epoch=None):
        """
        Sampling all the frames and all the objects
        """
        if self.sort_frames:
            # ordered by frame id
            frames = sorted(video.frames, key=lambda x: x.frame_idx)
        else:
            # use the original order
            frames = video.frames
        object_ids = segment_loader.load(frames[0].frame_idx).keys()
        if len(object_ids) == 0:
            raise Exception("First frame of the video has no objects")

        return SampledFramesAndObjects(frames=frames, object_ids=object_ids)
