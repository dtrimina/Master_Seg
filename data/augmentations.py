import mmcv
import cv2
import numpy as np
from mmcv.utils import deprecated_api_warning

import torch

import common


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, sample):
        for t in self.transforms:
            sample = t(sample)
        return sample


class PhotoMetricDistortion(object):

    def __init__(self, brightness_delta_range=(-32, 32), contrast_range=(0.5, 1.5), saturation_range=(0.5, 1.5),
                 hue_delta=18, prob=0.95):
        self.brightness_delta_lower, self.brightness_delta_upper = brightness_delta_range
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta
        self.prob = prob

    def convert(self, img, alpha=1, beta=0):
        """Multiple with alpha and add beat with clip."""
        img = img.astype(np.float32) * alpha + beta
        img = np.clip(img, 0, 255)
        return img.astype(np.uint8)

    def image_transform(self, img):
        # random brightness
        if common.torch_randint(0, 2):
            img = self.convert(img, beta=common.torch_rand(self.brightness_delta_lower, self.brightness_delta_upper))
        # mode == 0 --> do random contrast first
        # mode == 1 --> do random contrast last
        mode = common.torch_randint(0, 2)
        # do random contrast first
        if mode == 1 and common.torch_randint(0, 2):
            img = self.convert(img, alpha=common.torch_rand(self.contrast_lower, self.contrast_upper))
        # random saturation
        if common.torch_randint(0, 2):
            img = mmcv.bgr2hsv(img)
            img[:, :, 1] = self.convert(img[:, :, 1],
                                        alpha=common.torch_rand(self.saturation_lower, self.saturation_upper))
            img = mmcv.hsv2bgr(img)
        # random hue
        if common.torch_randint(0, 2):
            img = mmcv.bgr2hsv(img)
            img[:, :, 0] = (img[:, :, 0].astype(int) + common.torch_rand(-self.hue_delta, self.hue_delta)) % 180
            img = mmcv.hsv2bgr(img)
        # do random contrast last
        if mode == 0 and common.torch_randint(0, 2):
            img = self.convert(img, alpha=common.torch_rand(self.contrast_lower, self.contrast_upper))
        return img

    def add_noise(self, img):
        if common.torch_randint(0, 2):
            max_noise_data = common.torch_randint(1, 25)
            noise = torch.randint(-1 * max_noise_data, max_noise_data, img.shape).numpy()
            img = img + noise
            img = np.clip(img, 0, 255).astype(np.uint8)
        return img

    def __call__(self, results):
        if common.torch_rand(0, 1) < self.prob:
            assert "img" in results
            results['img'] = self.image_transform(results["img"])
            results['img'] = self.add_noise(results['img'])

        if common.torch_rand(0, 1) < self.prob and "dep" in results:
            results["dep"] = self.image_transform(results["dep"])
            results["dep"] = self.add_noise(results["dep"])
        return results


class GaussianBlur(object):
    """Blurs image with randomly chosen Gaussian blur.
    The image can be a PIL Image or a Tensor, in which case it is expected
    to have [..., C, H, W] shape, where ... means an arbitrary number of leading
    dimensions

    Args:
        kernel_size (int or sequence): Size of the Gaussian kernel.
        sigma (float or tuple of float (min, max)): Standard deviation to be used for
            creating kernel to perform blurring. If float, sigma is fixed. If it is tuple
            of float (min, max), sigma is chosen uniformly at random to lie in the
            given range.

    Returns:
        PIL Image or Tensor: Gaussian blurred version of the input image.

    """

    def __init__(self, kernel_size=(3, 3), sigma=(0.1, 2.0), prob=0.5, kernel_size_options=None):
        super().__init__()

        assert isinstance(sigma, tuple)
        assert isinstance(kernel_size, tuple)
        self.sigma = sigma
        self.kernel_size = kernel_size
        self.kernel_size_options = kernel_size_options
        self.prob = prob

    def get_random_params(self):
        if self.kernel_size_options:
            i = common.torch_rand_choice(self.kernel_size_options)
            kernel_size = (i, i)
        else:
            kernel_size = self.kernel_size
        sigma = common.torch_rand(self.sigma[0], self.sigma[1])
        return kernel_size, sigma

    def __call__(self, results):
        if common.torch_rand(0, 1) < self.prob:
            assert 'img' in results
            kernel_size, sigma = self.get_random_params()
            results['img'] = cv2.GaussianBlur(results['img'], kernel_size, sigma)
        # if common.torch_rand(0, 1) < self.prob and "dep" in results:
        #     kernel_size, sigma = self.get_random_params()
        #     results['dep'] = cv2.GaussianBlur(results['dep'], kernel_size, sigma)

        return results


class Resize(object):
    """Resize images & seg.

    This transform resizes the input image to some scale. If the input dict
    contains the key "scale", then the scale in the input dict is used,
    otherwise the specified scale in the init method is used.

    ``img_scale`` can be Nong, a tuple (single-scale) or a list of tuple
    (multi-scale). There are 4 multiscale modes:

    - ``ratio_range is not None``:
    1. When img_scale is None, img_scale is the shape of image in results
        (img_scale = results['img'].shape[:2]) and the image is resized based
        on the original size. (mode 1)
    2. When img_scale is a tuple (single-scale), randomly sample a ratio from
        the ratio range and multiply it with the image scale. (mode 2)

    - ``ratio_range is None and multiscale_mode == "range"``: randomly sample a
    scale from the a range. (mode 3)

    - ``ratio_range is None and multiscale_mode == "value"``: randomly sample a
    scale from multiple scales. (mode 4)

    Args:
        img_scale (tuple or list[tuple]): Images scales for resizing.
        multiscale_mode (str): Either "range" or "value".
        ratio_range (tuple[float]): (min_ratio, max_ratio)
        keep_ratio (bool): Whether to keep the aspect ratio when resizing the
            image.
    """

    def __init__(self,
                 img_scale=None,
                 multiscale_mode='range',
                 ratio_range=None,
                 keep_ratio=True):
        if img_scale is None:
            self.img_scale = None
        else:
            if isinstance(img_scale, list):
                self.img_scale = img_scale
            else:
                self.img_scale = [img_scale]
            assert mmcv.is_list_of(self.img_scale, tuple)

        if ratio_range is not None:
            # mode 1: given img_scale=None and a range of image ratio
            # mode 2: given a scale and a range of image ratio
            assert self.img_scale is None or len(self.img_scale) == 1
        else:
            # mode 3 and 4: given multiple scales or a range of scales
            assert multiscale_mode in ['value', 'range']

        self.multiscale_mode = multiscale_mode
        self.ratio_range = ratio_range
        self.keep_ratio = keep_ratio

    @staticmethod
    def random_select(img_scales):
        """Randomly select an img_scale from given candidates.

        Args:
            img_scales (list[tuple]): Images scales for selection.

        Returns:
            (tuple, int): Returns a tuple ``(img_scale, scale_dix)``,
                where ``img_scale`` is the selected image scale and
                ``scale_idx`` is the selected index in the given candidates.
        """

        assert mmcv.is_list_of(img_scales, tuple)
        scale_idx = common.torch_randint(0, len(img_scales))
        img_scale = img_scales[scale_idx]
        return img_scale, scale_idx

    @staticmethod
    def random_sample(img_scales):
        """Randomly sample an img_scale when ``multiscale_mode=='range'``.

        Args:
            img_scales (list[tuple]): Images scale range for sampling.
                There must be two tuples in img_scales, which specify the lower
                and uper bound of image scales.

        Returns:
            (tuple, None): Returns a tuple ``(img_scale, None)``, where
                ``img_scale`` is sampled scale and None is just a placeholder
                to be consistent with :func:`random_select`.
        """

        assert mmcv.is_list_of(img_scales, tuple) and len(img_scales) == 2
        img_scale_long = [max(s) for s in img_scales]
        img_scale_short = [min(s) for s in img_scales]
        long_edge = common.torch_randint(
            min(img_scale_long),
            max(img_scale_long) + 1)
        short_edge = common.torch_randint(
            min(img_scale_short),
            max(img_scale_short) + 1)
        img_scale = (long_edge, short_edge)
        return img_scale, None

    @staticmethod
    def random_sample_ratio(img_scale, ratio_range):
        """Randomly sample an img_scale when ``ratio_range`` is specified.

        A ratio will be randomly sampled from the range specified by
        ``ratio_range``. Then it would be multiplied with ``img_scale`` to
        generate sampled scale.

        Args:
            img_scale (tuple): Images scale base to multiply with ratio.
            ratio_range (tuple[float]): The minimum and maximum ratio to scale
                the ``img_scale``.

        Returns:
            (tuple, None): Returns a tuple ``(scale, None)``, where
                ``scale`` is sampled ratio multiplied with ``img_scale`` and
                None is just a placeholder to be consistent with
                :func:`random_select`.
        """

        assert isinstance(img_scale, tuple) and len(img_scale) == 2
        min_ratio, max_ratio = ratio_range
        assert min_ratio <= max_ratio
        # ratio = random.random_sample() * (max_ratio - min_ratio) + min_ratio
        ratio = common.torch_rand(0, 1) * (max_ratio - min_ratio) + min_ratio
        scale = int(img_scale[0] * ratio), int(img_scale[1] * ratio)
        return scale, None

    def _random_scale(self, results):
        """Randomly sample an img_scale according to ``ratio_range`` and
        ``multiscale_mode``.

        If ``ratio_range`` is specified, a ratio will be sampled and be
        multiplied with ``img_scale``.
        If multiple scales are specified by ``img_scale``, a scale will be
        sampled according to ``multiscale_mode``.
        Otherwise, single scale will be used.

        Args:
            results (dict): Result dict from :obj:`data`.

        Returns:
            dict: Two new keys 'scale` and 'scale_idx` are added into
                ``results``, which would be used by subsequent pipelines.
        """

        if self.ratio_range is not None:
            if self.img_scale is None:
                h, w = results['img'].shape[:2]
                scale, scale_idx = self.random_sample_ratio((w, h),
                                                            self.ratio_range)
            else:
                scale, scale_idx = self.random_sample_ratio(
                    self.img_scale[0], self.ratio_range)
        elif len(self.img_scale) == 1:
            scale, scale_idx = self.img_scale[0], 0
        elif self.multiscale_mode == 'range':
            scale, scale_idx = self.random_sample(self.img_scale)
        elif self.multiscale_mode == 'value':
            scale, scale_idx = self.random_select(self.img_scale)
        else:
            raise NotImplementedError

        results['scale'] = scale
        results['scale_idx'] = scale_idx

    def _resize_img(self, results):
        """Resize images with ``results['scale']``."""
        if self.keep_ratio:
            img, scale_factor = mmcv.imrescale(
                results['img'], results['scale'], return_scale=True)
            # the w_scale and h_scale has minor difference
            # a real fix should be done in the mmcv.imrescale in the future
            new_h, new_w = img.shape[:2]
            h, w = results['img'].shape[:2]
            w_scale = new_w / w
            h_scale = new_h / h
        else:
            img, w_scale, h_scale = mmcv.imresize(
                results['img'], results['scale'], return_scale=True)
        scale_factor = np.array([w_scale, h_scale, w_scale, h_scale],
                                dtype=np.float32)
        results['img'] = img
        results['img_shape'] = img.shape
        results['pad_shape'] = img.shape  # in case that there is no padding
        results['scale_factor'] = scale_factor
        results['keep_ratio'] = self.keep_ratio

    def _resize_dep(self, results):
        """Resize images with ``results['scale']``."""
        if self.keep_ratio:
            dep, scale_factor = mmcv.imrescale(
                results['dep'], results['scale'], return_scale=True)
        else:
            dep, w_scale, h_scale = mmcv.imresize(
                results['dep'], results['scale'], return_scale=True)

        results['dep'] = dep

    # def _resize_seg_fs_boundary(self, results):
    #     """Resize semantic segmentation map with ``results['scale']``."""
    #
    #     key = 'seg_fs_boundary'
    #     assert key in results, f'need key seg_fields'
    #     if self.keep_ratio:
    #         gt_seg = mmcv.imrescale(
    #             results[key], results['scale'], interpolation='nearest')
    #     else:
    #         gt_seg = mmcv.imresize(
    #             results[key], results['scale'], interpolation='nearest')
    #     results[key] = gt_seg

    def _resize_seg_mask(self, results, key):
        """Resize semantic segmentation map with ``results['scale']``."""

        assert key in results, f'need key seg_fields'
        if self.keep_ratio:
            gt_seg = mmcv.imrescale(
                results[key], results['scale'], interpolation='nearest')
        else:
            gt_seg = mmcv.imresize(
                results[key], results['scale'], interpolation='nearest')
        results[key] = gt_seg

    def _resize_center_point(self, results, key):
        assert key in results, f'need key center_point'
        h, w = results['img'].shape[:2]
        if self.keep_ratio:
            new_size, scale_factor = mmcv.rescale_size((w, h), results['scale'], return_scale=True)
        else:
            new_size = results['scale']

        new_w = results['center_point'][0] * (new_size[0] / w)
        new_h = results['center_point'][1] * (new_size[1] / h)
        results[key] = (new_w, new_h)

    def __call__(self, results):

        assert "img" in results
        assert "label_cls" in results
        # assert "seg_fs_boundary" in results

        if 'scale' not in results:
            self._random_scale(results)

        if 'center_point' in results:
            self._resize_center_point(results, key='center_point')

        self._resize_img(results)
        self._resize_dep(results)
        self._resize_seg_mask(results, key='label_cls')

        if 'label_freespace' in results:
            self._resize_seg_mask(results, key='label_freespace')
        return results


class RandomRotate(object):
    """Rotate the image & seg.

    Args:
        prob (float): The rotation probability.
        degree (float, tuple[float]): Range of degrees to select from. If
            degree is a number instead of tuple like (min, max),
            the range of degree will be (``-degree``, ``+degree``)
        pad_val (float, optional): Padding value of image. Default: 0.
        seg_pad_val (float, optional): Padding value of segmentation map.
            Default: 255.
        center (tuple[float], optional): Center point (w, h) of the rotation in
            the source image. If not specified, the center of the image will be
            used. Default: None.
        auto_bound (bool): Whether to adjust the image size to cover the whole
            rotated image. Default: False
    """

    def __init__(self,
                 prob,
                 degree,
                 pad_val=0,
                 seg_pad_val=255,
                 center=None,
                 auto_bound=False):
        self.prob = prob
        assert 0 <= prob <= 1
        if isinstance(degree, (float, int)):
            assert degree > 0, f'degree {degree} should be positive'
            self.degree = (-degree, degree)
        else:
            self.degree = degree
        assert len(self.degree) == 2, f'degree {self.degree} should be a ' \
                                      f'tuple of (min, max)'
        self.pal_val = pad_val
        self.seg_pad_val = seg_pad_val
        self.center = center
        self.auto_bound = auto_bound

    def _imrotate_point(self, w, h, center_point, angle, center=None, scale=1.0):
        angle = - angle / 180.0 * np.pi
        center_point = np.asarray(center_point).reshape(1, 2)

        if center is None:  # canvas center taken as coordinate origin in order to perform rotation
            center = np.array([[(w - 1) * 0.5, (h - 1) * 0.5]])
        rotation_mat = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        center_point = (center_point - center) @ rotation_mat + center
        center_point = center_point.reshape(-1)
        return (center[0], center_point[1])

    def __call__(self, results):
        """Call function to rotate image, semantic segmentation maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Rotated results.
        """

        rotate = True if common.torch_rand(0, 1) < self.prob else False
        degree = common.torch_rand(min(*self.degree), max(*self.degree))

        if rotate:
            # rotate image
            results['img'] = mmcv.imrotate(
                results['img'],
                angle=degree,
                border_value=self.pal_val,
                center=self.center,
                auto_bound=self.auto_bound)

            # rotate dep
            if "dep" in results:
                results["dep"] = mmcv.imrotate(
                    results['dep'],
                    angle=degree,
                    border_value=self.pal_val,
                    center=self.center,
                    auto_bound=self.auto_bound)

            # rotate segs
            if 'label_cls' in results.keys():
                results['label_cls'] = mmcv.imrotate(
                    results['label_cls'],
                    angle=degree,
                    border_value=self.seg_pad_val,
                    center=self.center,
                    auto_bound=self.auto_bound,
                    interpolation='nearest')

            # rotate freespace
            if 'label_freespace' in results.keys():
                results['label_freespace'] = mmcv.imrotate(
                    results['label_freespace'],
                    angle=degree,
                    border_value=self.seg_pad_val,
                    center=self.center,
                    auto_bound=self.auto_bound,
                    interpolation='nearest')

            # # rotate segs
            # if 'seg_fs_boundary' in results.keys():
            #     results['seg_fs_boundary'] = mmcv.imrotate(
            #         results['seg_fs_boundary'],
            #         angle=degree,
            #         border_value=self.seg_pad_val,
            #         center=self.center,
            #         auto_bound=self.auto_bound,
            #         interpolation='nearest')

            if 'center_point' in results.keys():
                h, w = results['img_shape'][:2]
                results['key_points'] = self._imrotate_point(
                    w, h, results['center_point'],
                    angle=degree,
                    center=self.center,
                )

        return results


class RandomCrop(object):
    """Random crop the image & seg.

    Args:
        crop_size (tuple): Expected size after cropping, (h, w).
        cat_max_ratio (float): The maximum ratio that single category could
            occupy.
    """

    def __init__(self, crop_size, cat_max_ratio=1., ignore_index=255):
        assert crop_size[0] > 0 and crop_size[1] > 0
        self.crop_size = crop_size
        self.cat_max_ratio = cat_max_ratio
        self.ignore_index = ignore_index

    def get_crop_bbox(self, img):
        """Randomly get a crop bounding box."""
        margin_h = max(img.shape[0] - self.crop_size[0], 0)
        margin_w = max(img.shape[1] - self.crop_size[1], 0)
        offset_h = common.torch_randint(0, margin_h + 1)
        offset_w = common.torch_randint(0, margin_w + 1)
        crop_y1, crop_y2 = offset_h, offset_h + self.crop_size[0]
        crop_x1, crop_x2 = offset_w, offset_w + self.crop_size[1]

        return crop_y1, crop_y2, crop_x1, crop_x2

    def crop(self, img, crop_bbox):
        """Crop from ``img``"""
        crop_y1, crop_y2, crop_x1, crop_x2 = crop_bbox
        img = img[crop_y1:crop_y2, crop_x1:crop_x2, ...]
        return img

    def __call__(self, results):
        """Call function to randomly crop images, semantic segmentation maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Randomly cropped results, 'img_shape' key in result dict is
                updated according to crop size.
        """

        img = results['img']
        crop_bbox = self.get_crop_bbox(img)
        if self.cat_max_ratio < 1.:
            # Repeat 10 times
            for _ in range(10):
                seg_temp = self.crop(results['gt_semantic_seg'], crop_bbox)
                labels, cnt = np.unique(seg_temp, return_counts=True)
                cnt = cnt[labels != self.ignore_index]
                if len(cnt) > 1 and np.max(cnt) / np.sum(
                        cnt) < self.cat_max_ratio:
                    break
                crop_bbox = self.get_crop_bbox(img)

        # crop the image
        img = self.crop(img, crop_bbox)
        img_shape = img.shape
        results['img'] = img
        results['img_shape'] = img_shape

        # crop dep
        if "dep" in results:
            results['dep'] = self.crop(results['dep'], crop_bbox)

        # crop label_cls
        if 'label_cls' in results:
            results['label_cls'] = self.crop(results['label_cls'], crop_bbox)

        # crop label_freespace
        if 'label_freespace' in results:
            results['label_freespace'] = self.crop(results['label_freespace'], crop_bbox)

        # # crop semantic seg
        # if 'seg_fs_boundary' in results:
        #     results['seg_fs_boundary'] = self.crop(results['seg_fs_boundary'], crop_bbox)

        # crop center_point
        if 'center_point' in results:
            crop_y1, crop_y2, crop_x1, crop_x2 = crop_bbox
            px, py = results['center_point']
            px -= crop_x1
            py -= crop_y1
            results['center_point'] = (px, py)

        return results


class RandomFlip(object):
    """Flip the image & seg.

    If the input dict contains the key "flip", then the flag will be used,
    otherwise it will be randomly decided by a ratio specified in the init
    method.

    Args:
        prob (float, optional): The flipping probability. Default: None.
        direction(str, optional): The flipping direction. Options are
            'horizontal' and 'vertical'. Default: 'horizontal'.
    """

    @deprecated_api_warning({'flip_ratio': 'prob'}, cls_name='RandomFlip')
    def __init__(self, prob=None, direction='horizontal'):
        self.prob = prob
        self.direction = direction
        if prob is not None:
            assert prob >= 0 and prob <= 1
        assert direction in ['horizontal', 'vertical']

    def __call__(self, results):
        """Call function to flip bounding boxes, masks, semantic segmentation
        maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Flipped results, 'flip', 'flip_direction' keys are added into
                result dict.
        """

        if 'flip' not in results:
            flip = True if common.torch_rand(0, 1) < self.prob else False
            results['flip'] = flip
        if 'flip_direction' not in results:
            results['flip_direction'] = self.direction
        if results['flip']:
            # flip img
            results['img'] = mmcv.imflip(results['img'], direction=results['flip_direction'])

            # flip dep
            if "dep" in results:
                results['dep'] = mmcv.imflip(results['dep'], direction=results['flip_direction'])

            # flip label_cls
            if 'label_cls' in results:
                # use copy() to make numpy stride positive
                results['label_cls'] = mmcv.imflip(
                    results['label_cls'], direction=results['flip_direction']).copy()

            # flip freespace
            if 'label_freespace' in results:
                # use copy() to make numpy stride positive
                results['label_freespace'] = mmcv.imflip(
                    results['label_freespace'], direction=results['flip_direction']).copy()

            # # flip label_cls
            # if 'seg_fs_boundary' in results:
            #     # use copy() to make numpy stride positive
            #     results['seg_fs_boundary'] = mmcv.imflip(
            #         results['seg_fs_boundary'], direction=results['flip_direction']).copy()
        return results


class Pad(object):
    """Pad the image & mask.

    There are two padding modes: (1) pad to a fixed size and (2) pad to the
    minimum size that is divisible by some number.
    Added keys are "pad_shape", "pad_fixed_size", "pad_size_divisor",

    Args:
        size (tuple, optional): Fixed padding size.
        size_divisor (int, optional): The divisor of padded size.
        pad_val (float, optional): Padding value. Default: 0.
        seg_pad_val (float, optional): Padding value of segmentation map.
            Default: 255.
    """

    def __init__(self,
                 size=None,
                 size_divisor=None,
                 pad_val=0,
                 seg_pad_val=255):
        self.size = size
        self.size_divisor = size_divisor
        self.pad_val = pad_val
        self.seg_pad_val = seg_pad_val
        # only one of size and size_divisor should be valid
        assert size is not None or size_divisor is not None
        assert size is None or size_divisor is None

    def _pad_img(self, results):
        """Pad images according to ``self.size``."""
        if self.size is not None:
            padded_img = mmcv.impad(
                results['img'], shape=self.size, pad_val=self.pad_val)
        elif self.size_divisor is not None:
            padded_img = mmcv.impad_to_multiple(
                results['img'], self.size_divisor, pad_val=self.pad_val)
        results['img'] = padded_img
        results['pad_shape'] = padded_img.shape
        results['pad_fixed_size'] = self.size
        results['pad_size_divisor'] = self.size_divisor

    def _pad_dep(self, results):
        """Pad images according to ``self.size``."""
        if self.size is not None:
            padded_img = mmcv.impad(
                results['dep'], shape=self.size, pad_val=self.pad_val)
        elif self.size_divisor is not None:
            padded_img = mmcv.impad_to_multiple(
                results['dep'], self.size_divisor, pad_val=self.pad_val)
        results['dep'] = padded_img

    def _pad_seg_mask(self, results, key):
        """Pad masks according to ``results['pad_shape']``."""
        assert key in results
        results[key] = mmcv.impad(
            results[key],
            shape=results['pad_shape'][:2],
            pad_val=self.seg_pad_val)

    def __call__(self, results):
        """Call function to pad images, masks, semantic segmentation maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Updated result dict.
        """

        self._pad_img(results)
        self._pad_dep(results)
        self._pad_seg_mask(results, key='label_cls')
        if 'label_freespace' in results:
            self._pad_seg_mask(results, key='label_freespace')
        return results
