"""
NUS CS4243 Lab3 AY25/26: Image Segmentation and Texture Description - Assignment.
TA: He Qiyuan.
"""
import numpy as np
import cv2
from scipy.ndimage import convolve
import itertools


# ============================================================
# Part 1: Mean-Shift Segmentation (L05)
# ============================================================

# TASK 1.1 #
def construct_feature_space(image: np.array, spatial_weight: float) -> np.array:
    """Convert an RGB image into a 5D feature space for Mean-Shift.

    Each pixel becomes (spatial_weight*row, spatial_weight*col, R, G, B).

    Args:
        image (np.array): Input RGB image of shape (H, W, 3), uint8.
        spatial_weight (float): Weight for spatial coordinates.

    Returns:
        features (np.array): Feature array of shape (H*W, 5), float64.
    """
    # TASK 1.1 #
    h, w = image.shape[:2]
    xv, yv = np.meshgrid(np.arange(0, h), np.arange(0, w), indexing='ij')
    coors = np.stack((xv, yv), axis=-1).astype(np.float64) * spatial_weight
    features = np.concatenate((coors, image), axis=2).reshape((-1, 5))
    # TASK 1.1 #

    return features


# TASK 1.2 #
def mean_shift_step(data: np.array, point: np.array, bandwidth: float) -> np.array:
    """Compute a single Mean-Shift update for a given point.

    1. Compute Euclidean distances from point to all data points.
    2. Apply Gaussian kernel: w_i = exp(-||x_i - point||^2 / (2 * bandwidth^2)).
    3. Compute weighted mean: new_point = sum(w_i * x_i) / sum(w_i).

    Args:
        data (np.array): All data points, shape (N, D).
        point (np.array): Current point, shape (D,).
        bandwidth (float): Kernel bandwidth parameter.

    Returns:
        new_point (np.array): Updated point after one mean-shift step, shape (D,).
    """
    # TASK 1.2 #
    distances = np.linalg.norm(data - point, axis = 1)
    w = np.exp(-distances ** 2 / (2 * bandwidth ** 2))
    w = w[..., np.newaxis]
    new_point = np.sum(w * data, axis=0) / np.sum(w)
    # TASK 1.2 #

    return new_point


# TASK 1.3 #
def mean_shift_segmentation(data: np.array, bandwidth: float, threshold: float = 1e-3, max_iter: int = 50) -> np.array:
    """Run Mean-Shift segmentation on the data.

    For each point:
        1. Iteratively apply mean_shift_step() until shift < threshold or max_iter.
        2. Record the converged mode.
    After all points converge:
        3. Merge modes within bandwidth/2 of each other.
        4. Assign segment labels.

    Args:
        data (np.array): Feature array of shape (N, D).
        bandwidth (float): Kernel bandwidth.
        threshold (float): Convergence threshold for shift distance.
        max_iter (int): Maximum iterations per point.

    Returns:
        labels (np.array): Segment label for each data point, shape (N,).
    """
    # TASK 1.3 #
    N = data.shape[0]
    modes = np.zeros_like(data)
    for i in range(N):
        point = data[i].copy()
        for _ in range(max_iter):
            new_point = mean_shift_step(data, point, bandwidth)
            if np.linalg.norm(new_point - point) < threshold:
                point = new_point
                break
            point = new_point
        modes[i] = point

    labels = -np.ones(N, dtype=int)
    current_label = 0

    # very naive approach.
    # iterate from left to right, on unassigned mode
    # find all other nearby unassigned modes and merge them
    for i in range(N):
        if labels[i] != -1:
            continue
        distances = np.linalg.norm(modes - modes[i], axis=1)
        mask = (distances < bandwidth / 2) & (labels == -1)
        labels[mask] = current_label
        current_label += 1
    

    # TASK 1.3 #

    return labels


# ============================================================
# Part 2: Texture Description and Segmentation (L06)
# ============================================================

# TASK 2.1 #
def create_gabor_filter(sigma: float, theta: float, lambd: float, psi: float = 0) -> np.array:
    """Create a 2D Gabor filter kernel.

    g(x,y) = exp(-(x'^2 + gamma^2 * y'^2)/(2*sigma^2)) * cos(2*pi*x'/lambda + psi)
    where x' = x*cos(theta) + y*sin(theta), y' = -x*sin(theta) + y*cos(theta),
    gamma = 0.5 (fixed aspect ratio).
    Kernel size: (6*sigma+1) x (6*sigma+1).

    Args:
        sigma (float): Standard deviation of the Gaussian envelope.
        theta (float): Orientation of the filter in radians.
        lambd (float): Wavelength of the sinusoidal component.
        psi (float): Phase offset (default 0).

    Returns:
        kernel (np.array): 2D Gabor filter kernel.
    """
    # TASK 2.1 #
    h = 6 * int(sigma) + 1
    len = h // 2
    xs, ys = np.meshgrid(np.arange(-len, len + 1), np.arange(-len, len + 1), indexing='xy')
    x = xs * np.cos(theta) + ys * np.sin(theta)
    y = -xs * np.sin(theta) + ys * np.cos(theta)
    kernel = np.exp(-(x ** 2 + 0.25 * y ** 2) / (2 * sigma ** 2)) * np.cos(2 * np.pi * x / lambd + psi)
    # TASK 2.1 #

    return kernel


# TASK 2.2 #
def create_log_filter(sigma: int) -> np.array:
    """Create a Laplacian of Gaussian (LoG) kernel.

    Kernel size: (6*sigma+1) x (6*sigma+1).

    Args:
        sigma (int): Standard deviation of the Gaussian.

    Returns:
        kernel (np.array): LoG kernel.
    """
    # TASK 2.2 #
    h = 6 * int(sigma) + 1
    len = h // 2
    x, y = np.meshgrid(np.arange(-len, len + 1), np.arange(-len, len + 1), indexing='ij')
    kernel = -(1 / (np.pi * sigma**4)) * (1 - ((x**2 + y**2) / (2 * sigma**2))) * np.exp(-(x**2 + y**2) / (2 * sigma**2))
    # TASK 2.2 #

    return kernel


# TASK 2.3 #
def build_filter_bank_responses(image: np.array, filter_bank: list) -> np.array:
    """Apply the filter bank to the grayscale image and return D-dimensional features.

    Steps:
        1. Convert image to grayscale (float64, normalized to [0,1]).
        2. For each filter kernel in filter_bank, convolve with grayscale image.
        3. Stack all responses into shape (H, W, D).

    The filter bank contains 12 filters:
        - 8 Gabor (4 orientations x 2 scales) applied to grayscale
        - 4 LoG (4 scales) applied to grayscale

    Hint: use convolve(gray, kernel, mode='reflect') from scipy.ndimage.

    Args:
        image (np.array): Input RGB image of shape (H, W, 3), uint8.
        filter_bank (list): List of 2D filter kernels.

    Returns:
        feats (np.array): Feature map of shape (H, W, 12).
    """
    # TASK 2.3 #
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float64) / 255.0
    results = list(map(lambda kernel: convolve(gray, kernel, mode='reflect'), filter_bank))
    feats = np.stack(results, axis=-1)
    # TASK 2.3 #

    return feats


# TASK 2.4 #
from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import KDTree


class TextonDictionary:
    def __init__(self, filter_bank, n_textons=200):
        self.n_textons = n_textons
        self.filter_bank = filter_bank
        self.cluster_centers_ = None  # shape (n_textons, D) after learning
        self.tree_ = None             # KDTree built from cluster_centers_

    def learn_dictionary(self, training_imgs):
        """Learn texton dictionary from training images.

        Steps:
            1. For each training image, call build_filter_bank_responses()
               to get per-pixel feature vectors of shape (H, W, D).
            2. Reshape and collect all feature vectors from all images into
               a single array of shape (total_pixels, D).
            3. Cluster them using MiniBatchKMeans(n_clusters=self.n_textons).
            4. Store the cluster centers in self.cluster_centers_.
            5. Build a KDTree from self.cluster_centers_ and store in self.tree_.

        API reference:
            kmeans = MiniBatchKMeans(n_clusters=..., random_state=0, batch_size=1000)
            kmeans.fit(data)          # data: (N, D)
            kmeans.cluster_centers_   # result: (n_clusters, D)

            self.tree_ = KDTree(self.cluster_centers_)

        Args:
            training_imgs (list[np.array]): List of training RGB images.
        """
        # TASK 2.4a #
        D = len(self.filter_bank)
        imgs_response = list(map(lambda img: build_filter_bank_responses(img, self.filter_bank).reshape(-1, D), training_imgs))
        data = np.concatenate(imgs_response, axis=0)

        kmeans = MiniBatchKMeans(n_clusters=self.n_textons, random_state=0, batch_size=1000)
        kmeans.fit(data)
        self.cluster_centers_ = kmeans.cluster_centers_

        self.tree_ = KDTree(kmeans.cluster_centers_)
        # TASK 2.4a #

        pass

    def assign_textons(self, img):
        """Assign a texton ID to each pixel of the input image.

        Steps:
            1. Call build_filter_bank_responses() to get features of shape (H, W, D).
            2. Reshape to (H*W, D).
            3. Use self.tree_.query() to find the nearest cluster center for each pixel.
            4. Reshape the indices to (H, W) as the texton map.

        API reference:
            distances, indices = self.tree_.query(features, k=1)
            # features: (N, D)  ->  indices: (N, 1), distances: (N, 1)
            texton_map = indices.flatten().reshape(H, W)

        Args:
            img (np.array): Input RGB image of shape (H, W, 3).

        Returns:
            texton_map (np.array): Texton ID map of shape (H, W).
        """
        # TASK 2.4b #
        img_response = build_filter_bank_responses(img, self.filter_bank)
        H, W, D = img_response.shape
        data = img_response.reshape(-1, D)
        texton_map = self.tree_.query(data, return_distance=False).flatten().reshape(H, W)
        # TASK 2.4b #

        return texton_map


# TASK 2.5 #
def compute_texton_histogram(texton_map: np.array, n_textons: int, window_size: int) -> np.array:
    """Compute per-pixel texton histogram within a rectangular window.

    For each pixel, gather texton IDs in a window_size x window_size window
    centered at that pixel. Compute the normalized histogram (sum to 1).

    Args:
        texton_map (np.array): Texton ID map of shape (H, W). Range of values must be [0, n_textons)
        n_textons (int): Number of textons (histogram bins).
        window_size (int): Side length of the rectangular window. Must be odd.

    Returns:
        hists (np.array): Per-pixel histogram features of shape (H, W, n_textons).
    """
    # TASK 2.5 #
    # range of values for labels is [0, n_textons)
    H, W = texton_map.shape[:2]
    gap = window_size // 2
    hists = np.zeros((H, W, n_textons))
    bins = [i for i in range(n_textons + 1)]
    for i, j in itertools.product(range(H), range(W)):
        rs, re = max(0, i - gap), min(H - 1, i + gap)
        cs, ce = max(0, j - gap), min(W - 1, j + gap)
        values, _ =  np.histogram(texton_map[rs:re+1, cs:ce+1], bins)
        values = values / ((re - rs + 1) * (ce - cs + 1))
        hists[i, j] = values
    # TASK 2.5 #

    return hists
