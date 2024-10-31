"""From joseywallace"""


import os.path
import cv2
import numpy as np
import imageio.v2 as imageio
import matplotlib.pyplot as plt
from scipy import signal
from scipy.ndimage import convolve
# from jupyterthemes import jtplot
# jtplot.style(theme='grade3', grid=False, ticks=True, context='paper', figsize=(20, 15), fscale=1.4)


# HELPER FUNCTIONS
# HELP WITH LOADING AND WRITING TO FILE
def load_images(PATH, OUT_PATH=None):
    """将视频转为一帧一帧的图片并保存（如果有OUT_PATH），方便分析"""
    if not os.path.exists(PATH):
        os.mkdir(PATH)
    cap = cv2.VideoCapture(PATH)
    again = True
    i = 0
    imgs = []
    rimgs = []
    while again:
        again, img = cap.read()
        if again:
            # img_r = img[..., 2].copy()
            # img[..., 2], img[..., 0] = img[..., 0], img_r           # BGR -> RGB
            # cv2.resize(arr, dsize, fx, fy, interpolation)         img, (new-height, new-width), multiple of width, multiple of height,
            img_r = cv2.resize(img, None, fx=0.25, fy=0.25)         # zoom out to quarter缩小至1/4，降低ECC的运算尺度
            imgs.append(img_r)
            rimgs.append(img)
            if not OUT_PATH is None:
                filename = OUT_PATH + "".join([str(0)]*(3-len(str(i)))) + str(i) +'.png'
                cv2.imwrite(filename, img_r)
            i += 1
        else:
            break
    cap.release()
    return imgs, rimgs


def create_gif(filenames, PATH):
    """将静态图片组转为gif动态图"""
    kargs = { 'duration': 0.0333 }      # 周期0.0333s，频率30pfs
    images = []
    for filename in filenames:
        images.append(imageio.imread(filename))
    imageio.mimsave(PATH, images, **kargs)
    

# HELP WITH VISUALIZING
def imshow_with_trajectory(images, warp_stack, PATH, ij):
    if not os.path.exists(PATH):
        os.mkdir(PATH)
    traj_dict = {
        (0, 0): 'Width',       (0, 1): 'sin(Theta)', (0, 2): 'X',
        (1, 0): '-sin(Theta)', (1, 1): 'Height',     (1, 2): 'Y',
    }
    i, j = ij
    filenames = []
    load = np.cumsum(warp_stack[:, i, j], axis=0)
    warp_x = np.arange(len(warp_stack))
    ylabel = traj_dict[ij] + ' Trajectory'
    for k in range(1, len(warp_stack)):
        f, (a0, a1) = plt.subplots(2, 1, gridspec_kw={'height_ratios':[3, 1]})

        a0.axis('off')
        a0.imshow(cv2.cvtColor(images[k], cv2.COLOR_RGB2BGR))       # 保存gif用bgr

        a1.plot(warp_x, load)
        a1.scatter(k, load[k], c='r', s=100)
        a1.set_xlabel('Frame')
        a1.set_ylabel(ylabel)
        
        if not PATH is None:
            filename = '{}/{:0>3}.png'.format(PATH, k)
            plt.savefig(filename)
            filenames.append(filename)
        plt.close()
    return filenames


def get_border_pads(img_shape, warp_invs):
    """
    返回所有帧的在初始状态下占用的的边缘大小，用于填充显示界面，使所有帧都能完整展示。但实际中不可用（非因果）。
    使用四个顶点作为定位（因为从3D到2D面的完全定位需要4点，但这里用的是(2)欧几里德（MOTION_EUCLIDEAN），所以使用3点也没问题）
    """
    maxmin = []
    corners = np.array([                    # 使用顶点，因为矩形的顶点最远离中心
        [0,            0,            1],    # left-top      pad
        [img_shape[1], 0,            1],    # right-top     pad
        [0,            img_shape[0], 1],    # left-bottom   pad
        [img_shape[1], img_shape[0], 1],    # right-bottom  pad
    ], dtype=np.float32).T
    trace_back_corners = corners.copy()
    for warp_inv in warp_invs:
        np.matmul(warp_inv, corners, out=trace_back_corners)      # (3, 4)相当于四个顶点分别计算再合并
        xmax, xmin = trace_back_corners[0].max(), trace_back_corners[0].min()
        ymax, ymin = trace_back_corners[1].max(), trace_back_corners[1].min()
        maxmin.append((ymin, xmin, ymax, xmax))
    maxmin = np.array(maxmin).T
    top = maxmin[0].min()
    print('top', maxmin[0].argmin())
    left = maxmin[1].min()
    print('left', maxmin[1].argmin())
    bottom = maxmin[2].max()
    print('bottom', maxmin[2].argmax())
    right = maxmin[3].max()
    print('right', maxmin[3].argmax())
    # top 78
    # left 70
    # bottom 93
    # right 81
    return int(-top), int(bottom-img_shape[0]), int(-left), int(right-img_shape[1])  # 转换成以图像数组零点(left-top)的位置


# CORE FUNCTIONS
# FINDING THE TRAJECTORY
def get_homography(img1, img2, motion = cv2.MOTION_EUCLIDEAN):
    if len(img1.shape) == 3:
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    if len(img2.shape) == 3:
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)
    if motion == cv2.MOTION_HOMOGRAPHY:
        warpMatrix=np.eye(3, 3, dtype=np.float32)
    else:
        warpMatrix=np.eye(2, 3, dtype=np.float32)
    # 增强系数最大化ECC
    'https://blog.csdn.net/LuohenYJ/article/details/89455660' \
    '(1)平移（MOTION_TRANSLATION）：上图Original图像对坐标点（x，y）变换以获得TRANSLATION图像。我们只需要估算两个参数x和y。'
    '(2)欧几里德（MOTION_EUCLIDEAN）：上图EUCLIDEAN图像是Original图像的旋转和移位版本。所以有三个参数x，y和角度。' \
    '您将注意到当一个正方形经历欧几里德变换时，尺寸不会改变，平行线保持平行，并且在转换后直角保持不变。'
    '(3)仿射（MOTION_AFFINE）：仿射变换是旋转，平移（移位），缩放和剪切的组合。该变换有六个参数。当正方形经历仿射变换时，平行线保持平行，但是以直角相交的线不再保持正交。'
    '(4)Homography（MOTION_HOMOGRAPHY）：上述所有变换都是2D变换。它们不考虑3D效果。另一方面，单应性变换可以解释一些3D效果（但不是全部）。' \
    '该变换有8个参数。使用Homography转换时的正方形可以更改为任何四边形。'
    '在OpenCV中，仿射变换存储在2×3大小的矩阵中。翻译和欧几里德变换是仿射变换的特例。在平移中，旋转，比例和剪切参数为零，而在欧几里德变换中，比例和剪切参数为零。' \
    '因此，平移和欧几里德变换也存储在2×3矩阵中。一旦估计了这个矩阵（我们将在下一节中看到），就可以使用函数warpAffine使图像对齐。' \
    '另一方面，Homography存储在3×3矩阵中。一旦估计了Homography，就可以使用warpPerspective使图像对齐。' \
    'http://xanthippi.ceid.upatras.gr/people/evangelidis/george_files/PAMI_2008.pdf'
    warp_matrix = cv2.findTransformECC(templateImage=img1, inputImage=img2, warpMatrix=warpMatrix, motionType=motion)[1]
    return warp_matrix


def create_warp_stack(imgs):
    warp_stack = np.empty((len(imgs)-1, 2, 3), dtype=np.float32)
    for i, img in enumerate(imgs[1:]):
        warp_stack[i] = get_homography(imgs[i], img)        # pos1是上一帧，pos2是下一帧
    # warp_stack = [get_homography(imgs[i-1], img) for i, img in enumerate(imgs) if i]
    # return np.array(warp_stack, dtype=np.float32)
    return warp_stack


def homography_inv(warp_stack: [..., 2, 3]):
    """
    应用变换，计算连续的位置变换矩阵（因为输入是对每一帧的变换矩阵），即从第一帧开始（初始位置），
    连乘得到从第一帧到当前帧的变换矩阵，然后返回它的逆，即变成从当前帧位置转换到第一帧状态的变换矩阵。
    x' = x + y*sin(O) + Tx
    y' = y - x*sin(O) + Ty
    x',y' is the new pixel position, x,y is the origin frame pixel,
    O is the rotation of camera, Tx,Ty is the translation of camera.
    """
    H_tot = np.eye(3).astype(np.float32)       # 初始矩阵设置为单位矩阵E，因为E*A=A
    w = np.eye(3).astype(np.float32)           # 用于将warp展宽至3*3
    for warp in warp_stack:
        # w = np.concatenate([warp, [[0, 0, 1]]], axis=0)
        w[:2] = warp
        np.matmul(w, H_tot, out=H_tot)
        yield np.linalg.inv(H_tot)#[:2]


# DETERMINING THE SMOOTHED TRAJECTORY
def gauss_convolve(trajectory, window, sigma):
    """
    trajectory
        cv2.MOTION_HOMOGRAPHY [frames, 3, 3]
        cv2.MOTION_EUCLIDEAN [frames, 2, 3]
    window 高斯函数（窗口）大小
    sigma 高斯函数标准差std
    signal.gaussian = exp(- (t-window/2)**2 / (2*std))生成的数组长度为window，最大值出现在window/2、为1.

    gauss(t, 0, std) = 1/sqrt(2*pi*std) * exp(- t**2 / (2*std))
    F[gauss(t, 0, std)] = exp(- w**2 / (2/std))     F[gauss](w=0) = 1, w_0 = sqrt(ln(2)/std)
    IN @ gauss = F(-1)[F[IN] * F[gauss]]    时域卷积 == 频域乘积
    对高频衰减，即低通。从而实现滤波，同时：std越大、gauss越宽、F[gauss]越集中、截止频率越低；
    有：std = ln(2) / (2*pi*f_0)**2 ~= 0.01755762319317072 / (f_0/fps)**2
    实际运动中截止频率f_0是可以变化的，即运动加速快慢影响理想的std大小，但是它往往难以精确估算，可以通过实验确定比较合适值。
    """
    kernel = signal.gaussian(window, std=sigma)
    kernel /= np.sum(kernel)    # 归一化，使其满足概率分布
    return convolve(trajectory, kernel)     # frames + window - 1


def moving_average(warp_stack, sigma_mat):
    """
    通过应用高斯滤波使运动路径平滑化
    但这是一个非因果系统，无法应用于实时控制，实际应用需要寻找其它方法：可以考虑functions.py中的卡曼尔滤波器KFP
    """
    x, y = warp_stack.shape[1:]
    original_trajectory = np.cumsum(warp_stack, axis=0)         # 对Tx，Ty逐帧积分，原始路径
    smoothed_trajectory = np.zeros_like(original_trajectory)    # 高斯滤波（因为抖动往往服从高斯分布）后的平滑路径
    for i in range(x):
        for j in range(y):
            smoothed_trajectory[:, i, j] = gauss_convolve(original_trajectory[:, i, j], 1000, sigma_mat[i, j])

    # 通过相邻做差得到Tx',Ty'，== convolve(arr[:, i, j], [0, 1, -1])     mode='reflect' 完全重合
    smoothed_warp = np.apply_along_axis(lambda m: convolve(m, [0, 1, -1]), axis=0, arr=smoothed_trajectory)

    smoothed_warp[:, 0, 0] = 0      # width
    smoothed_warp[:, 1, 1] = 0      # height
    print(smoothed_warp.shape)
    return smoothed_warp, smoothed_trajectory, original_trajectory


# APPLYING THE SMOOTHED TRAJECTORY TO THE IMAGES

def apply_warping_fullview(images, raw_images, warp_stack, PATH=None):
    """
    应用运动路径，展示防抖效果。
    但是这个算法在计算高分辨率图像时消耗时间极大。
    """
    H = list(homography_inv(warp_stack))      # 将图像转换至初始状态的对应位置，即对齐展示，实现视频防抖。
    top, bottom, left, right = get_border_pads(img_shape=images[0].shape, warp_invs=H)
    # H.insert(0, )
    H = iter(H)
    imgs = []
    warp_init = np.array(
        [
            [0, 0, left],               #
            [0, 0, top],                #
            [0, 0, 0]
        ], dtype=np.float32)
    shape = (images[0].shape[1]+(left+right), images[0].shape[0]+(top+bottom))
    for i, img in enumerate(images[1:]):
        H_tot = next(H) + warp_init
        # 应用Tx，Ty，O，对齐
        img_warp = cv2.warpPerspective(img, H_tot, shape)
        if not PATH is None:
            filename = '{}/{:0>3}.png'.format(PATH, i)  # PATH + "".join([str(0)]*(3-len(str(i)))) + str(i) +'.png'
            cv2.imwrite(filename, img_warp)
        imgs.append(img_warp)
    return imgs
