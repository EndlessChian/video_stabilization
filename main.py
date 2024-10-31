from stabilize import *


# load the images and create a plot of the trajectory
(imgs, rimgs), name = load_images(r'C:\Users\magia\Pictures\Camera Roll\a.mp4', OUT_PATH='./frames_result1/'), 'result1'
ws = create_warp_stack(imgs)
# 得到相机运动路径Tx，Ty（主要是通过cv2.findTransformECC完成的）

i, j = 0, 2
ws_x = np.arange(len(ws))
ws_stack = ws[:, i, j]
cum_ws = np.cumsum(ws_stack, axis=0)
plt.scatter(ws_x, ws_stack, label='X Velocity')
plt.plot(ws_x, ws_stack)
plt.scatter(ws_x, cum_ws, label='X Trajectory')
plt.plot(ws_x, cum_ws)
plt.legend()
plt.xlabel('Frame')
plt.savefig(name+'_trajectory.png')

# calculate the smoothed trajectory and output the zeroed images
smoothed_warp, smoothed_trajectory, original_trajectory = \
    moving_average(ws,
            sigma_mat=np.array([    # std
                [5,  # width     变化后的图像宽度，与摄像头远近运动有关，实际使用中我们不希望画面大小发生变化。后续会置零
                 15,    # sin(T)    0.215rad/frame
                 10],   # Tx        0.263rad/frame
                [15,    # -sin(T)
                5,   # height    变化后的图像高度，与摄像头远近运动有关，实际使用中我们不希望画面大小发生变化。后续会置零
                10]     # Ty
            ]))
new_imgs = apply_warping_fullview(images=imgs, raw_images=rimgs, warp_stack=ws-smoothed_warp, PATH='./out/')

#plot the original and smoothed trajectory
f, (a0, a1) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [1, 1]})

i, j = 0, 2
original_trajectory = np.array(original_trajectory)
smoothed_trajectory = np.array(smoothed_trajectory)
origin_stack = original_trajectory[:, i, j]
smooth_stack = smoothed_trajectory[:, i, j]
origin_x = np.arange(len(original_trajectory))
smooth_x = np.arange(len(smoothed_trajectory))
a0.scatter(origin_x, origin_stack, label='Original')
a0.plot(origin_x, origin_stack)
a0.scatter(smooth_x, smooth_stack, label='Smoothed')
a0.plot(smooth_x, smooth_stack)
a0.legend()
a0.set_ylabel('X trajectory')
a0.xaxis.set_ticklabels([])

i, j = 0, 1
origin_stack = original_trajectory[:, i, j]
smooth_stack = smoothed_trajectory[:, i, j]
a1.scatter(origin_x, origin_stack, label='Original')
a1.plot(origin_x, origin_stack)
a1.scatter(smooth_x, smooth_stack, label='Smoothed')
a1.plot(smooth_x, smooth_stack)
a1.legend()
a1.set_xlabel('Frame')
a1.set_ylabel('Sin(Theta) trajectory')
plt.savefig(name+'_smoothed.png')

# create an image that show both the trajectory and video frames
filenames = imshow_with_trajectory(images=new_imgs, warp_stack=ws-smoothed_warp, PATH='./out_'+name+'/', ij=(0, 2))

# create gif
create_gif(filenames, './'+name+'.gif')
