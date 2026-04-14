# -*- encoding: utf-8 -*-
'''
@File         :vis_gt.py
@Date         :2026/03/17 14:34:48
@Author       :Binge.Van
@E-mail       :1367955240@qq.com
@Version      :V1.0.0
@Description  :

'''


import os,sys

import json
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R

class_colors = [
    (0,0,255),(255,0,0),(0,255,0),
    (255,255,0),(0,255,255),(255,0,255),
    (125,125,125)
]




# ------------------------ 坐标变换-------------------------
def create_3d_bbox_corners(center,dimensions,rotation):
    l, w, h = dimensions
    
    corners_local = np.array([
        [l/2, w/2, h/2],
        [l/2, w/2, -h/2],
        [l/2, -w/2, h/2],
        [l/2, -w/2, -h/2],
        [-l/2, w/2, h/2],
        [-l/2, w/2, -h/2],
        [-l/2, -w/2, h/2],
        [-l/2, -w/2, -h/2]
    ])
    
    # 创建旋转矩阵 (顺序: yaw -> pitch -> roll)
    roll, pitch, yaw = rotation
    # rot_matrix = R.from_euler('zyx', [yaw, pitch, roll]).as_matrix()
    rot_matrix = R.from_euler('xyz', [roll, pitch, yaw]).as_matrix()
    
    corners_global = corners_local @ rot_matrix.T + center
    
    return corners_global

def project_lidar_to_camera(points_3d,T_lidar2camera):
    
    # 添加齐次坐标
    points_homo = np.hstack([points_3d, np.ones((points_3d.shape[0], 1))])
    
    # 转换到相机坐标系
    points_camera = np.matmul(T_lidar2camera, points_homo.T).T# (T_lidar2camera @ points_homo.T).T
    
    return points_camera[:, :3]

def mask_image(projected_points,mask,image_height,image_width):
    for i, projected_point in enumerate(projected_points):
        if projected_point[0] >=image_width or projected_point[0]<=0 or projected_point[1]>=image_height or projected_point[1] <0:
            mask[i] = False

def project_camera_to_image( points_camera,camera_intrinsic,distortion_coeffs,image_height,image_width):
    points_3d_reshaped = points_camera.reshape(-1, 1, 3).astype(np.float32)
    rvec = np.zeros((3, 1), dtype=np.float32)
    tvec = np.zeros((3, 1), dtype=np.float32)
    # projected_points, jacobian = cv2.projectPoints(
    #         points_3d_reshaped,
    #         rvec,
    #         tvec,
    #         camera_intrinsic,
    #         distortion_coeffs
    #     )

    projected_points, _ = cv2.fisheye.projectPoints(
            points_3d_reshaped,
            rvec,
            tvec,
            camera_intrinsic,
            distortion_coeffs[:4]  # 鱼眼相机只需要前4个畸变系数
        )
    projected_points = projected_points.reshape(-1, 2)
    mask = points_camera[:, 2] > 0
    mask_image(projected_points,mask,image_height,image_width)
    
    return projected_points,mask

def draw_poly_image(img_with_bbox,pixels,mask,color):
    edges = [
            (0, 1), (0, 2), (1, 3), (2, 3),  # 前面
            (4, 5), (4, 6), (5, 7), (6, 7),  # 后面
            (0, 4), (1, 5), (2, 6), (3, 7)   # 连接前后
        ]
    for edge in edges:
        i, j = edge
        if mask[i] and mask[j]:
            cv2.line(img_with_bbox, 
                    tuple(map(int,pixels[i])), 
                    tuple(map(int,pixels[j])), 
                    color, 2)
    
    # 绘制角点
    for i, pixel in enumerate(pixels):
        if mask[i]:
            cv2.circle(img_with_bbox, tuple(map(int,pixel)), 3, (0, 0, 255), -1)
    
    return img_with_bbox




#----------------------- 可视化 ---------------

def draw_bev_grid(bird_view, world_size_x=200, world_size_y=80, 
                  out_size_h=200*8, out_size_w=80*8, ego_position=120,
                  grid_interval=10, color=(128, 128, 128), thickness=1):
    """
    在BEV图上绘制距离网格线，以ego为中心，每隔指定距离画一条线
    
    参数:
        bird_view: BEV图像
        world_size_x: 世界坐标系前后方向范围（米），默认200
        world_size_y: 世界坐标系左右方向范围（米），默认80
        out_size_h: BEV图高度（像素），默认1600
        out_size_w: BEV图宽度（像素），默认640
        ego_position: ego在前后方向的位置（米），默认120（往前120米处）
        grid_interval: 网格间隔（米），默认10
        color: 网格线颜色 (B, G, R)，默认灰色
        thickness: 线条粗细，默认1
    
    返回:
        bird_view: 绘制网格后的BEV图像
    """
    if bird_view is None:
        bird_view = np.zeros((out_size_h, out_size_w, 3), dtype=np.uint8) + 255
    
    # ego在BEV图上的像素位置
    ego_x_pixel = int(out_size_w / 2)  # 左右中心
    ego_y_pixel = int(ego_position * (out_size_h / world_size_x))  # 往前ego_position米的位置
    
    # 缩放比例
    scale_x = out_size_w / world_size_y  # 左右方向：像素/米
    scale_y = out_size_h / world_size_x  # 前后方向：像素/米
    
    # 绘制左右方向的网格线（垂直方向，表示左右距离）
    # 从ego位置开始，向左和向右各绘制
    max_offset_x = int(world_size_y / 2)  # 左右各40米
    for offset in range(grid_interval, max_offset_x + grid_interval, grid_interval):
        # 向右的线（ego右侧）
        x_pixel = int(ego_x_pixel + offset * scale_x)
        if 0 <= x_pixel < out_size_w:
            cv2.line(bird_view, (x_pixel, 0), (x_pixel, out_size_h), color, thickness)
            # 添加距离标签
            cv2.putText(bird_view, f"{offset}m", (x_pixel + 2, ego_y_pixel - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # 向左的线（ego左侧）
        x_pixel = int(ego_x_pixel - offset * scale_x)
        if 0 <= x_pixel < out_size_w:
            cv2.line(bird_view, (x_pixel, 0), (x_pixel, out_size_h), color, thickness)
            # 添加距离标签
            cv2.putText(bird_view, f"{offset}m", (x_pixel + 2, ego_y_pixel - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    # 绘制前后方向的网格线（水平方向，表示前后距离）
    # 从ego位置开始，向前和向后各绘制
    max_offset_y_forward = int(ego_position)  # 向前最多120米
    max_offset_y_backward = int(world_size_x - ego_position)  # 向后最多80米
    
    # 向前的线（ego前方，y减小）
    for offset in range(grid_interval, max_offset_y_forward + grid_interval, grid_interval):
        y_pixel = int(ego_y_pixel - offset * scale_y)
        if 0 <= y_pixel < out_size_h:
            cv2.line(bird_view, (0, y_pixel), (out_size_w, y_pixel), color, thickness)
            # 添加距离标签
            cv2.putText(bird_view, f"{offset}m", (ego_x_pixel + 5, y_pixel - 2),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    # 向后的线（ego后方，y增大）
    for offset in range(grid_interval, max_offset_y_backward + grid_interval, grid_interval):
        y_pixel = int(ego_y_pixel + offset * scale_y)
        if 0 <= y_pixel < out_size_h:
            cv2.line(bird_view, (0, y_pixel), (out_size_w, y_pixel), color, thickness)
            # 添加距离标签
            cv2.putText(bird_view, f"{offset}m", (ego_x_pixel + 5, y_pixel - 2),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    # 绘制ego位置的十字线（更粗，用于标识ego位置）
    cv2.line(bird_view, (ego_x_pixel, 0), (ego_x_pixel, out_size_h), (0, 0, 255), 2)  # 红色垂直线
    cv2.line(bird_view, (0, ego_y_pixel), (out_size_w, ego_y_pixel), (0, 0, 255), 2)  # 红色水平线
    
    return bird_view

def draw_bev_traj(bev=None,bev_h=200*8,bev_w=80*8,trajs=None,world_x=200,world_y=80,color=(0,0,255)):
    # trajs = trajs.astype(np.uint16)
    ego = (bev_w//2,120*8)
    if bev is None:
        bev = np.zeros((bev_h,bev_w,3),dtype=np.uint8) + 255
    traj_x,traj_y = trajs[:,0],trajs[:,1]
    piex_y = ego[1] - traj_x * (bev_h /world_x )
    piex_x = ego[0] - traj_y * (bev_w /world_y )
    # print(piex_x,piex_y)
    for i in range(len(piex_x)):
    # 用不同颜色表示轨迹点
        px, py = piex_x[i],piex_y[i]
        cv2.circle(bev, (int(px),int(py)), 1, color, 2)  # 注意：OpenCV使用(x,y)即(列,行)
    return bev

def draw_traj(bev=None,pos = (0,0),bev_h=200*8,bev_w=80*8,trajs=None,masks=None,world_x=200,world_y=80,color=(0,0,255)):
    # trajs = trajs.astype(np.uint16)
    ego = (bev_w//2,120*8)
    if bev is None:
        bev = np.zeros((bev_h,bev_w,3),dtype=np.uint8) + 255
    traj_x,traj_y = trajs[:,0],trajs[:,1]
    piex_y = ego[1] - traj_x * (bev_h /world_x )
    piex_x = ego[0] - traj_y * (bev_w /world_y )


    # x,y = pos
    # p_y = ego[1] - x * (bev_h /world_x )
    # p_x = ego[0] - y * (bev_w /world_y )
    # cv2.circle(bev, (int(p_x),int(p_y)), 2, (0,0,0), 2)

    # print(piex_x,piex_y)
    for i in range(len(piex_x)):
    # 用不同颜色表示轨迹点
        px, py = piex_x[i],piex_y[i]
        mask = masks[i]
        if mask==0:continue
        cv2.circle(bev, (int(px),int(py)), 1, color, 2)  # 注意：OpenCV使用(x,y)即(列,行)
    return bev

def draw_bev_object(pts, bird_view=None, world_size_x=200, world_size_y=80, 
                    out_size_h=200*8, out_size_w=80*8, ego_position=120,
                    color=(0, 255, 0), thickness=2, draw_edges=True, fill_color=None):
    """
    在BEV图上绘制3D目标框
    
    参数:
        pts: 3D框的角点，可以是以下格式之一：
            - (3, 8) 或 (4, 8): 8个角点的3D坐标（ego坐标系）
            - (8, 3) 或 (8, 4): 8个角点的3D坐标（ego坐标系）
        bird_view: BEV图像，如果为None则创建新的
        world_size_x: 世界坐标系前后方向范围（米），默认200（往前120，往后80）
        world_size_y: 世界坐标系左右方向范围（米），默认80（左右各40）
        out_size_h: BEV图高度（像素），默认1600
        out_size_w: BEV图宽度（像素），默认640
        ego_position: ego在前后方向的位置（米），默认120（往前120米处）
        color: 绘制颜色 (B, G, R)，默认绿色
        thickness: 线条粗细，默认2
        draw_edges: 是否绘制3D框的边，默认True
        fill_color: 填充颜色 (B, G, R)，如果为None则不填充
    
    返回:
        bird_view: 绘制后的BEV图像
        bev_pts_2d: (2, 8) BEV图上的2D坐标
    """
    # 创建或使用现有的BEV图
    if bird_view is None:
        bird_view = np.zeros((out_size_h, out_size_w, 3), dtype=np.uint8) + 255
    
    # 处理输入格式：统一转换为 (3, 8) 格式
    if len(pts.shape) == 2:
        if pts.shape[0] == 3 or pts.shape[0] == 4:
            # (3, 8) 或 (4, 8) 格式
            pts_3d = pts[:3, :]  # 只取前3维（x, y, z），忽略齐次坐标
        elif pts.shape[1] == 3 or pts.shape[1] == 4:
            # (8, 3) 或 (8, 4) 格式，转置
            pts_3d = pts[:, :3].T  # (3, 8)
        else:
            raise ValueError(f"不支持的输入格式: {pts.shape}")
    else:
        raise ValueError(f"不支持的输入维度: {len(pts.shape)}")
    
    # NuScenes坐标系：x=forward(前), y=left(左), z=up(上)
    # BEV图坐标系：x=图像宽度方向(左右), y=图像高度方向(前后)
    # ego在BEV图中的位置：x=640 (左右中心), y=960 (往前120米的位置)
    
    # 提取x, y坐标（忽略z坐标，因为BEV是俯视图）
    pts_x = pts_3d[0, :]  # forward方向（前后）
    pts_y = pts_3d[1, :]  # left方向（左右）
    
    # 转换到BEV坐标系
    # BEV x: 左右方向，ego在中心 (world_size_y / 2.0)
    # BEV y: 前后方向，ego在ego_position位置（往前120米）
    bev_x = (world_size_y / 2.0) - pts_y  # 左右方向，左为正
    bev_y = ego_position - pts_x  # 前后方向，前为正（ego往前120米）
    
    # 缩放到图像尺寸
    bev_x_pixel = bev_x * (out_size_w / world_size_y)
    bev_y_pixel = bev_y * (out_size_h / world_size_x)
    
    # 组合为2D坐标 (2, 8)
    bev_pts_2d = np.array([bev_x_pixel, bev_y_pixel])  # (2, 8)
    
    # 检查点是否在图像范围内
    valid_mask = (bev_pts_2d[0, :] >= 0) & (bev_pts_2d[0, :] < out_size_w) & \
                 (bev_pts_2d[1, :] >= 0) & (bev_pts_2d[1, :] < out_size_h)
    
    if not draw_edges:
        return bird_view, bev_pts_2d,False
    
    # 如果至少有一些点在范围内，才绘制
    if valid_mask.sum() < 2:
        return bird_view, bev_pts_2d,False
    
    # 转换为整数坐标用于绘制
    bev_pts_int = bev_pts_2d.T.astype(int)  # (8, 2)
    
    # 如果需要填充，先填充底面（BEV俯视图中的矩形区域）
    # 底面的4个角点：1(前左下), 3(前右下), 7(后右下), 5(后左下)
    if fill_color is not None:
        bottom_face_indices = [1, 3, 7, 5]  # 底面的4个角点索引
        bottom_face_pts = bev_pts_int[bottom_face_indices]  # (4, 2)
        
        # 检查所有点是否在图像范围内（至少部分可见）
        bottom_valid = np.all((bottom_face_pts >= 0) & (bottom_face_pts < [out_size_w, out_size_h]), axis=1)
        if bottom_valid.sum() >= 3:  # 至少3个点可见才能填充
            # 转换为 (N, 1, 2) 格式，cv2.fillPoly需要的格式
            bottom_face_pts_reshaped = bottom_face_pts.reshape(-1, 1, 2)
            cv2.fillPoly(bird_view, [bottom_face_pts_reshaped], fill_color)
    
    # 定义3D框的12条边（与draw_3d_box_on_image中的定义一致）
    edges = [
        [0, 1], [0, 2], [0, 4],  # 从0出发的3条边
        [1, 3], [1, 5],           # 从1出发的2条边
        [2, 3], [2, 6],           # 从2出发的2条边
        [3, 7],                   # 从3出发的1条边
        [4, 5], [4, 6],           # 从4出发的2条边
        [5, 7], [6, 7]            # 剩余的边
    ]
    
    # 绘制每条边
    for edge in edges:
        pt1_idx, pt2_idx = edge[0], edge[1]
        pt1 = tuple(bev_pts_int[pt1_idx])
        pt2 = tuple(bev_pts_int[pt2_idx])
        
        # 检查点是否在图像范围内
        if (0 <= pt1[0] < out_size_w and 0 <= pt1[1] < out_size_h and
            0 <= pt2[0] < out_size_w and 0 <= pt2[1] < out_size_h):
            cv2.line(bird_view, pt1, pt2, color, thickness)
    
    # 可选：绘制角点（用于调试）
    # for i, pt in enumerate(bev_pts_int):
    #     if 0 <= pt[0] < out_size_w and 0 <= pt[1] < out_size_h:
    #         cv2.circle(bird_view, tuple(pt), 3, (255, 0, 0), -1)
    
    return bird_view, bev_pts_2d,True


def create_overview(image_dict, target_height=300):
    """
    将环视图像和BEV图拼接成一张大图
    :param image_dict: 字典，键为相机名称（如'FrontCam02'），值为图像路径或numpy数组
    :param output_path: 输出图像保存路径
    :param target_height: 统一缩放的高度
    """
    # 读取并缩放图像
    scaled = {}
    names = ['FrontCam02', 'RearCam01', 'SideFrontCam01', 'SideFrontCam02',
             'SideRearCam01', 'SideRearCam02', 'bev']  # 注意顺序与布局对应
    for name in names:
        img = image_dict[name]
        if isinstance(img, str):
            img = cv2.imread(img)
            if img is None:
                raise FileNotFoundError(f"无法读取图像：{img}")
        h, w = img.shape[:2]
        scale = target_height / h
        new_w = int(w * scale)
        resized = cv2.resize(img, (new_w, target_height), interpolation=cv2.INTER_AREA)
        scaled[name] = resized

    # 提取各图像宽度
    front_w = scaled['FrontCam02'].shape[1]
    left_front_w = scaled['SideFrontCam01'].shape[1]
    right_front_w = scaled['SideFrontCam02'].shape[1]
    rear_w = scaled['RearCam01'].shape[1]
    left_rear_w = scaled['SideRearCam01'].shape[1]
    right_rear_w = scaled['SideRearCam02'].shape[1]
    bev_w = scaled['bev'].shape[1]

    # 计算三行的总宽度
    row1_width = front_w
    row2_width = left_front_w + right_front_w + rear_w
    row3_width = left_rear_w + right_rear_w + bev_w

    # 画布宽度取三行的最大值，并预留一些边距
    canvas_width = max(row1_width, row2_width, row3_width)
    canvas_height = target_height * 3  # 三行
    canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

    # 放置第一行：前视居中
    x1 = (canvas_width - front_w) // 2
    y1 = 0
    canvas[y1:y1+target_height, x1:x1+front_w] = scaled['FrontCam02']

    # 放置第二行：左前、右前、后视整体居中
    x2 = (canvas_width - row2_width) // 2
    y2 = target_height
    canvas[y2:y2+target_height, x2:x2+left_front_w] = scaled['SideFrontCam01']
    x2 += left_front_w
    canvas[y2:y2+target_height, x2:x2+right_front_w] = scaled['SideFrontCam02']
    x2 += right_front_w
    canvas[y2:y2+target_height, x2:x2+rear_w] = scaled['RearCam01']

    # 放置第三行：左后、右后、BEV整体居中
    x3 = (canvas_width - row3_width) // 2
    y3 = target_height * 2
    canvas[y3:y3+target_height, x3:x3+left_rear_w] = scaled['SideRearCam01']
    x3 += left_rear_w
    canvas[y3:y3+target_height, x3:x3+right_rear_w] = scaled['SideRearCam02']
    x3 += right_rear_w
    canvas[y3:y3+target_height, x3:x3+bev_w] = scaled['bev']

    # 保存结果
    # cv2.imwrite(output_path, canvas)
    
    return canvas


def vis_dynamic_gt(camera_names,label_path,labels,bboxes,label_masks=None,bboxes_masks=None):
    # label_path = "/workspace/afb5szh-01/models/e2e_model/e2e_dataset_10Hz/20250508143951/label/13375C_20250508144327.000000_LidarFusion.json"
    obj_label = json.load(open(label_path,"r"))
    ego_pose = obj_label["ego_pose"]
    parameters = obj_label["parameters"]
    sensor_paths = obj_label["paths"]
    obj_num = labels.shape[0]
    images = {}
    for camera_name in camera_names:
        frontcam02_params = parameters[camera_name]
        image_height = frontcam02_params["image_height"]
        image_width = frontcam02_params["image_width"]
        T_lidar2camera = frontcam02_params["lidar2camera"]
        dist_coeffs = np.array(frontcam02_params["dist_coeffs"])
        camera_matrix = np.array(frontcam02_params["camera_matrix"])
        T_lidar2camera = np.array(T_lidar2camera)
        img_path = os.path.join(os.path.dirname(label_path).replace("label",camera_name),os.path.basename(sensor_paths[camera_name]))
        if not os.path.exists(img_path):
            img_path = label_path.replace("label",camera_name).replace(".json",".jpg")
        img = cv2.imread(img_path)
        
        for i in range(obj_num):
            label = labels[i]

            if label_masks is not None :
                label_mask = label_masks[i]
            else:
                label_mask = 1
            if label_mask == 0:
                color = (0,0,0)
            else:
                color = class_colors[int(label)]
            
            x, y, z, width, length, height, yaw, vx,vy,vz = bboxes[i,:10]
            pitch,roll = 0,0

            box3d_lidar = create_3d_bbox_corners([x,y,z],[length,width,height],[roll,pitch,yaw])
            box3d_camera = project_lidar_to_camera(box3d_lidar,T_lidar2camera)
            points_2d,mask = project_camera_to_image( box3d_camera,camera_matrix,dist_coeffs,image_height,image_width)

            img = draw_poly_image(img,points_2d,mask,color=color)
        images[camera_name] = img
        # cv2.imwrite(f"{camera_name}.jpg", img)
        # print(camera_name,img.shape)
    # print("ok ", label_path)

    bev_img = None  # 第一次调用时会自动创建
    bev_img = draw_bev_grid(
        bev_img,
        world_size_x=200,
        world_size_y=80,
        out_size_h=200*8,
        out_size_w=80*8,
        ego_position=120,
        grid_interval=10,  # 每隔10米画一条线
        color=(200, 200, 200),  # 浅灰色网格线
        thickness=1
    )
    for i in range(obj_num):
        label = labels[i]
        if label_masks is not None :
            label_mask = label_masks[i]
        else:
            label_mask = 1
        if label_mask == 0:
            color = (0,0,0)
        else:
            color = class_colors[int(label)]
        
        x, y, z, width, length, height, yaw, vx,vy,vz = bboxes[i,:10]
        pitch,roll = 0,0
        box3d_lidar = create_3d_bbox_corners([x,y,z],[length,width,height],[roll,pitch,yaw])

        bev_img, _,in_bev = draw_bev_object(
            box3d_lidar,  # (4, 8) 自车坐标系下的角点
            bird_view=bev_img,
            world_size_x=200,  # 前后200米
            world_size_y=80,  # 左右80米
            out_size_h=200*8,  # BEV图高度1600
            out_size_w=80*8,   # BEV图宽度640
            ego_position=120,  # ego往前120米的位置
            color=class_colors[int(label)],  # 使用分配的颜色
            thickness=2,
            fill_color=color  # 根据visibility_token填充
        ) 
        if traj_mask is not None:
            if label_mask != 0:
                dt = 0.1
                his_len = 5
                traj_v = bboxes[i,10:].reshape(-1,2)
                traj_mask = bboxes_masks[i,10:].reshape(-1,2)
                pos = np.array([x,y])[None]
                fur_traj = np.cumsum(traj_v[his_len:] * dt,0) + pos
                history_v = traj_v[:his_len]  
                history_v_rev = history_v[::-1]
                history_offset = -np.cumsum(history_v_rev * dt, axis=0)
                history_coords_rev = pos + history_offset 
                his_traj = history_coords_rev[::-1]
                reconstructed = np.vstack([his_traj, fur_traj])
                bev_img = draw_traj(bev=bev_img,pos={x, y},bev_h=200*8,bev_w=80*8,trajs=reconstructed,masks=traj_mask[:,0],world_x=200,world_y=80,color=class_colors[int(label)])
            # print(1)
        # 绘制轨迹
        cv2.imwrite("bev.jpg",bev_img)

    images["bev"] = bev_img
    # cv2.imwrite(f"bev.jpg", bev_img)
    # print("bev",bev_img.shape)
    # print("done")
    canvas = create_overview(images, target_height=300)
    # for k,v in images.items():
    #     cv2.imwrite(f"{k}.jpg", v)
    return canvas,images


def vis_dynamic_pred(camera_names, label_path, boxes_3d, scores_3d, labels_3d,
                     score_thresh=0.3):
    """
    可视化 Sparse4D 检测分支的预测结果。

    参数:
        camera_names: 相机名称列表，通常与 config["camera_names"] 一致
        label_path: 当前帧对应的 label json 路径（用于读取相机参数和图像）
        boxes_3d: 预测 3D 框，形状 (N, 10)，格式 [x,y,z,w,l,h,yaw,vx,vy,vz]
        scores_3d: 预测得分，形状 (N,)
        labels_3d: 预测类别 id，形状 (N,)
        score_thresh: 置信度阈值，低于该值的框将被过滤

    返回:
        canvas: 拼接了多相机视角和 BEV 的可视化图像 (H, W, 3)，BGR 格式
    """
    

    boxes_3d = np.asarray(boxes_3d)
    scores_3d = np.asarray(scores_3d)
    labels_3d = np.asarray(labels_3d)

    keep = scores_3d >= score_thresh
    # if keep.sum() == 0:
    #     return None,None

    boxes_keep = boxes_3d[keep]
    labels_keep = labels_3d[keep]

    canvas,images = vis_dynamic_gt(
        camera_names=camera_names,
        label_path=label_path,
        labels=labels_keep,
        bboxes=boxes_keep,
        label_masks=None,
        bboxes_masks=None,
    )
    return canvas,images

def vis_static_gt(camera_names,label_path,labels,pts):
    obj_label = json.load(open(label_path,"r"))
    ego_pose = obj_label["ego_pose"]
    parameters = obj_label["parameters"]
    sensor_paths = obj_label["paths"]
    obj_num = labels.shape[0]
    pt = pts[:,0]
    images = {}
    for camera_name in camera_names:
        frontcam02_params = parameters[camera_name]
        image_height = frontcam02_params["image_height"]
        image_width = frontcam02_params["image_width"]
        T_lidar2camera = frontcam02_params["lidar2camera"]
        dist_coeffs = np.array(frontcam02_params["dist_coeffs"])
        camera_matrix = np.array(frontcam02_params["camera_matrix"])
        T_lidar2camera = np.array(T_lidar2camera)
        img_path = os.path.join(os.path.dirname(label_path).replace("label",camera_name),os.path.basename(sensor_paths[camera_name]))
        if not os.path.exists(img_path):
            img_path = label_path.replace("label",camera_name).replace(".json",".jpg")
        img = cv2.imread(img_path)
        images[camera_name] = img

    # bev_img_gt = None
    # bev_img_gt = draw_bev_grid(
    #     bev_img_gt,
    #     world_size_x=200,
    #     world_size_y=80,
    #     out_size_h=200*8,
    #     out_size_w=80*8,
    #     ego_position=120,
    #     grid_interval=10,  # 每隔10米画一条线
    #     color=(200, 200, 200),  # 浅灰色网格线
    #     thickness=1
    # )
    # groups = obj_label["groups"]
    # for _, grp in groups.items():
    #     gtype = grp.get("type", "")
    #     props = grp.get("properties", {})

    #     # 映射到 3 类：divider / ped_crossing / boundary
    #     cls_id = None
    #     if gtype in ["lane_line", "real_lane_line", "imaginary_lane_line", "lane_center_line"]:
    #         cls_id = 0  # divider
    #     elif gtype == "road_marker_line":
    #         # 仅把 category==3 (斑马线) 当作人行横道，其它暂不作为 map GT
    #         cat = str(props.get("category", "-1"))
    #         if cat == "3":
    #             cls_id = 1  # ped_crossing
    #     elif gtype in ["road_edge", "non_drivable_area"]:
    #         cls_id = 2  # boundary / non-drivable
    #     else:
    #         continue

    #     if cls_id is None:
    #         continue

    #     color = class_colors[int(cls_id)]
    #     pts1 = []
    #     objects = grp.get("objects", [])
    #     for obj in objects:
    #         points = obj["points"]
    #         for point in points:
    #             pts1.append([point["x"],point["y"]])
    #     xyz = np.array(pts1)
    #     bev_img_gt = draw_bev_traj(bev=bev_img_gt,bev_h=200*8,bev_w=80*8,trajs=xyz,world_x=200,world_y=80,color=color)



    bev_img = None
    bev_img = draw_bev_grid(
        bev_img,
        world_size_x=200,
        world_size_y=80,
        out_size_h=200*8,
        out_size_w=80*8,
        ego_position=120,
        grid_interval=10,  # 每隔10米画一条线
        color=(200, 200, 200),  # 浅灰色网格线
        thickness=1
    )
    for i in range(obj_num):
        label = labels[i]
        p = pt[i]
        color = class_colors[int(label)]
        bev_img = draw_bev_traj(bev=bev_img,bev_h=200*8,bev_w=80*8,trajs=p,world_x=200,world_y=80,color=color)
    images["bev"] = bev_img
    canvas = create_overview(images, target_height=300)

    # # cv2.imwrite("img.jpg",bev_img)
    # cv2.imwrite("canvas.jpg",canvas)
    return canvas,images