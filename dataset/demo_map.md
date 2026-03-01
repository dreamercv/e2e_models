```javascript
{
  "schema_version": "openlane_rb",
  "collect_metadata": {
    "vin": "11174C",
    "collected_time": 1679573801
  },
  "frame_properties": {
    "valid": "1",                             // 该图片是否有效，既是否可标注, 1 有效, 0 无效
    "auto_label": "1",                        // 是否自动标注: 1 是, 0 否
    "align_0_cameras": ["FrontCam01", "FrontCam02", "SurCam02"],                     // 完全贴合摄像头
    "align_1_cameras": ["SurCam01", "RearCam01", "SideRearCam01", "SideRearCam02"],  // 较好贴合摄像头
    "align_in_global": "0",                   // 全图 2D 贴合度: 0 完全贴合, 1 较好贴合
    "tele_align_level": "1",                  // 远焦摄像头投影贴合级别: 0 完全贴合, 1 较好贴合, 2 较差贴合, 3 不可接受
    "ground_qualified": "1",                  // 地面元素是否合格, 1 合格，0 不合格
    "ground_unqualified_reasons": ["1"],      // 不合格的原因: 1 不贴合, 2 属性错误，这里需和 GData 对齐原因类型
    "air_qualified": "0",                     // 空中元素是否合格, 1 合格，0 不合格
    "air_unqualified_reasons": ["1"],         // 不合格的原因: 1 不贴合, 2 属性错误，这里需和 GData 对齐原因类型
  },
  "label_metadata": {
    "label_project": "eng_ad_static_3d_t",        // 值有: w3_ad_static_3d_t GData 和 人工,  eng_ad_static_3d_t_tx 人工 和 腾讯地图 reloc
    "label_rule_version": "1.1.0",                // 值有: 1.0.0, 1.1.0
    "revised_version": "1.0",                     // 数据修订版本号，每次数据重刷新增版本号
    "supplier": "xxx",
    "data_id": "161d9e958162bb5655b174a5e4628a58",
    "delivery_date": "2024-08-16",
    "seq_id": "12954C_20240726153535.800000",     // clip id
    "frame_num": 24                               // 当前帧在 clip 里的序号
  },
  "collected_frames": [
    {
      "resources": [
          {
              "sensor": "FrontCam01",                         // 传感器的名字
              "uri": "/datasets/xlabel/xxx/xxx.jpeg"        // 路径
          },
          {
              "sensor": "FrontCam02",                         // 传感器的名字
              "uri": "/datasets/xlabel/xxx/xxx.jpeg"        // 路径
          },
          {
              "sensor": "SideFrontCam01",                         // 传感器的名字
              "uri": "/datasets/xlabel/xxx/xxx.jpeg"        // 路径
          },
          {
              "sensor": "SideFrontCam02",                         // 传感器的名字
              "uri": "/datasets/xlabel/xxx/xxx.jpeg"        // 路径
          },
          {
              "sensor": "RearCam01",                         // 传感器的名字
              "uri": "/datasets/xlabel/xxx/xxx.jpeg"        // 路径
          },
          {
              "sensor": "SurCam01",                         // 传感器的名字
              "uri": "/datasets/xlabel/xxx/xxx.jpeg"        // 路径
          },
          {
              "sensor": "SurCam02",                         // 传感器的名字
              "uri": "/datasets/xlabel/xxx/xxx.jpeg"        // 路径
          },
          {
              "sensor": "SurCam03",                         // 传感器的名字
              "uri": "/datasets/xlabel/xxx/xxx.jpeg"        // 路径
          },
          {
              "sensor": "SurCam04",                         // 传感器的名字
              "uri": "/datasets/xlabel/xxx/xxx.jpeg"        // 路径
          },
          {
              "sensor": "LidarFusion",                      // 传感器的名字
              "uri": "/datasets/xlabel/xxx/xxx.pcd"         // 路径
          },
      ],
      "frame_info_uri": "/datasets/xlabel/xxx/xxx.json",
      "collected_time": 1679573801000
    }
  ],
   "associations": {
    "centerline_topology": "-1",           // 在 xLabel 侧不再计算该矩阵
    "vertical_topology": [
      {
          "lane_center_line": "1",         // 当前线是车道中心线，每个对象中，lane_center_line 与 lane_connection_line 只能有一个
          "lane_connection_line": "2",     // 当前线是车道连接关系线, 每个对象中，lane_center_line 与 lane_connection_line 只能有一个
          "predecessor": ["3"],            // 前驱，可能有多个
          "successor": ["4"]               // 后继，可能有多个
      }
    ],
    "horizontal_topology": [
      {
          "lane_center_line": "1",         // 该连接关系只适用于车道中心线
          "left_connection": ["3"],        // 左连接，可能有多个
          "right_connection": ["4"]        // 右连接，可能有多个
      }
    ],
    "lane_segment": [
      {
          "lane_center_line": "1",         // 当前线是车道中心线，每个对象中，lane_center_line 与 lane_connection_line 只能有一个
          "lane_connection_line": "2",     // 当前线是车道连接关系线，每个对象中，lane_center_line 与 lane_connection_line 只能有一个
          "left_lane_line": ["3"],         // 左车道线，可能有多个
          "right_lane_line": ["4"]         // 右车道线，可能有多个
      }
    ],
    "binding": [
      {
          "lane_center_line": "1",         // 当前线是车道中心线，每个对象中，lane_center_line 与 lane_connection_line 只能有一个
          "lane_connection_line": "2",     // 当前线是车道连接关系线，每个对象中，lane_center_line 与 lane_connection_line 只能有一个
          "traffic_element": ["3"]         // 所绑定的禁停区或停止线的
      }
    ],
    "double_line": [
      {
          "line_one": ["1"],               // 双线其中一条线
          "line_two": ["2"]                // 双线另一条线
      }
    ],
    "lane_line_endpoint": [
      {
        "endpoint": "3",                   // 分流, 和上面保持风格一致，原先的 track_id 字段改为 endpoint
        "in": ["1"],
        "out": ["2", "4"]
      },
      {
        "endpoint": "33",                  // 汇流
        "in": ["1", "2"],
        "out": ["4"]
      }
    ]
  },
  "group_type_list": {
    "lane_line": ["12"],
    "real_lane_line": ["1"],
    "imaginary_lane_line": ["2"],
    "lane_line_endpoint": ["3"],
    "lane_center_line": ["4"],
    "lane_connection_line": ["5"],
    "non_drivable_area": ["6"],
    "intersection": ["7"],
    "road_edge": ["8"],
    "road_marker_arrow": ["9"],
    "road_marker_line": ["1"],
    "road_marker_sign": ["11"]
  },
  "groups": {
    "1": {
      "type": "real_lane_line",         // 真实车道线, 不需要交付 2D 人工修改结果
      "properties": {
        "track_id": "1",                // 物体 track id, 字符串类型
        "category": "0",                // 类别: 0 虚线, 1 实线, 2 暂留类别, 3 粗虚线, 4 停车线, 5 导流区线
        "dual_attr": "00",              // 双线属性: 00 双虚线, 11 双实线, 01 左虚右实, 10 左实右虚, -1 无效（非双线）
        "color": "1",                   // 颜色: 1 白色, 2 黄色, 10 其他
        "function": "0",                // 功能: 0 常规, 1 待转区, 2 纵向减速线(鱼骨线), 3 潮汐车道线, 4 可变车道线, 5 停止线, 6 引导线
        "non_flush": "0",               // 非平齐线: 0 否, 1 是
      },
      "objects": [
        {
          "sensor": "LidarFusion",      // 传感器编号
          "type": "real_lane_line_3d",
          "geometry": "line_3d",
          "points": [{                  // 每个点添加一个可见性属性
            "x": 473.4729,
            "y": 611.3537,
            "z": 0.3538,
            "properties": {
              "v_11": ["F1", "F2", "S2"],         // 可见摄像头
              "v_00": "-1",                       // 视角不可见摄像头, 去除该属性
              "v_21": ["S3", "S4", "SF1", "SF2"], // 自车遮挡不可见摄像头
              "v_22": ["S3", "S4", "SF1", "SF2"], // 其他遮挡不可见摄像头
              "v_bev": "-1",                      // 该字段暂时不用, 保留字段
              "v": ['0', '1', '0', "1", '0', '1'] // 该点在各个摄像头里是否存在: 1 是, 0 否. 摄像头排列顺序: FrontCam01, FrontCam02, RearCam01, SideFrontCam01, SideFrontCam02, SideRearCam01, SideRearCam02, SurCam01, SurCam02, SurCam03, SurCam04
            }
          }, {
            "x": 732.6589,
            "y": 612.7096,
            "z": 0.3538,
            "properties": {
              "v_11": ["F1", "F2", "S2"],         // 可见摄像头
              "v_00": "-1",                       // 视角不可见摄像头, 去除该属性
              "v_21": ["S3", "S4", "SF1", "SF2"], // 自车遮挡不可见摄像头
              "v_22": ["S3", "S4", "SF1", "SF2"], // 其他遮挡不可见摄像头
              "v_bev": "-1",                      // 该字段暂时不用, 保留字段
              "v": ['0', '1', '0', "1", '0', '1'] // 该点在各个摄像头里是否存在: 1 是, 0 否. 摄像头排列顺序: FrontCam01, FrontCam02, RearCam01, SideFrontCam01, SideFrontCam02, SideRearCam01, SideRearCam02, SurCam01, SurCam02, SurCam03, SurCam04
            }
          }, {
            "x": 1032.6589,
            "y": 612.7096,
            "z": 0.3538,
            "properties": {
              "v_11": ["F1", "F2", "S2"],         // 可见摄像头
              "v_00": "-1",                       // 视角不可见摄像头, 去除该属性
              "v_21": ["S3", "S4", "SF1", "SF2"], // 自车遮挡不可见摄像头
              "v_22": ["S3", "S4", "SF1", "SF2"], // 其他遮挡不可见摄像头
              "v_bev": "-1",                      // 该字段暂时不用, 保留字段
              "v": ['0', '1', '0', "1", '0', '1'] // 该点在各个摄像头里是否存在: 1 是, 0 否. 摄像头排列顺序: FrontCam01, FrontCam02, RearCam01, SideFrontCam01, SideFrontCam02, SideRearCam01, SideRearCam02, SurCam01, SurCam02, SurCam03, SurCam04
            }
          }]
        }
      ]
    },
    "2": {
      "type": "imaginary_lane_line",          // 假想车道线, 不需要交付 2D 人工修改结果
      "properties": {
        "track_id": "2",                      // 物体 track id, 字符串类型
        "category": "1"                       // 类型: 1 变道假想线, 2 标线假想线, 3 边界假想线, 4 路口假想线, 5 假想线(非路口)
      },
      "objects": [
        {
          "sensor": "LidarFusion",            // 传感器编号
          "type": "imaginary_lane_line_3d",
          "geometry": "line_3d",
          "points": [{
            "x": 473.4729,
            "y": 611.3537,
            "z": 0.3538
          }, {
            "x": 732.6589,
            "y": 612.7096,
            "z": 0.3538
          }, {
            "x": 1032.6589,
            "y": 612.7096,
            "z": 0.3538
          }]
        }
      ]
    },
    "3": {
      "type": "lane_line_endpoint",       // 车道线端点, 不需要交付 2D 人工修改结果
      "properties": {
        "track_id": "3",                  // 物体 track id, 字符串类型
        "category": "1"                   // 类型: 1 汇流, 2 分流
      },
      "objects": [
        {
          "sensor": "LidarFusion",            // 传感器编号
          "type": "lane_line_endpoint_3d",
          "geometry": "point_3d",             // 3D 点
          "points": [{
            "x": 473.4729,
            "y": 611.3537,
            "z": 0.3537
          }]
        }
      ]
    },
    "4": {
      "type": "lane_center_line",     // 车道中心线, 不需要交付 2D 人工修改结果
      "properties": {
        "track_id": "4",              // 物体 track id, 字符串类型
        "category": "0"               // 类型: 0 普通车道, 1 可变车道, 2 减速车道, 3 公交车道, 4 潮汐车道, 5 待转区车道, 6 非机动车道, 7 停车位车道, 10 其他车道,
                                      // 21 应急车道, 22 停靠车道, 23 分流加速车道, 24 汇流减速车道
                                      // 31 左转专用车道, 32 右转专用车道, 33 左掉头专用车道, 34 右掉头专用车道
      },
      "extra_properties": {
        "L_TYPE": "1",                // 车道类型: 0 虚拟车道, 1 机动车道(常规车道), 2 非机动车道, 3 机非混合车道, 4 摩托车道, 5 人行道, 6 暂留类别, 7 暂留类别, 8 可变导向车道(可变车道), 9 HOV 车道, 10 右侧加速车道, 11 右侧减速车道, 12 左侧加速车道, 13 左侧减速车道, 14 变速车道(加减速 混合车道)(复合车道), 15 收费站车道, 16 检查站车道, 17 公交专用车道(公交车道), 18 爬坡车道, 19 暂留类别, 20 暂留类别, 21 应急车道, 22 紧急停车带, 23 停车车道, 24 危险品专用车道, 25 海关监管车道, 26 避险车道引道, 27 错车道, 28 非行驶区域, 29 借道区, 30 有轨电车车道, 31 ETC车道, 32 公交港湾车道, 33 特殊车辆专用车道, 34 暂留类别, 35 暂留类别, 36 暂留类别, 37 暂留类别, 38 暂留类别, 39 暂留类别, 40 暂留类别, 41 暂留类别, 42 暂留类别, 43 暂留类别, 44 暂留类别, 45 暂留类别, 46 暂留类别, 47 暂留类别, 48 暂留类别, 49 暂留类别, 99 其它车
        "T_DRCT": "-1"                // 车道引导方向: 0 直行, 1 左转, 2 右转, 3 向左合流 4 向右合流 5 掉头, 
        "L_NO": "-1",                 // 车道编号, 数字化方向从左到右 1)上下线分离:-1，-2... 2)非上下线分离:...2，1，- 1，-2...
        "DIRECTION": "1"              // 道路通行方向: 1 双向 2 正向 3 逆向
      },
      "objects": [
        {
          "sensor": "LidarFusion",       // 传感器编号
          "type": "lane_center_line_3d",
          "geometry": "line_3d",
          "points": [
            {
              "x": 473.4729,
              "y": 611.3537,
              "z": 0.3537
            }, {
              "x": 732.6589,
              "y": 612.7096,
              "z": 0.3537
            }, {
              "x": 1002.6589,
              "y": 612.7096,
              "z": 0.3537
            }
          ]
        }
      ]
    },
    "5": {
      "type": "lane_connection_line",     // 路口连接关系线, 不需要交付 2D 人工修改结果
      "properties": {
        "track_id": "5",                  // 物体 track id, 字符串类型
        "direction": "1"                  // 方向: 1 左转, 2 右转, 3 直行, 4 掉头
      },
      "extra_properties": {
        "L_TYPE": "",                     // 车道类型: 0 虚拟车道, 1 机动车道(常规车道), 2 非机动车道, 3 机非混合车道, 4 摩托车道, 5 人行道, 6 暂留类别, 7 暂留类别, 8 可变导向车道(可变车道), 9 HOV 车道, 10 右侧加速车道, 11 右侧减速车道, 12 左侧加速车道, 13 左侧减速车道, 14 变速车道(加减速 混合车道)(复合车道), 15 收费站车道, 16 检查站车道, 17 公交专用车道(公交车道), 18 爬坡车道, 19 暂留类别, 20 暂留类别, 21 应急车道, 22 紧急停车带, 23 停车车道, 24 危险品专用车道, 25 海关监管车道, 26 避险车道引道, 27 错车道, 28 非行驶区域, 29 借道区, 30 有轨电车车道, 31 ETC车道, 32 公交港湾车道, 33 特殊车辆专用车道, 34 暂留类别, 35 暂留类别, 36 暂留类别, 37 暂留类别, 38 暂留类别, 39 暂留类别, 40 暂留类别, 41 暂留类别, 42 暂留类别, 43 暂留类别, 44 暂留类别, 45 暂留类别, 46 暂留类别, 47 暂留类别, 48 暂留类别, 49 暂留类别, 99 其它车
        "T_DRCT": "-1"                // 车道引导方向: 0 直行, 1 左转, 2 右转, 3 向左合流 4 向右合流 5 掉头, 
        "L_NO": "-1"                  // 车道编号, 数字化方向从左到右 1)上下线分离:-1，-2... 2)非上下线分离:...2，1，- 1，-2...
        "DIRECTION": "1"              // 道路通行方向: 1 双向 2 正向 3 逆向
      },
      "objects": [
        {
          "sensor": "LidarFusion",        // 传感器编号
          "type": "lane_connection_line_3d",
          "geometry": "line_3d",
          "points": [{
            "x": 473.4729,
            "y": 611.3537,
            "z": 0.3537
          }, {
            "x": 732.6589,
            "y": 612.7096,
            "z": 0.3537
          }, {
            "x": 1032.6589,
            "y": 612.7096,
            "z": 0.3537
          }]
        }
      ]
    },
    "6": {
      "type": "non_drivable_area",        // 不可行驶区域, 不需要交付 2D 人工修改结果
      "properties": {
        "track_id": "6",                  // 物体 track id, 字符串类型
        "category": "1"                   // 类型: 1 停车位区域, 2 非机动车道, 3 导流区域, 4 其他不可通行
      },
      "objects": [
        {
          "sensor": "LidarFusion",        // 传感器编号
          "type": "non_drivable_area_3d",
          "geometry": "polygon_3d",       // 3D 多边形, 首尾点坐标不相同
          "points": [{
            "x": 473.4729,
            "y": 611.3537,
            "z": 0.3537
          }, {
            "x": 732.6589,
            "y": 612.7096,
            "z": 0.3537
          }, {
            "x": 1032.6589,
            "y": 612.7096,
            "z": 0.3537
          }]
        }
      ]
    },
    "7": {
      "type": "intersection",             // 路口, 不需要交付 2D 人工修改结果
      "properties": {
        "track_id": "7",                  // 物体 track id, 字符串类型
      },
      "extra_properties": {
        "TYPE": "3"                       // 路口类型, 透传 sq3 type: 0 未分类, 1 其它, 2 普通路口, 3 掉头口, 4 环岛, 5 主辅路出口, 6 主辅路入口, 7 主辅路出入口, 8 大门口
      }
      "objects": [
        {
          "sensor": "LidarFusion",        // 传感器编号
          "type": "intersection_3d",
          "geometry": "polygon_3d",       // 3D 多边形, 首尾点坐标不相同
          "properties": {
              "v_11": ["F1", "F2", "S2"],         // 可见摄像头
              "v_00": "-1",                       // 视角不可见摄像头, 去除该属性
              "v_21": ["S3", "S4", "SF1", "SF2"], // 自车遮挡不可见摄像头
              "v_22": ["S3", "S4", "SF1", "SF2"], // 其他遮挡不可见摄像头
              "v_bev": "-1",                      // 该字段暂时不用, 保留字段
              "v": ['0', '1', '0', "1", '0', '1'] // 该点在各个摄像头里是否存在: 1 是, 0 否. 摄像头排列顺序: FrontCam01, FrontCam02, RearCam01, SideFrontCam01, SideFrontCam02, SideRearCam01, SideRearCam02, SurCam01, SurCam02, SurCam03, SurCam04
          },
          "points": [{
            "x": 473.4729,
            "y": 611.3537,
            "z": 0.3537,
            "properties": {
              "v_11": ["F1", "F2", "S2"],         // 可见摄像头
              "v_00": "-1",                       // 视角不可见摄像头, 去除该属性
              "v_21": ["S3", "S4", "SF1", "SF2"], // 自车遮挡不可见摄像头
              "v_22": ["S3", "S4", "SF1", "SF2"], // 其他遮挡不可见摄像头
              "v_bev": "-1",                      // 该字段暂时不用, 保留字段
              "v": ['0', '1', '0', "1", '0', '1'] // 该点在各个摄像头里是否存在: 1 是, 0 否. 摄像头排列顺序: FrontCam01, FrontCam02, RearCam01, SideFrontCam01, SideFrontCam02, SideRearCam01, SideRearCam02, SurCam01, SurCam02, SurCam03, SurCam04
            }
          }, {
            "x": 732.6589,
            "y": 612.7096,
            "z": 0.3537,
            "properties": {
              "v_11": ["F1", "F2", "S2"],         // 可见摄像头
              "v_00": "-1",                       // 视角不可见摄像头, 去除该属性
              "v_21": ["S3", "S4", "SF1", "SF2"], // 自车遮挡不可见摄像头
              "v_22": ["S3", "S4", "SF1", "SF2"], // 其他遮挡不可见摄像头
              "v_bev": "-1",                      // 该字段暂时不用, 保留字段
              "v": ['0', '1', '0', "1", '0', '1'] // 该点在各个摄像头里是否存在: 1 是, 0 否. 摄像头排列顺序: FrontCam01, FrontCam02, RearCam01, SideFrontCam01, SideFrontCam02, SideRearCam01, SideRearCam02, SurCam01, SurCam02, SurCam03, SurCam04
            }
          }, {
            "x": 1032.6589,
            "y": 612.7096,
            "z": 0.3537,
            "properties": {
              "v_11": ["F1", "F2", "S2"],         // 可见摄像头
              "v_00": "-1",                       // 视角不可见摄像头, 去除该属性
              "v_21": ["S3", "S4", "SF1", "SF2"], // 自车遮挡不可见摄像头
              "v_22": ["S3", "S4", "SF1", "SF2"], // 其他遮挡不可见摄像头
              "v_bev": "-1",                      // 该字段暂时不用, 保留字段
              "v": ['0', '1', '0', "1", '0', '1'] // 该点在各个摄像头里是否存在: 1 是, 0 否. 摄像头排列顺序: FrontCam01, FrontCam02, RearCam01, SideFrontCam01, SideFrontCam02, SideRearCam01, SideRearCam02, SurCam01, SurCam02, SurCam03, SurCam04
            }
          }, {
            "x": 1032.6589,
            "y": 612.7096,
            "z": 0.3537,
            "properties": {
              "v_11": ["F1", "F2", "S2"],         // 可见摄像头
              "v_00": "-1",                       // 视角不可见摄像头, 去除该属性
              "v_21": ["S3", "S4", "SF1", "SF2"], // 自车遮挡不可见摄像头
              "v_22": ["S3", "S4", "SF1", "SF2"], // 其他遮挡不可见摄像头
              "v_bev": "-1",                      // 该字段暂时不用, 保留字段
              "v": ['0', '1', '0', "1", '0', '1'] // 该点在各个摄像头里是否存在: 1 是, 0 否. 摄像头排列顺序: FrontCam01, FrontCam02, RearCam01, SideFrontCam01, SideFrontCam02, SideRearCam01, SideRearCam02, SurCam01, SurCam02, SurCam03, SurCam04
            }
          }, {
            "x": 1032.6589,
            "y": 612.7096,
            "z": 0.3537,
            "properties": {
              "v_11": ["F1", "F2", "S2"],         // 可见摄像头
              "v_00": "-1",                       // 视角不可见摄像头, 去除该属性
              "v_21": ["S3", "S4", "SF1", "SF2"], // 自车遮挡不可见摄像头
              "v_22": ["S3", "S4", "SF1", "SF2"], // 其他遮挡不可见摄像头
              "v_bev": "-1",                      // 该字段暂时不用, 保留字段
              "v": ['0', '1', '0', "1", '0', '1'] // 该点在各个摄像头里是否存在: 1 是, 0 否. 摄像头排列顺序: FrontCam01, FrontCam02, RearCam01, SideFrontCam01, SideFrontCam02, SideRearCam01, SideRearCam02, SurCam01, SurCam02, SurCam03, SurCam04
            }
          }]
        }
      ]
    },
    "8": {
      "type": "road_edge",        // 车道边线, 不需要交付 2D 人工修改结果
      "properties": {
        "track_id": "8",          // 物体 track id, 字符串类型
        "category": "1",          // 类别: 1 花圃花坛路沿(只有 GData 有), 2 路边石路沿, 3 墙体路沿, 4 断崖式路沿, 5 平路沿, 6 栅栏路沿, 7 隔离墩路沿, 8 成排交通设施, 9 暂留类别, 10 其他 11 无隔离设施 12 物理隔离 13防护栏 14防护网 15杆 16 其它不可跨越防护设施 17 其它可跨越防护设施 18.自然边界
      },
      "objects": [
        {
          "sensor": "LidarFusion",        // 传感器编号
          "type": "road_edge_3d",
          "geometry": "line_3d",          // 3D 线
          "points": [{
            "x": 473.4729,
            "y": 611.3537,
            "z": 0.3537,
            "properties": {
              "v_11": ["F1", "F2", "S2"],         // 可见摄像头
              "v_00": "-1",                       // 视角不可见摄像头, 去除该属性
              "v_21": ["S3", "S4", "SF1", "SF2"], // 自车遮挡不可见摄像头
              "v_22": ["S3", "S4", "SF1", "SF2"], // 其他遮挡不可见摄像头
              "v_bev": "-1",                      // 该字段暂时不用, 保留字段
              "v": ['0', '1', '0', "1", '0', '1'] // 该点在各个摄像头里是否存在: 1 是, 0 否. 摄像头排列顺序: FrontCam01, FrontCam02, RearCam01, SideFrontCam01, SideFrontCam02, SideRearCam01, SideRearCam02, SurCam01, SurCam02, SurCam03, SurCam04
            }
          }, {
            "x": 732.6589,
            "y": 612.7096,
            "z": 0.3537,
            "properties": {
              "v_11": ["F1", "F2", "S2"],         // 可见摄像头
              "v_00": "-1",                       // 视角不可见摄像头, 去除该属性
              "v_21": ["S3", "S4", "SF1", "SF2"], // 自车遮挡不可见摄像头
              "v_22": ["S3", "S4", "SF1", "SF2"], // 其他遮挡不可见摄像头
              "v_bev": "-1",                      // 该字段暂时不用, 保留字段
              "v": ['0', '1', '0', "1", '0', '1'] // 该点在各个摄像头里是否存在: 1 是, 0 否. 摄像头排列顺序: FrontCam01, FrontCam02, RearCam01, SideFrontCam01, SideFrontCam02, SideRearCam01, SideRearCam02, SurCam01, SurCam02, SurCam03, SurCam04
            }
          }, {
            "x": 1032.6589,
            "y": 612.7096,
            "z": 0.3537,
            "properties": {
              "v_11": ["F1", "F2", "S2"],         // 可见摄像头
              "v_00": "-1",                       // 视角不可见摄像头, 去除该属性
              "v_21": ["S3", "S4", "SF1", "SF2"], // 自车遮挡不可见摄像头
              "v_22": ["S3", "S4", "SF1", "SF2"], // 其他遮挡不可见摄像头
              "v_bev": "-1",                      // 该字段暂时不用, 保留字段
              "v": ['0', '1', '0', "1", '0', '1'] // 该点在各个摄像头里是否存在: 1 是, 0 否. 摄像头排列顺序: FrontCam01, FrontCam02, RearCam01, SideFrontCam01, SideFrontCam02, SideRearCam01, SideRearCam02, SurCam01, SurCam02, SurCam03, SurCam04
            }
          }]
        }
      ]
    },
    "9": {
      "type": "road_marker_arrow",        // 地面箭头, 不需要交付 2D 人工修改结果
      "properties": {
        "track_id": "9",                  // 物体 track id, 字符串类型
        "category": "1",                  // 类型: 1 直行, 2 左转, 3 右转, 4 直行或左转, 5 直行或右转, 6 掉头, 7 直行或掉头, 8 左转或掉头, 9 左转或右转, 10 右转或掉头, 11 直行或左转或右转, 12 有左弯或需向左合流, 13 有右弯或需向右合流, 14 禁止路标, 15 直行或左转或掉头, 16 直行或右转或掉头, 17 左转或右转或掉头, 18 禁止掉头, 19 禁止左转, 20 禁止右转。如果还有其他，继续在回标时遇到后继续增加
        "color": "1",                     // 颜色: 1 白色, 2 黄色, 3 其他
        "standard": "0",                  // 是否国标: 0 否, 1 是
      },
      "objects": [
        {
          "sensor": "LidarFusion",                // 传感器编号
          "type": "road_marker_arrow_3d",
          "geometry": "polygon_3d",               // 3D 框, 有且仅有 4 个点，且四个角度为 90 度, 点的 z 值差需要在 10cm 内，四个角度容许 +-1 度的误差
          "points": [{
            "x": 473.4729,
            "y": 611.3537,
            "z": 0.3537,
            "properties": {
              "v_11": ["F1", "F2", "S2"],         // 可见摄像头
              "v_00": "-1",                       // 视角不可见摄像头, 去除该属性
              "v_21": ["S3", "S4", "SF1", "SF2"], // 自车遮挡不可见摄像头
              "v_22": ["S3", "S4", "SF1", "SF2"], // 其他遮挡不可见摄像头
              "v_bev": "-1",                      // 该字段暂时不用, 保留字段
              "v": ['0', '1', '0', "1", '0', '1'] // 该点在各个摄像头里是否存在: 1 是, 0 否. 摄像头排列顺序: FrontCam01, FrontCam02, RearCam01, SideFrontCam01, SideFrontCam02, SideRearCam01, SideRearCam02, SurCam01, SurCam02, SurCam03, SurCam04
            }
          }, {
            "x": 732.6589,
            "y": 612.7096,
            "z": 0.3537,
            "properties": {
              "v_11": ["F1", "F2", "S2"],         // 可见摄像头
              "v_00": "-1",                       // 视角不可见摄像头, 去除该属性
              "v_21": ["S3", "S4", "SF1", "SF2"], // 自车遮挡不可见摄像头
              "v_22": ["S3", "S4", "SF1", "SF2"], // 其他遮挡不可见摄像头
              "v_bev": "-1",                      // 该字段暂时不用, 保留字段
              "v": ['0', '1', '0', "1", '0', '1'] // 该点在各个摄像头里是否存在: 1 是, 0 否. 摄像头排列顺序: FrontCam01, FrontCam02, RearCam01, SideFrontCam01, SideFrontCam02, SideRearCam01, SideRearCam02, SurCam01, SurCam02, SurCam03, SurCam04
            }
          }, {
            "x": 1032.6589,
            "y": 612.7096,
            "z": 0.3537,
            "properties": {
              "v_11": ["F1", "F2", "S2"],         // 可见摄像头
              "v_00": "-1",                       // 视角不可见摄像头, 去除该属性
              "v_21": ["S3", "S4", "SF1", "SF2"], // 自车遮挡不可见摄像头
              "v_22": ["S3", "S4", "SF1", "SF2"], // 其他遮挡不可见摄像头
              "v_bev": "-1",                      // 该字段暂时不用, 保留字段
              "v": ['0', '1', '0', "1", '0', '1'] // 该点在各个摄像头里是否存在: 1 是, 0 否. 摄像头排列顺序: FrontCam01, FrontCam02, RearCam01, SideFrontCam01, SideFrontCam02, SideRearCam01, SideRearCam02, SurCam01, SurCam02, SurCam03, SurCam04
            }
          }, {
            "x": 1032.6589,
            "y": 612.7096,
            "z": 0.3537,
            "properties": {
              "v_11": ["F1", "F2", "S2"],         // 可见摄像头
              "v_00": "-1",                       // 视角不可见摄像头, 去除该属性
              "v_21": ["S3", "S4", "SF1", "SF2"], // 自车遮挡不可见摄像头
              "v_22": ["S3", "S4", "SF1", "SF2"], // 其他遮挡不可见摄像头
              "v_bev": "-1",                      // 该字段暂时不用, 保留字段
              "v": ['0', '1', '0', "1", '0', '1'] // 该点在各个摄像头里是否存在: 1 是, 0 否. 摄像头排列顺序: FrontCam01, FrontCam02, RearCam01, SideFrontCam01, SideFrontCam02, SideRearCam01, SideRearCam02, SurCam01, SurCam02, SurCam03, SurCam04
            }
          }, {
            "x": 473.4729,
            "y": 611.3537,
            "z": 0.3537,
            "properties": {
              "v_11": ["F1", "F2", "S2"],         // 可见摄像头
              "v_00": "-1",                       // 视角不可见摄像头, 去除该属性
              "v_21": ["S3", "S4", "SF1", "SF2"], // 自车遮挡不可见摄像头
              "v_22": ["S3", "S4", "SF1", "SF2"], // 其他遮挡不可见摄像头
              "v_bev": "-1",                      // 该字段暂时不用, 保留字段
              "v": ['0', '1', '0', "1", '0', '1'] // 该点在各个摄像头里是否存在: 1 是, 0 否. 摄像头排列顺序: FrontCam01, FrontCam02, RearCam01, SideFrontCam01, SideFrontCam02, SideRearCam01, SideRearCam02, SurCam01, SurCam02, SurCam03, SurCam04
            }
          }]
        }
      ]
    },
    "10": {
      "type": "road_marker_line",         // 地面标线, 不需要交付 2D 人工修改结果
      "properties": {
        "track_id": "10",                  // 物体 track id, 字符串类型
        "category": "1",                  // 类型: 1 减速带, 2 横向减速线, 3 斑马线, 4 停止线, 5 停止让行线, 6 减速让行线, 7 禁停区, 8 车距确认线
        "color": "1",                     // 颜色: 1 白色, 2 暂留颜色值 3 其他
        "zebra_crossing_standard": "0"    // 斑马线是否国标: 0 否, 1 是, -1 无效，对于非斑马线填该值
      },
      "objects": [
        {
          "sensor": "LidarFusion",        // 传感器编号
          "type": "road_marker_line_3d",
          "geometry": "polygon_3d",       // 3D 多边形，首尾点坐标不相同, 对与 停止线用该值可能为 line_3d
          "points": [{
            "x": 473.4729,
            "y": 611.3537,
            "z": 0.3537,
            "properties": {
              "v_11": ["F1", "F2", "S2"],         // 可见摄像头
              "v_00": "-1",                       // 视角不可见摄像头, 去除该属性
              "v_21": ["S3", "S4", "SF1", "SF2"], // 自车遮挡不可见摄像头
              "v_22": ["S3", "S4", "SF1", "SF2"], // 其他遮挡不可见摄像头
              "v_bev": "-1",                      // 该字段暂时不用, 保留字段
              "v": ['0', '1', '0', "1", '0', '1'] // 该点在各个摄像头里是否存在: 1 是, 0 否. 摄像头排列顺序: FrontCam01, FrontCam02, RearCam01, SideFrontCam01, SideFrontCam02, SideRearCam01, SideRearCam02, SurCam01, SurCam02, SurCam03, SurCam04
            }
          }, {
            "x": 732.6589,
            "y": 612.7096,
            "z": 0.3537,
            "properties": {
              "v_11": ["F1", "F2", "S2"],         // 可见摄像头
              "v_00": "-1",                       // 视角不可见摄像头, 去除该属性
              "v_21": ["S3", "S4", "SF1", "SF2"], // 自车遮挡不可见摄像头
              "v_22": ["S3", "S4", "SF1", "SF2"], // 其他遮挡不可见摄像头
              "v_bev": "-1",                      // 该字段暂时不用, 保留字段
              "v": ['0', '1', '0', "1", '0', '1'] // 该点在各个摄像头里是否存在: 1 是, 0 否. 摄像头排列顺序: FrontCam01, FrontCam02, RearCam01, SideFrontCam01, SideFrontCam02, SideRearCam01, SideRearCam02, SurCam01, SurCam02, SurCam03, SurCam04
            }
          }, {
            "x": 1032.6589,
            "y": 612.7096,
            "z": 0.3537,
            "properties": {
              "v_11": ["F1", "F2", "S2"],         // 可见摄像头
              "v_00": "-1",                       // 视角不可见摄像头, 去除该属性
              "v_21": ["S3", "S4", "SF1", "SF2"], // 自车遮挡不可见摄像头
              "v_22": ["S3", "S4", "SF1", "SF2"], // 其他遮挡不可见摄像头
              "v_bev": "-1",                      // 该字段暂时不用, 保留字段
              "v": ['0', '1', '0', "1", '0', '1'] // 该点在各个摄像头里是否存在: 1 是, 0 否. 摄像头排列顺序: FrontCam01, FrontCam02, RearCam01, SideFrontCam01, SideFrontCam02, SideRearCam01, SideRearCam02, SurCam01, SurCam02, SurCam03, SurCam04
            }
          }, {
            "x": 473.4729,
            "y": 611.3537,
            "z": 0.3537,
            "properties": {
              "v_11": ["F1", "F2", "S2"],         // 可见摄像头
              "v_00": "-1",                       // 视角不可见摄像头, 去除该属性
              "v_21": ["S3", "S4", "SF1", "SF2"], // 自车遮挡不可见摄像头
              "v_22": ["S3", "S4", "SF1", "SF2"], // 其他遮挡不可见摄像头
              "v_bev": "-1",                      // 该字段暂时不用, 保留字段
              "v": ['0', '1', '0', "1", '0', '1'] // 该点在各个摄像头里是否存在: 1 是, 0 否. 摄像头排列顺序: FrontCam01, FrontCam02, RearCam01, SideFrontCam01, SideFrontCam02, SideRearCam01, SideRearCam02, SurCam01, SurCam02, SurCam03, SurCam04
            }
          }, {
            "x": 473.4729,
            "y": 611.3537,
            "z": 0.3537,
            "properties": {
              "v_11": ["F1", "F2", "S2"],         // 可见摄像头
              "v_00": "-1",                       // 视角不可见摄像头, 去除该属性
              "v_21": ["S3", "S4", "SF1", "SF2"], // 自车遮挡不可见摄像头
              "v_22": ["S3", "S4", "SF1", "SF2"], // 其他遮挡不可见摄像头
              "v_bev": "-1",                      // 该字段暂时不用, 保留字段
              "v": ['0', '1', '0', "1", '0', '1'] // 该点在各个摄像头里是否存在: 1 是, 0 否. 摄像头排列顺序: FrontCam01, FrontCam02, RearCam01, SideFrontCam01, SideFrontCam02, SideRearCam01, SideRearCam02, SurCam01, SurCam02, SurCam03, SurCam04
            }
          }]
        }
      ]
    },
    "11": {
      "type": "road_marker_sign",         // 地面标识, 不需要交付 2D 人工修改结果
      "properties": {
        "track_id": "12",                 // 物体 track id, 字符串类型
        "category": "1",                  // 类型: 1 文字, 2 数字, 3 减速让行标志, 4 人行横道提醒标志, 5 图形
      },
      "objects": [
        {
          "sensor": "LidarFusion",        // 传感器编号
          "type": "road_marker_sign_3d",
          "geometry": "polygon_3d",       // 3D 多边形，首尾点坐标不相同
          "points": [{
            "x": 473.4729,
            "y": 611.3537,
            "z": 0.3537,
            "properties": {
              "v_11": ["F1", "F2", "S2"],         // 可见摄像头
              "v_00": "-1",                       // 视角不可见摄像头, 去除该属性
              "v_21": ["S3", "S4", "SF1", "SF2"], // 自车遮挡不可见摄像头
              "v_22": ["S3", "S4", "SF1", "SF2"], // 其他遮挡不可见摄像头
              "v_bev": "-1",                      // 该字段暂时不用, 保留字段
              "v": ['0', '1', '0', "1", '0', '1'] // 该点在各个摄像头里是否存在: 1 是, 0 否. 摄像头排列顺序: FrontCam01, FrontCam02, RearCam01, SideFrontCam01, SideFrontCam02, SideRearCam01, SideRearCam02, SurCam01, SurCam02, SurCam03, SurCam04
            }
          },{
            "x": 732.6589,
            "y": 612.7096,
            "z": 0.3537,
            "properties": {
              "v_11": ["F1", "F2", "S2"],         // 可见摄像头
              "v_00": "-1",                       // 视角不可见摄像头, 去除该属性
              "v_21": ["S3", "S4", "SF1", "SF2"], // 自车遮挡不可见摄像头
              "v_22": ["S3", "S4", "SF1", "SF2"], // 其他遮挡不可见摄像头
              "v_bev": "-1",                      // 该字段暂时不用, 保留字段
              "v": ['0', '1', '0', "1", '0', '1'] // 该点在各个摄像头里是否存在: 1 是, 0 否. 摄像头排列顺序: FrontCam01, FrontCam02, RearCam01, SideFrontCam01, SideFrontCam02, SideRearCam01, SideRearCam02, SurCam01, SurCam02, SurCam03, SurCam04
            }
          },{
            "x": 1032.6589,
            "y": 612.7096,
            "z": 0.3537,
            "properties": {
              "v_11": ["F1", "F2", "S2"],         // 可见摄像头
              "v_00": "-1",                       // 视角不可见摄像头, 去除该属性
              "v_21": ["S3", "S4", "SF1", "SF2"], // 自车遮挡不可见摄像头
              "v_22": ["S3", "S4", "SF1", "SF2"], // 其他遮挡不可见摄像头
              "v_bev": "-1",                      // 该字段暂时不用, 保留字段
              "v": ['0', '1', '0', "1", '0', '1'] // 该点在各个摄像头里是否存在: 1 是, 0 否. 摄像头排列顺序: FrontCam01, FrontCam02, RearCam01, SideFrontCam01, SideFrontCam02, SideRearCam01, SideRearCam02, SurCam01, SurCam02, SurCam03, SurCam04
            }
          },{
            "x": 473.4729,
            "y": 611.3537,
            "z": 0.3537,
            "properties": {
              "v_11": ["F1", "F2", "S2"],         // 可见摄像头
              "v_00": "-1",                       // 视角不可见摄像头, 去除该属性
              "v_21": ["S3", "S4", "SF1", "SF2"], // 自车遮挡不可见摄像头
              "v_22": ["S3", "S4", "SF1", "SF2"], // 其他遮挡不可见摄像头
              "v_bev": "-1",                      // 该字段暂时不用, 保留字段
              "v": ['0', '1', '0', "1", '0', '1'] // 该点在各个摄像头里是否存在: 1 是, 0 否. 摄像头排列顺序: FrontCam01, FrontCam02, RearCam01, SideFrontCam01, SideFrontCam02, SideRearCam01, SideRearCam02, SurCam01, SurCam02, SurCam03, SurCam04
            }
          },{
            "x": 473.4729,
            "y": 611.3537,
            "z": 0.3537,
            "properties": {
              "v_11": ["F1", "F2", "S2"],         // 可见摄像头
              "v_00": "-1",                       // 视角不可见摄像头, 去除该属性
              "v_21": ["S3", "S4", "SF1", "SF2"], // 自车遮挡不可见摄像头
              "v_22": ["S3", "S4", "SF1", "SF2"], // 其他遮挡不可见摄像头
              "v_bev": "-1",                      // 该字段暂时不用, 保留字段
              "v": ['0', '1', '0', "1", '0', '1'] // 该点在各个摄像头里是否存在: 1 是, 0 否. 摄像头排列顺序: FrontCam01, FrontCam02, RearCam01, SideFrontCam01, SideFrontCam02, SideRearCam01, SideRearCam02, SurCam01, SurCam02, SurCam03, SurCam04
            }
          }]
        }
      ]
    },
    "12": {
      "type": "lane_line",              // 车道线, 不需要交付 2D 人工修改结果
      "properties": {
        "track_id": "12",                // 物体 track id, 字符串类型
      },
      "objects": [
        {
          "sensor": "LidarFusion4D",      // 传感器编号
          "type": "lane_line_3d",
          "geometry": "line_3d",
          "properties": {
            "function": "0",            // 功能: 0 常规, 1 待转区, 2 纵向减速线(鱼骨线), 3 潮汐车道线, 4 可变车道线, 待对齐 RestrictionLine 和 RestrictionPoint 后再决定是否保留该属性
          },
          "extra_properties": {
            "TYPE": ["c"],              // 透传 sq3 type: a:未分类、b:其它、c:实线、d:虚线、e:短粗虚线、f:粗实线、g:导流带标线、h:停车位标线、i:虚拟线、j:路口虚拟线、s:虚拟辅助线
            // k:道路边缘、l:施工围挡、m:防护栏、n:路缘石、o:墙体、p:防护网、q:自然边界、r:物理隔离、 l~p 暂不使用，均归属 为 r 物理隔离. 从左至右位数组合表达
            "COLOR": ["c"],             // 透传 sq3 color: a:未分类 b:其它(红色、粉色等其它颜色) c 白色、d 黄色、e 橙色、f 蓝色、g 绿色、h 不适用, 从左至右位数组合表达                
            "EX_INFO": ["e"],           // 透传 EX_INFO: b:其它、c:无附属线 d:附属线 e:纵向减速标线 f:可变导向车道标线 g:借道区标线 h:公交车专用道标识线 i:HOV 车道标识线 z:车道边线 当 EX_INFO=d~i 时，从左至右位数组合表达
            "SEPARATE": "1"             // 透传 SEPARATE: 0:未分类 1:无隔离设施 2:物理隔离 3:防护栏 4:路缘石 5:墙体 6:防护网 7:杆 8:其它不可跨越防护设施 9:其它可跨越防护设施
          },
          "points": [{                  // 每个点添加一个可见性属性
            "x": 473.4729,
            "y": 611.3537,
            "z": 0.3538,
            "properties": {
              "v_11": ["F1", "F2", "S2"],         // 可见摄像头
              "v_00": "-1",                       // 视角不可见摄像头, 去除该属性
              "v_21": ["S3", "S4", "SF1", "SF2"], // 自车遮挡不可见摄像头
              "v_22": ["S3", "S4", "SF1", "SF2"], // 其他遮挡不可见摄像头
              "v_bev": "-1",                      // 该字段暂时不用, 保留字段
              "v": ['0', '1', '0', "1", '0', '1'] // 该点在各个摄像头里是否存在: 1 是, 0 否. 摄像头排列顺序: FrontCam01, FrontCam02, RearCam01, SideFrontCam01, SideFrontCam02, SideRearCam01, SideRearCam02, SurCam01, SurCam02, SurCam03, SurCam04
            }
          }, {
            "x": 732.6589,
            "y": 612.7096,
            "z": 0.3538,
            "properties": {
              "v_11": ["F1", "F2", "S2"],         // 可见摄像头
              "v_00": "-1",                       // 视角不可见摄像头, 去除该属性
              "v_21": ["S3", "S4", "SF1", "SF2"], // 自车遮挡不可见摄像头
              "v_22": ["S3", "S4", "SF1", "SF2"], // 其他遮挡不可见摄像头
              "v_bev": "-1",                      // 该字段暂时不用, 保留字段
              "v": ['0', '1', '0', "1", '0', '1'] // 该点在各个摄像头里是否存在: 1 是, 0 否. 摄像头排列顺序: FrontCam01, FrontCam02, RearCam01, SideFrontCam01, SideFrontCam02, SideRearCam01, SideRearCam02, SurCam01, SurCam02, SurCam03, SurCam04
            }
          }, {
            "x": 1032.6589,
            "y": 612.7096,
            "z": 0.3538,
            "properties": {
              "v_11": ["F1", "F2", "S2"],         // 可见摄像头
              "v_00": "-1",                       // 视角不可见摄像头, 去除该属性
              "v_21": ["S3", "S4", "SF1", "SF2"], // 自车遮挡不可见摄像头
              "v_22": ["S3", "S4", "SF1", "SF2"], // 其他遮挡不可见摄像头
              "v_bev": "-1",                      // 该字段暂时不用, 保留字段
              "v": ['0', '1', '0', "1", '0', '1'] // 该点在各个摄像头里是否存在: 1 是, 0 否. 摄像头排列顺序: FrontCam01, FrontCam02, RearCam01, SideFrontCam01, SideFrontCam02, SideRearCam01, SideRearCam02, SurCam01, SurCam02, SurCam03, SurCam04
            }
          }]
        },
        {
          "sensor": "LidarFusion4D",        // 传感器编号
          "type": "break_point_3d",
          "geometry": "point_3d",
          "points": [{                      // 车道线断点, 用以标识后续车道线的信息
            "x": 473.4729,
            "y": 611.3537,
            "z": 0.3538,
            "extra_properties": {
              "TYPE": ["c"],              // 透传 sq3 type: a:无变化、b:其它、c:实线、d:虚线、e:短粗虚线、f:粗实线、g:导流带标线、h:停车位标线、i:虚拟线、j:路口虚拟线、k:道路边缘、l:施工围挡、m:防护栏、n:路缘石、o:墙体、p:防护网、q:自然边界、r:物理隔离、s:虚拟辅助线, l~p 暂不使用，均归属 为 r 物理隔离. 从左至右位数组合表达
              "COLOR": ["c"],                 // 透传 sq3 color: a:无变化 b:其它(红色、粉色等其它颜 色) c 白色、d 黄色、e 橙色、f 蓝色、g 绿色、h 不适用, 从左至右位数组合表达                
              "EX_INFO": ["e"],               // 透传 EX_INFO: b:其它、c:无附属线 d:附属线 e:纵向减速标线 f:可变导向车道标线 g:借道区标线 h:公交车专用道标识线 i:HOV 车道标识线 z:车道边线 当 EX_INFO=d~i 时，从左至右位数组合表达
              "SEPARATE": "1"                 // 透传 SEPARATE: 0:无变化 1:无隔离设施 2:物理隔离 3:防护栏 4:路缘石 5:墙体 6:防护网 7:杆 8:其它不可跨越防护设施 9:其它可跨越防护设施
            }
          },{                      // 每个点添加一个可见性属性
            "x": 473.4729,
            "y": 611.3537,
            "z": 0.3538,
            "extra_properties": {
              "TYPE": ["c"],              // 透传 sq3 type: a:无变化、b:其它、c:实线、d:虚线、e:短粗虚线、f:粗实线、g:导流带标线、h:停车位标线、i:虚拟线、j:路口虚拟线、k:道路边缘、l:施工围挡、m:防护栏、n:路缘石、o:墙体、p:防护网、q:自然边界、r:物理隔离、s:虚拟辅助线, l~p 暂不使用，均归属 为 r 物理隔离. 从左至右位数组合表达
              "COLOR": ["c"],                 // 透传 sq3 color: a:无变化 b:其它(红色、粉色等其它颜 色) c 白色、d 黄色、e 橙色、f 蓝色、g 绿色、h 不适用, 从左至右位数组合表达                
              "EX_INFO": ["e"],               // 透传 EX_INFO: b:其它、c:无附属线 d:附属线 e:纵向减速标线 f:可变导向车道标线 g:借道区标线 h:公交车专用道标识线 i:HOV 车道标识线 z:车道边线 当 EX_INFO=d~i 时，从左至右位数组合表达
              "SEPARATE": "1"                 // 透传 SEPARATE: 0:无变化 1:无隔离设施 2:物理隔离 3:防护栏 4:路缘石 5:墙体 6:防护网 7:杆 8:其它不可跨越防护设施 9:其它可跨越防护设施
            }
          }]
        }
      ]
    }
  }
}
```