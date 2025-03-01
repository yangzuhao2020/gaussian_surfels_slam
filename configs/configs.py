import os
from datasets.gradslam_datasets.dataconfig import load_dataset_config

def setup_config_defaults(config):
    """ 确保 config 具有所有必要的默认值 """
    
    # Tracking 相关参数
    tracking_defaults = {
        "use_depth_loss_thres": False,
        "depth_loss_thres": 100000,
        "visualize_tracking_loss": False
    }
    config["tracking"] = {**tracking_defaults, **config.get("tracking", {})}
    # **是和并字典的一种方式。如果不存在tracking_defaults 中的字典，则添加到到字典中，如果存在的话，则不修改原来的参数。
    
    # 关键帧选择
    config.setdefault("distance_keyframe_selection", False)
    if config["distance_keyframe_selection"]:
        print("Using CDF Keyframe Selection. Note that 'mapping window size' is useless.")
        config.setdefault("distance_current_frame_prob", 0.5)

    # 高斯简化
    config.setdefault("gaussian_simplification", True)
    if not config["gaussian_simplification"]:
        print("Using Full Gaussian Representation, which may cause unstable optimization if not fully optimized.")

    return config


def setup_directories(config):
    """ 创建实验结果输出目录 """
    output_dir = os.path.join(config["workdir"], config["run_name"])
    eval_dir = os.path.join(output_dir, "eval")
    os.makedirs(eval_dir, exist_ok=True)
    return output_dir, eval_dir


def setup_dataset_config(dataset_config):
    """ 处理数据集默认参数 
    dataset_config → 完整的数据集配置
    gradslam_data_cfg → GradSLAM 相关的数据配置，包括数据集的名称 字典
    seperate_densification_res → 是否单独设置了密度化(Densification)图像分辨率 bool
    seperate_tracking_res → 是否单独设置了 Tracking(跟踪)图像分辨率 bool
    """
    dataset_defaults = {
        "train_or_test": "all",
        "preload": False,
        "ignore_bad": False,
        "use_train_split": True
    }
    dataset_config = {**dataset_defaults, **dataset_config}

    # GradSLAM 数据集配置
    if "gradslam_data_cfg" not in dataset_config:
        gradslam_data_cfg = {"dataset_name": dataset_config["dataset_name"]}
    else:
        gradslam_data_cfg = load_dataset_config(dataset_config["gradslam_data_cfg"])

    # 处理图像分辨率
    dataset_config.setdefault("densification_image_height", dataset_config["desired_image_height"])
    dataset_config.setdefault("densification_image_width", dataset_config["desired_image_width"])
    dataset_config.setdefault("tracking_image_height", dataset_config["desired_image_height"])
    dataset_config.setdefault("tracking_image_width", dataset_config["desired_image_width"])

    seperate_densification_res = (
        dataset_config["densification_image_height"] != dataset_config["desired_image_height"] or 
        dataset_config["densification_image_width"] != dataset_config["desired_image_width"]
    )

    seperate_tracking_res = (
        dataset_config["tracking_image_height"] != dataset_config["desired_image_height"] or 
        dataset_config["tracking_image_width"] != dataset_config["desired_image_width"]
    )
    dataset_config["seperate_densification_res"] = seperate_densification_res
    dataset_config["seperate_tracking_res"] = seperate_tracking_res

    return dataset_config, gradslam_data_cfg