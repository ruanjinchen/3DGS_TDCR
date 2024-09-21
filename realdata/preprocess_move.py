import json
import numpy as np

fovx = 0.92
fovy = 0.71

info_30d_train = {"images": []}
info_30d_test = {"images": []}
info_all_train = {"images": []}
info_all_test = {"images": []}
with open("camangle_transforms_map.json") as fid:
    camangle_transforms_map = json.load(fid)


def process(date: str, limit: str):
    with open("robot/img_" + limit + "_" + date + "/transforms_train.json") as fid:
        transforms_train = json.load(fid)
        frames = transforms_train["frames"]
        for frame in frames:
            file_path = frame["file_path"]
            image_name = file_path.split("/")[-1]
            ex = {"name": image_name, "width": 640, "height": 480, "camera_id": 2, "fovx": fovx, "fovy": fovy,
                  "image_path": "realdata/output_robot/" + limit + "_" + date + "/train/" + image_name,
                  "joint_id": frame["joint_id"]}
            cam_angle = frame["cam_angle"]
            key = date + "_" + str(cam_angle)
            [R, T] = camangle_transforms_map[key]
            ex["R"] = R
            ex["T"] = T
            if limit == "30d":
                info_30d_train["images"].append(ex)
            else:
                info_all_train["images"].append(ex)

    try:
        with open("robot/img_" + limit + "_" + date + "/transforms_test.json") as fid:
            transforms_test = json.load(fid)
            frames = transforms_test["frames"]
            for frame in frames:
                file_path = frame["file_path"]
                image_name = file_path.split("/")[-1]
                ex = {"name": image_name, "width": 640, "height": 480, "camera_id": 2, "fovx": fovx, "fovy": fovy,
                      "image_path": "realdata/output_robot/" + limit + "_" + date + "/test/" + image_name,
                      "joint_id": frame["joint_id"]}
                cam_angle = frame["cam_angle"]
                key = date + "_" + str(cam_angle)
                [R, T] = camangle_transforms_map[key]
                ex["R"] = R
                ex["T"] = T
                if limit == "30d":
                    info_30d_test["images"].append(ex)
                else:
                    info_all_test["images"].append(ex)
    except FileNotFoundError:
        print("no test data for " + date)

process("814", "30d")
process("814", "all")
process("815", "all")
process("816", "all")
process("817", "all")
process("818", "all")
process("819", "all")

with open("info_30d_train.json", "w") as fid:
    json.dump(info_30d_train, fid, indent=4)
with open("info_30d_test.json", "w") as fid:
    json.dump(info_30d_test, fid, indent=4)
with open("info_all_train.json", "w") as fid:
    json.dump(info_all_train, fid, indent=4)
with open("info_all_test.json", "w") as fid:
    json.dump(info_all_test, fid, indent=4)


