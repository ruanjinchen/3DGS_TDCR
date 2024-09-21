import json
import numpy as np
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal

# read cameras.txt
class Camera:
    def __init__(self, camera_id, width, height, params):
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.params = params
        self.fovx = focal2fov(params[0], width)
        self.fovy = focal2fov(params[1], height)

path_camera = "../814withtest/cameras.txt"
cameras = []
with open(path_camera, "r") as fid:
    while True:
        line = fid.readline()
        if not line:
            break
        line = line.strip()
        if len(line) > 0 and line[0] != "#":
            elems = line.split()
            camera_id = int(elems[0])
            model = elems[1]
            assert model == "PINHOLE", "While the loader support other types, the rest of the code assumes PINHOLE"
            width = int(elems[2])
            height = int(elems[3])
            params = np.array(tuple(map(float, elems[4:])))
            cameras.append(Camera(camera_id, width, height, params))

print("cameras.txt loaded")
# camera[0]'s picture is downscaled to 640x480, fov doesn't change
cameras[0].width = 640
cameras[0].height = 480

# cameras[1]'s fov only keep 2 decimal places, all cameras except cameras[0] are same
cameras[1].fovx = 0.92
cameras[1].fovy = 0.71


# prepare a json for every image
# read images.txt
info = {"images": []}
info_test = {"images": []}
ex_for_robot = {"814": [],
                "815": [],
                "816": [],
                "817": [],
                "818": [],
                "819": [],}

path_image = "../814withtest/images.txt"
with open(path_image, "r") as fid:
    while True:
        line = fid.readline()
        if not line:
            break
        line = line.strip()
        if len(line) > 0 and line[0] != "#":
            elems = line.split()
            image_id = int(elems[0])
            qvec = np.array(tuple(map(float, elems[1:5])))
            tvec = np.array(tuple(map(float, elems[5:8])))
            camera_id = int(elems[8])
            image_name = elems[9]
            elems = fid.readline().split()
            xys = np.column_stack([tuple(map(float, elems[0::3])),
                                   tuple(map(float, elems[1::3]))])
            point3D_ids = np.array(tuple(map(int, elems[2::3])))
            uid = image_id
            R = np.transpose(qvec2rotmat(qvec))
            T = np.array(tvec)
            width = 640
            height = 480
            if camera_id == 1:
                fovx = cameras[0].fovx
                fovy = cameras[0].fovy
            else:
                fovx = cameras[1].fovx
                fovy = cameras[1].fovy
            if camera_id == 1:
                image_name = image_name.split("/")[-1] # get rid of the path
                image_path = "realdata/output_phone/" + image_name

                info["images"].append({"name": image_name, "image_path": image_path, "width": width, "height": height, "R": R.tolist(), "T": T.tolist(), "camera_id": camera_id, "fovx": fovx, "fovy": fovy, "joint_id" : 0})
            else:
                image_name_first= image_name.split("/")[0]
                if image_name_first == "robot":
                    ex_for_robot["814"].append({"name": image_name, "width": width, "height": height, "R": R.tolist(), "T": T.tolist(), "camera_id": 2, "fovx": fovx, "fovy": fovy})
                elif image_name_first == "robot815":
                    ex_for_robot["815"].append({"name": image_name, "width": width, "height": height, "R": R.tolist(), "T": T.tolist(), "camera_id": 2, "fovx": fovx, "fovy": fovy})
                elif image_name_first == "robot816":
                    ex_for_robot["816"].append({"name": image_name, "width": width, "height": height, "R": R.tolist(), "T": T.tolist(), "camera_id": 2, "fovx": fovx, "fovy": fovy})
                elif image_name_first == "robot818":
                    ex_for_robot["818"].append({"name": image_name, "width": width, "height": height, "R": R.tolist(), "T": T.tolist(), "camera_id": 2, "fovx": fovx, "fovy": fovy})

ex_for_robot["817"] = ex_for_robot["816"]
ex_for_robot["819"] = ex_for_robot["818"]

print(len(info["images"]))
camangle_transforms_map = {}

# read transforms_train.json and transforms_test.json
# global i


def process(date: str):
    i = 0
    with open("../realdata/robot/img_" + date + "/transforms_train.json") as fid:
        transforms_train = json.load(fid)
        frames = transforms_train["frames"]

        for frame in frames:
            file_path = frame["file_path"]
            image_name = file_path.split("/")[-1]
            ex = ex_for_robot[date][i]
            ex["image_path"] = "realdata/output_robot/" + date + "/train/" + image_name
            ex["name"] = image_name
            ex["joint_id"] = frame["joint_id"]
            info["images"].append(ex)
            cam_angle = frame["cam_angle"]
            # stack joints list to a string
            cam_angle = str(cam_angle)
            key = date + "_" + cam_angle
            camangle_transforms_map[key] = [ex["R"], ex["T"]]
            i += 1
    try:
        with open("../realdata/robot/img_" + date + "/transforms_test.json") as fid:
            transforms_test = json.load(fid)
            frames = transforms_test["frames"]

            for frame in frames:
                file_path = frame["file_path"]
                image_name = file_path.split("/")[-1]
                ex = ex_for_robot[date][i]
                ex["image_path"] = "realdata/output_robot/" + date + "/test/" + image_name
                ex["name"] = image_name
                ex["joint_id"] = frame["joint_id"]
                info_test["images"].append(ex)
                cam_angle = frame["cam_angle"]
                # stack joints list to a string
                cam_angle = str(cam_angle)
                key = date + "_" + cam_angle
                camangle_transforms_map[key] = [ex["R"], ex["T"]]
                i += 1
    except FileNotFoundError:
        print("no test data for " + date)
    return i

process("814")
process("815")
process("816")
process("817")
process("818")
process("819")
# output info to json
with open("info_0_train.json", "w") as fid:
    json.dump(info, fid, indent=4)
with open("info_0_test.json", "w") as fid:
    json.dump(info_test, fid, indent=4)
with open("camangle_transforms_map.json", "w") as fid:
    json.dump(camangle_transforms_map, fid, indent=4)
