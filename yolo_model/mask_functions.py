from conf.common import config, NoDetect_file_path, honey_NoDetect_file_path
from utils.mask_func import imgDict, HoneyimgDict, AutoLabel, HoneyAutoLabel
import cv2
import torch
import numpy as np
from urllib import request

from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression, scale_coords, set_logging
from utils.torch_utils import select_device
import warnings
warnings.filterwarnings(action='ignore')

imgForderPath = config.imgForderPath
weights = config.yoloModelPath
view_img = False
save_txt = True
imgsz = 640
conf_thres = 0.5
iou_thres = 0.5
device = ''
save_conf = False
trace = False
nosave = False
classes = None
agnostic_nms = False
augment = False

# Initialize
# set_logging()
device = select_device(device)
# half = device.type != 'cpu'  # half precision only supported on CUDA
half = False

# Load model
model = attempt_load(weights, map_location=device)  # load FP32 model
stride = int(model.stride.max())  # model stride
imgsz = check_img_size(imgsz, s=stride)  # check img_size

# Run inference
if device.type != 'cpu':
    model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
old_img_w = old_img_h = imgsz
old_img_b = 1

def Detect(source, sex):
    img_name = source.split('/')[-1].split('.')[0]

    # Load image
    img0 = cv2.imread(source)  # BGR
    assert img0 is not None, 'Image Not Found ' + source

    # padding image
    image = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)

    # 가로, 세로에 대해 부족한 margin 계산
    height, width = image.shape[0:2]
    margin = [np.abs(height - width) // 2, np.abs(height - width) // 2]

    # 부족한 길이가 절반으로 안 떨어질 경우 +1
    if np.abs(height - width) % 2 != 0:
        margin[0] += 1

    # 가로, 세로 가운데 부족한 쪽에 margin 추가
    if height < width:
        margin_list = [margin, [0, 0]]
    else:
        margin_list = [[0, 0], margin]

    # color 이미지일 경우 color 채널 margin 추가
    if len(image.shape) == 3:
        margin_list.append([0, 0])

    # 이미지에 margin 추가
    output = np.pad(image, margin_list, mode='constant')
    output = cv2.resize(output, (640, 640))
    new_output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)

    # Padded resize
    img = letterbox(new_output, imgsz, stride=stride)[0]
    im0 = img.copy()

    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0 정규화
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Inference
    with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
        pred = model(img, augment=augment)[0]

    # Apply NMS 모델이 예측한 bbox중 겹치는 부분이 많은 것들을 제거
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes=classes, agnostic=agnostic_nms)

    # Process detections
    det = pred[0]
    y1, x1, z1 = np.shape(im0)
    conf = 0
    img_dict = {}
    label_lst = []
    cls_lst = []
    cls_name_lst = []
    conf_lst = []
    percent_lst = []
    if len(det):
        # Rescale boxes from img_size to im0 size
        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
        # Write results
        for *xyxy, conf, cls in reversed(det):
            x = int(xyxy[0])
            y = int(xyxy[1])
            w = int(xyxy[2])
            h = int(xyxy[3])
            w1,h1, _ = im0[y:h, x:w].shape
            percent = round((w1 * h1) / (x1 * y1) * 100, 2)
            percent_lst.append(percent)
            cls_lst.append(int(cls))
            conf = int(conf * 100)
            conf_lst.append(conf)

        # label nameing
        label_name = {0 : "HF", 1 : "MF", 2 : "OF", 3 : "CF", 4 : "AF", 5 : "BF", 6 : "FF", 7 : "UB", 8 : "ID"}
        for cls in range(len(cls_lst)):
            cls_name_lst.append(list(map(lambda x: label_name[x], cls_lst))[cls])

        img_dict, cls_lst, label_lst, conf_lst = AutoLabel(source, sex, img_dict, cls_lst, label_lst, conf_lst, img_name, conf, percent_lst)

    else:  # 객체 인식을 못한 경우
        img_dict["authentication"] = imgDict(source, img_name, conf, NoDetect_file_path)
        cls_lst = ["ND"]
        cls_name_lst = ['ND']
        conf_lst = [0]
        label_lst.append(2)
    return img_dict,  cls_name_lst, conf_lst, label_lst
def HoneyDetect(index, source, sex):
    img_name = f'honey_{index}'
    data = request.urlopen(source).read()
    encoded_img = np.fromstring(data, dtype=np.uint8)
    img0 = cv2.imdecode(encoded_img, cv2.IMREAD_COLOR)
    # Load image
    assert img0 is not None, 'Image Not Found ' + source

    # padding image
    image = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)

    # 가로, 세로에 대해 부족한 margin 계산
    height, width = image.shape[0:2]
    margin = [np.abs(height - width) // 2, np.abs(height - width) // 2]

    # 부족한 길이가 절반으로 안 떨어질 경우 +1
    if np.abs(height - width) % 2 != 0:
        margin[0] += 1

    # 가로, 세로 가운데 부족한 쪽에 margin 추가
    if height < width:
        margin_list = [margin, [0, 0]]
    else:
        margin_list = [[0, 0], margin]

    # color 이미지일 경우 color 채널 margin 추가
    if len(image.shape) == 3:
        margin_list.append([0, 0])

    # 이미지에 margin 추가
    output = np.pad(image, margin_list, mode='constant')
    output = cv2.resize(output, (640, 640))
    new_output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)

    # Padded resize
    img = letterbox(new_output, imgsz, stride=stride)[0]
    im0 = img.copy()

    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0 정규화
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Inference
    with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
        pred = model(img, augment=augment)[0]

    # Apply NMS 모델이 예측한 bbox중 겹치는 부분이 많은 것들을 제거
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes=classes, agnostic=agnostic_nms)

    # Process detections
    det = pred[0]
    y1, x1, z1 = np.shape(im0)
    conf = 0
    img_dict = {}
    label_lst = []
    cls_lst = []
    cls_name_lst = []
    conf_lst = []
    percent_lst = []
    if len(det):
        # Rescale boxes from img_size to im0 size
        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
        # Write results
        for *xyxy, conf, cls in reversed(det):
            x = int(xyxy[0])
            y = int(xyxy[1])
            w = int(xyxy[2])
            h = int(xyxy[3])
            w1,h1, _ = im0[y:h, x:w].shape
            percent = round((w1 * h1) / (x1 * y1) * 100, 2)
            percent_lst.append(percent)
            cls_lst.append(int(cls))
            conf = int(conf * 100)
            conf_lst.append(conf)

        # label nameing
        label_name = {0 : "HF", 1 : "MF", 2 : "OF", 3 : "CF", 4 : "AF", 5 : "BF", 6 : "FF", 7 : "UB", 8 : "ID"}
        for cls in range(len(cls_lst)):
            cls_name_lst.append(list(map(lambda x: label_name[x], cls_lst))[cls])

        img_dict, cls_lst, label_lst, conf_lst = HoneyAutoLabel(im0, sex, img_dict, cls_lst, label_lst, conf_lst, img_name, conf, percent_lst)

    else:  # 객체 인식을 못한 경우
        img_dict["authentication"] = HoneyimgDict(im0, img_name, conf, honey_NoDetect_file_path)
        cls_name_lst = ['ND']
        conf_lst = [0]
        label_lst.append(2)
    return img_dict,  cls_name_lst, conf_lst, label_lst