"""
요청 형식/타입
{
	"mem_sex":"m", #str
  "img_path":{  #dict
		"dir_path":["/2022/09/20/1/1942442_2022092017174807642.jpg"] #list(str)
     }
}
타입 설명
mem_sex : 회원 성별, 타입 string
Img_path : 이미지 경로 사전, 타입 dictionary
dir_path  : 이미지 경로 , 타입 list(str)


리턴 형식/타입
{
	“authentication": “True”/”False”/”Wait” #str
	"label": 0 #int
}

True : 대표사진 자동 인증
False : 대표사진 자동 거부
Wait : 대표사진 관리자 인증 대기

예외 발생시 리턴 형식/타입
{}

"""
from mask_functions import Detect, HoneyDetect,imgForderPath
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
from typing import List
from conf.gunicorn_log import LogConfig
import traceback
import time
import warnings
warnings.filterwarnings(action='ignore')

log = LogConfig()

class img_path(BaseModel):
    mem_sex: str = None
    img_path: dict
    dir_path: Optional[List[str]]

class honey_img_path(BaseModel):
    honey_img: Optional[List[dict]]
    mem_sex: str = None
    url_path: str = None

app = FastAPI()

@app.post('/predict')
async def make_prediction(img_path : img_path):
    img_dict = {}
    dir_path = img_path.img_path['dir_path']
    sex = img_path.mem_sex
    img_dir = imgForderPath + str(dir_path[0])
    try:
        log.Log(f"MaskAPI_Request  -> sex : {sex}, "                                   # 성별
                f"img_dir : {img_dir}")                                                # 요청된 이미지 경로
        start = time.time()
        img_dict, cls_lst, cls_name_lst, conf_lst, label_lst = Detect(img_dir, sex)
        img_dict['label'] = label_lst
        total_time = round(time.time() - start, 3)
        log.Log(f"MaskAPI_Response -> Authentication : {img_dict['authentication']}, " # 인증 결과
                f"return_label : {label_lst}, "                                        # 리턴시 라벨 번호  
                f"label : {cls_name_lst}, "                                            # 사진 라벨 이름
                f"confidence : {conf_lst}, "                                           # 신뢰도(정확도)
                f"Elapsed Time -> {total_time}")                                       # 처리 시간
    except Exception as e:
        trace_back = traceback.format_exc()
        message = str(e) + "\n" + str(trace_back)
        log.error_log(f'[Except Error!] {message}')

    return JSONResponse(img_dict)
@app.post('/honey')
async def make_prediction(honey_img_path : honey_img_path):
    results_list = []
    dir_path_lst = honey_img_path.honey_img
    try:

        for i in range(len(dir_path_lst)):
            start = time.time()
            sex = dir_path_lst[i]['mem_sex']
            url_path = dir_path_lst[i]["url_path"]
            log.Log(f"MaskAPI_Request  -> sex : {sex}, "                                   # 성별
                    f"url_path : {url_path}")                                                # 요청된 이미지 경로
            img_dict, cls_name_lst, conf_lst, label_lst = HoneyDetect(i, url_path, sex)
            img_dict['label'] = label_lst
            total_time = round(time.time() - start, 3)
            log.Log(f"MaskAPI_Response -> Authentication : {img_dict['authentication']}, " # 인증 결과
                    f"return_label : {label_lst}, "                                        # 리턴시 라벨 번호  
                    f"label : {cls_name_lst}, "                                            # 사진 라벨 이름
                    f"confidence : {conf_lst}, "                                           # 신뢰도(정확도)
                    f"Elapsed Time -> {total_time}")                                       # 처리 시간
            results_list.append(img_dict)
    except Exception as e:
        trace_back = traceback.format_exc()
        message = str(e) + "\n" + str(trace_back)
        log.error_log(f'[Except Error!] {message}')

    return JSONResponse(results_list)
