import shutil, cv2
from conf.common import *

def imgDictTrue(source, img_name, conf, file_path):
    file_path_name = file_path + '/' + img_name + '_' + str(conf) + '%.jpg'
    shutil.copy2(source, file_path_name)
    return "True"
def imgDict(source, img_name, conf, file_path):
    file_path_name = file_path + '/' + img_name + '_' + str(conf) + '%.jpg'
    shutil.copy2(source, file_path_name)
    return "False"
def imgDictWait(source, img_name, conf, file_path):
    file_path_name = file_path + '/' + img_name + '_' + str(conf) + '%.jpg'
    shutil.copy2(source, file_path_name)
    return "Wait"
def HoneyimgDictTrue(im0, img_name, conf, file_path):
    file_path_name = file_path + '/' + img_name + '_' + str(conf) + '%.jpg'
    cv2.imwrite(f"{file_path_name}", im0)
    return "True"
def HoneyimgDict(im0, img_name, conf, file_path):
    file_path_name = file_path + '/' + img_name + '_' + str(conf) + '%.jpg'
    cv2.imwrite(f"{file_path_name}", im0)
    return "False"
def HoneyimgDictWait(im0, img_name, conf, file_path):
    file_path_name = file_path + '/' + img_name + '_' + str(conf) + '%.jpg'
    cv2.imwrite(f"{file_path_name}", im0)
    return "Wait"

def UB_IDfinder(cls, source, img_dict, cls_lst, label_lst, conf_lst, img_name, conf, res_int, file_path):
    check_lst = []
    # enumerate() 함수를 이용하여 특정 값이 포함된 모든 인덱스를 추출
    result_list = [i for i, value in enumerate(cls_lst) if value == cls]
    for re in result_list:
        if conf_lst[re] >= 60:
            check_lst.append(True)
        else:
            check_lst.append(False)
    if False not in check_lst:
        img_dict["authentication"] = imgDictWait(source, img_name, conf, file_path)
        label_lst.append(res_int)
    else:
        img_dict["authentication"] = imgDictWait(source, img_name, conf, None_file_path)
        label_lst.append(11)
    return img_dict, label_lst
def HoneyUB_IDfinder(im0, cls, img_dict, cls_lst, label_lst, conf_lst, img_name, conf, res_int, file_path):
    check_lst = []
    # enumerate() 함수를 이용하여 특정 값이 포함된 모든 인덱스를 추출
    result_list = [i for i, value in enumerate(cls_lst) if value == cls]
    for re in result_list:
        if conf_lst[re] >= 60:
            check_lst.append(True)
        else:
            check_lst.append(False)
    if False not in check_lst:
        img_dict["authentication"] = HoneyimgDictWait(im0,img_name, conf, file_path)
        label_lst.append(res_int)
    else:
        img_dict["authentication"] = HoneyimgDictWait(im0, img_name, conf, None_file_path)
        label_lst.append(11)
    return img_dict, label_lst
def AutoLabel(source, sex, img_dict, cls_lst, label_lst, conf_lst, img_name, conf, percent_lst):
    global UB_file_path ,ID_file_path, None_file_path, HF_f_path, HF_m_path, HFU_file_path, NHF_file_path, CF_file_path, AF_file_path,\
    BF_file_path ,FF_file_path, HFs_path, HFs_W_file_path, HFs_F_file_path

    if 7 in cls_lst:  # 객체가 하나라도 UB일 경우
        img_dict, label_lst = UB_IDfinder(7, source, img_dict, cls_lst, label_lst, conf_lst, img_name, conf, 16, UB_file_path)

    elif 8 in cls_lst:  # 객체가 하나라도 ID일 경우
        img_dict, label_lst = UB_IDfinder(8, source, img_dict, cls_lst, label_lst, conf_lst, img_name, conf, 17, ID_file_path)
    else:
        if len(cls_lst) == 1:  # 객체가 1개인 경우
            if percent_lst[0] <= 0.15:
                img_dict["authentication"] = imgDictWait(source, img_name, conf, None_file_path)
                label_lst.append(18)
            else:
                if conf >= 80:  # conf >= 80 인 경우
                    if int(cls_lst[0]) == 0:  # 라벨이 0(HF)인 경우
                        if sex == "f":  # 여자일 경우
                            img_dict["authentication"] = imgDictTrue(source, img_name, conf, HF_f_path)
                            label_lst.append(1)
                        else:  # 남자일 경우
                            img_dict["authentication"] = imgDictTrue(source, img_name, conf, HF_m_path)
                            label_lst.append(1)
                    else:  # 객체가 HF가 아닌 (HFU,NHF,CF,AF,BF,FF)인 경우
                        if int(cls_lst[0]) == 1:
                            img_dict["authentication"] = imgDict(source, img_name, conf, HFU_file_path)
                            label_lst.append(9)
                        elif int(cls_lst[0]) == 2:
                            img_dict["authentication"] = imgDict(source, img_name, conf, NHF_file_path)
                            label_lst.append(3)
                        elif int(cls_lst[0]) == 3:
                            img_dict["authentication"] = imgDict(source, img_name, conf, CF_file_path)
                            label_lst.append(4)
                        elif int(cls_lst[0]) == 4:
                            img_dict["authentication"] = imgDict(source, img_name, conf, AF_file_path)
                            label_lst.append(5)
                        elif int(cls_lst[0]) == 5:
                            img_dict["authentication"] = imgDictWait(source, img_name, conf, BF_file_path)
                            label_lst.append(14)
                        elif int(cls_lst[0]) == 6:
                            img_dict["authentication"] = imgDict(source, img_name, conf, FF_file_path)
                            label_lst.append(15)
                        else:
                            img_dict["authentication"] = imgDictWait(source, img_name, conf, None_file_path)
                            label_lst.append(11)
                else:  # 50 < conf < 80 인 경우
                    img_dict["authentication"] = imgDictWait(source, img_name, conf, None_file_path)
                    label_lst.append(11)
        else:  # 객체가 2개 이상인 경우
            if 0 in cls_lst:  # 객체 중에 HF가 있는 경우
                check_lst = []
                for c in conf_lst:
                    if c >= 80:
                        check_lst.append(True)
                    else:
                        check_lst.append(False)
                if False not in check_lst:
                    if len(cls_lst) == 2:  # 객체가 2개이고 자동 인증인 경우(HF + (OF, AF, BF))
                        if 2 in cls_lst:  # HF + OF
                            img_dict["authentication"] = imgDictTrue(source, img_name, conf, HFs_path)
                            label_lst.append(13)
                        elif 4 in cls_lst:  # HF + AF
                            img_dict["authentication"] = imgDictTrue(source, img_name, conf, HFs_path)
                            label_lst.append(13)
                        elif 5 in cls_lst:  # HF + BF
                            img_dict["authentication"] = imgDictWait(source, img_name, conf, HFs_path)
                            label_lst.append(13)
                        else:
                            img_dict["authentication"] = imgDictWait(source, img_name, conf, HFs_W_file_path)
                            label_lst.append(12)
                    else:
                        img_dict["authentication"] = imgDictWait(source, img_name, conf, HFs_W_file_path)
                        label_lst.append(12)
                else:
                    img_dict["authentication"] = imgDictWait(source, img_name, conf, None_file_path)
                    label_lst.append(11)
            else:  # 객체 중에 HF가 없는 경우
                img_dict["authentication"] = imgDict(source, img_name, conf, HFs_F_file_path)
                label_lst.append(10)
    return img_dict, cls_lst, label_lst, conf_lst
def HoneyAutoLabel(im0, sex, img_dict, cls_lst, label_lst, conf_lst, img_name, conf, percent_lst):
    global UB_file_path ,ID_file_path, None_file_path, HF_f_path, HF_m_path, HFU_file_path, NHF_file_path, CF_file_path, AF_file_path,\
    BF_file_path ,FF_file_path, HFs_path, HFs_W_file_path, HFs_F_file_path

    UB_file_path = honey_UB_file_path
    ID_file_path = honey_ID_file_path
    None_file_path = honey_None_file_path
    HF_f_path = honey_HF_f_path
    HF_m_path = honey_HF_m_path
    HFU_file_path = honey_HFU_file_path
    NHF_file_path = honey_NHF_file_path
    CF_file_path = honey_CF_file_path
    AF_file_path = honey_AF_file_path
    BF_file_path = honey_BF_file_path
    FF_file_path = honey_FF_file_path
    HFs_path = honey_HFs_path
    HFs_W_file_path = honey_HFs_W_file_path
    HFs_F_file_path = honey_HFs_F_file_path

    if 7 in cls_lst:  # 객체가 하나라도 UB일 경우
        img_dict, label_lst = HoneyUB_IDfinder(im0, 7, img_dict, cls_lst, label_lst, conf_lst, img_name, conf, 16, UB_file_path)

    elif 8 in cls_lst:  # 객체가 하나라도 ID일 경우
        img_dict, label_lst = HoneyUB_IDfinder(im0, 8, img_dict, cls_lst, label_lst, conf_lst, img_name, conf, 17, ID_file_path)
    else:
        if len(cls_lst) == 1:  # 객체가 1개인 경우
            if percent_lst[0] <= 0.15:
                img_dict["authentication"] = HoneyimgDictWait(im0, img_name, conf, None_file_path)
                label_lst.append(18)
            else:
                if conf >= 80:  # conf >= 80 인 경우
                    if int(cls_lst[0]) == 0:  # 라벨이 0(HF)인 경우
                        if sex == "f":  # 여자일 경우
                            img_dict["authentication"] = HoneyimgDictTrue(im0, img_name, conf, HF_f_path)
                            label_lst.append(1)
                        else:  # 남자일 경우
                            img_dict["authentication"] = HoneyimgDictTrue(im0, img_name, conf, HF_m_path)
                            label_lst.append(1)
                    else:  # 객체가 HF가 아닌 (HFU,NHF,CF,AF,BF,FF)인 경우
                        if int(cls_lst[0]) == 1:
                            img_dict["authentication"] = HoneyimgDict(im0, img_name, conf, HFU_file_path)
                            label_lst.append(9)
                        elif int(cls_lst[0]) == 2:
                            img_dict["authentication"] = HoneyimgDict(im0, img_name, conf, NHF_file_path)
                            label_lst.append(3)
                        elif int(cls_lst[0]) == 3:
                            img_dict["authentication"] = HoneyimgDict(im0, img_name, conf, CF_file_path)
                            label_lst.append(4)
                        elif int(cls_lst[0]) == 4:
                            img_dict["authentication"] = imgDict(im0, img_name, conf, AF_file_path)
                            label_lst.append(5)
                        elif int(cls_lst[0]) == 5:
                            img_dict["authentication"] = HoneyimgDictWait(im0, img_name, conf, BF_file_path)
                            label_lst.append(14)
                        elif int(cls_lst[0]) == 6:
                            img_dict["authentication"] = HoneyimgDict(im0, img_name, conf, FF_file_path)
                            label_lst.append(15)
                        else:
                            img_dict["authentication"] = HoneyimgDictWait(im0, img_name, conf, None_file_path)
                            label_lst.append(11)
                else:  # 50 < conf < 80 인 경우
                    img_dict["authentication"] = HoneyimgDictWait(im0, img_name, conf, None_file_path)
                    label_lst.append(11)
        else:  # 객체가 2개 이상인 경우
            if 0 in cls_lst:  # 객체 중에 HF가 있는 경우
                check_lst = []
                for c in conf_lst:
                    if c >= 80:
                        check_lst.append(True)
                    else:
                        check_lst.append(False)
                if False not in check_lst:
                    if len(cls_lst) == 2:  # 객체가 2개이고 자동 인증인 경우(HF + (OF, AF, BF))
                        if 2 in cls_lst:  # HF + OF
                            img_dict["authentication"] = HoneyimgDictTrue(im0, img_name, conf, HFs_path)
                            label_lst.append(13)
                        elif 4 in cls_lst:  # HF + AF
                            img_dict["authentication"] = HoneyimgDictTrue(im0, img_name, conf, HFs_path)
                            label_lst.append(13)
                        elif 5 in cls_lst:  # HF + BF
                            img_dict["authentication"] = HoneyimgDictWait(im0, img_name, conf, HFs_path)
                            label_lst.append(13)
                        else:
                            img_dict["authentication"] = HoneyimgDictWait(im0, img_name, conf, HFs_W_file_path)
                            label_lst.append(12)
                    else:
                        img_dict["authentication"] = HoneyimgDictWait(im0, img_name, conf, HFs_W_file_path)
                        label_lst.append(12)
                else:
                    img_dict["authentication"] = HoneyimgDictWait(im0, img_name, conf, None_file_path)
                    label_lst.append(11)
            else:  # 객체 중에 HF가 없는 경우
                img_dict["authentication"] = HoneyimgDict(im0, img_name, conf, HFs_F_file_path)
                label_lst.append(10)
    return img_dict, cls_lst, label_lst, conf_lst