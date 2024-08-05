import getpass
import os
import shutil
import warnings
warnings.filterwarnings(action='ignore')
##############################################################################################
# PATH configure
##############################################################################################
# user name ex) local : "PC", server : "yeoai"
userName = getpass.getuser()
startPath = os.path.dirname(os.path.realpath(__file__)) # start_api file Execution Path
localUserPath = f'C:/Users/{userName}'
serverUserPath = f'/home/{userName}'

# ENV name
envName = "mask_torch"
# api name
apiName = "ai_mask_yolo"
# model name
modelName = '9label_profile_model.pt'
# server data conf
serverApiDataPath = "mask_api_data" # data path
serverImgForderPath = 'photo_real/yeo_photo' # img file path
dev_serverImgForderPath = 'photo_real/yeo_photo_dev' # img file dev server path

# result data save path
learning_data_path = serverUserPath + '/learning_data/mask_filter/yolo'
HF_f_path = learning_data_path + '/HF/female'
HF_m_path = learning_data_path + '/HF/male'
HFs_path = learning_data_path + '/HFs' # HFs 추가
HFU_file_path = learning_data_path + '/HFU'
HFs_W_file_path = learning_data_path + '/HFs_W'
HFs_F_file_path = learning_data_path + '/HFs_F'
NHF_file_path = learning_data_path + '/NHF'
None_file_path = learning_data_path + '/None'
NoDetect_file_path = learning_data_path + '/NoDetect'
CF_file_path = learning_data_path + '/CF'
AF_file_path = learning_data_path + '/AF'
BF_file_path = learning_data_path + '/BF' # BF 추가
FF_file_path = learning_data_path + '/FF' # FF 추가
UB_file_path = learning_data_path + '/UB' # UB 추가
ID_file_path = learning_data_path + '/ID' # ID 추가

# honey result data save path
honey_learning_data_path = serverUserPath + '/learning_data/mask_filter/honey'
honey_HF_f_path = honey_learning_data_path + '/HF/female'
honey_HF_m_path = honey_learning_data_path + '/HF/male'
honey_HFs_path = honey_learning_data_path + '/HFs' # HFs 추가
honey_HFU_file_path = honey_learning_data_path + '/HFU'
honey_HFs_W_file_path = honey_learning_data_path + '/HFs_W'
honey_HFs_F_file_path = honey_learning_data_path + '/HFs_F'
honey_NHF_file_path = honey_learning_data_path + '/NHF'
honey_None_file_path = honey_learning_data_path + '/None'
honey_NoDetect_file_path = honey_learning_data_path + '/NoDetect'
honey_CF_file_path = honey_learning_data_path + '/CF'
honey_AF_file_path = honey_learning_data_path + '/AF'
honey_BF_file_path = honey_learning_data_path + '/BF' # BF 추가
honey_FF_file_path = honey_learning_data_path + '/FF' # FF 추가
honey_UB_file_path = honey_learning_data_path + '/UB' # UB 추가
honey_ID_file_path = honey_learning_data_path + '/ID' # ID 추가
##############################################################################################
# Configure Class
##############################################################################################
class BasicConfig:
    # port
    port = 8000
    # log path
    serverLogPath = "LOGS" + '/' + apiName + '_api'
    logPath = serverUserPath + '/' + serverLogPath + '/' + apiName
    # ai_data path - file load path
    serverModelPath = serverUserPath + '/ai_data/' + serverApiDataPath
    imgForderPath = serverUserPath + '/' + serverImgForderPath
    yoloModelPath = serverModelPath + '/' + modelName
class DevConfig():
    # port
    port = 8800
    # log path
    serverDevLogPath = "LOGS_devel" + '/' + apiName + '_api'
    logPath = serverUserPath + '/' + serverDevLogPath + '/' + apiName
    # ai_data path - file load path
    serverModelPath = serverUserPath + '/ai_data/' + serverApiDataPath
    imgForderPath = serverUserPath + '/' + dev_serverImgForderPath
    yoloModelPath = serverModelPath + '/' + modelName

##############################################################################################
# 실행 위치에 따른 api 가동
serverPath = serverUserPath + "/API/" + apiName + "_api/conf"
serverDevPath = serverUserPath + "/API_devel/" + apiName + "_api/conf"
serverDevTestPath = serverUserPath + "/API_devel/" + apiName + "_api_test/conf"

if userName == "yeoai":
    if startPath == serverPath:
        config = BasicConfig()
    elif startPath == serverDevPath:
        config = DevConfig()

##############################################################################################
# Uvicorn configure - 유비콘 사용시
##############################################################################################
def Uconf():
    uconf = {
        "host" : "0.0.0.0",
        "port" : config.port
    }
    if startPath == serverDevPath:
        uconf['port'] = config.port
    elif startPath == serverDevTestPath:
        uconf['port'] = config.port
    return uconf

##############################################################################################
# Gunicorn configure - 구니콘 사용시
##############################################################################################
# gunicorn pid path
pidFilePath = f"/var/run/yeoai/{apiName}.pid"
pidFilePath_dev = f"/var/run/yeoai/{apiName}_dev.pid"
pidFilePath_test = f"/var/run/yeoai/{apiName}_dev_test.pid"

def Gconf():
    #a15, a16 gconf
    gconf = {"bind": f"0.0.0.0:{config.port}",
             "workers" : 1,
             "worker_class" : "uvicorn.workers.UvicornWorker",
             "pidfile" : pidFilePath,
             "user" : 1000,
             "group" : 1000
     }
    if startPath == serverDevPath:
        # a15_dev
        gconf['bind'] = f"0.0.0.0:{config.port}"
        gconf['workers'] = 1
        gconf['pidfile'] = pidFilePath_dev

    elif startPath == serverDevTestPath:
        # a15_dev_test
        gconf['bind'] = f"0.0.0.0:{config.port}"
        gconf['workers'] = 1
        gconf['pidfile'] = pidFilePath_test

    return gconf














