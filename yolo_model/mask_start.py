import os
from conf.common import envName

# uvicorn.run("mask_filtering:app", host=uconf.get("host"), port=uconf.get("port"), access_log=False)
os.system(f"/home/yeoai/anaconda3/envs/{envName}/bin/gunicorn mask_filtering:app --config ./conf/gconf.py")