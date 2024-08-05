import conf.common as c
from conf.gunicorn_log import LOGGING_CONFIG
gconf = c.Gconf()

bind = gconf.get("bind")
workers = gconf.get("workers")
worker_class = gconf.get("worker_class")
pidfile = gconf.get("pidfile")
user: gconf.get("user")
group: gconf.get("group")
logconfig_dict = LOGGING_CONFIG



