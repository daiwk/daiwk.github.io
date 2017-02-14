from bs4 import BeautifulSoup

import sys
import os
import time
import traceback
import logging
import urllib2
import json

dic_server = {
    "idl_off": "http://xxxxxidl_off/job/",
    "idl_dl_cpu": "http://xxxxxidl_cpu/job/",
    "ecom_off": "http://xxxxxecom_off/job/", 
}

from argparse import ArgumentParser

def init_args():
    """
    """
    arg_parser = ArgumentParser(description="get mpi job status")

    arg_parser.add_argument("-j", "--jobid", dest="jobid", required=True, 
            help="jobid, if jobid = 288262.xxxxxx, set 288262")
    
    arg_parser.add_argument("-s", "--server", \
            dest="server", help="server: idl_off, ecom_off", required=True)
    args = None 

    try:
        args = arg_parser.parse_args()
    except Exception as e:
        logging.fatal(str(e))
 
    args = arg_parser.parse_args()

    return args


def get_page(url):
    """
    """
    data = urllib2.urlopen(url).read() 
    page_data = data.decode('GBK')
    soup = BeautifulSoup(page_data)
    return soup

def get_job_status(jobid, server):
    """
    """
    host = dic_server[server]
    url = host + "i-" + jobid
    print url
    while True:
        soup = get_page(url)
        trs = soup.find_all("tr")
        for tr in trs:
            if tr.th.string == "Exit Status":
                ret_status = tr.td.string
                if ret_status != "None":
                    return int(ret_status)
        time.sleep(60)
        print "not finished, continue querying status..."


if __name__ == "__main__":
    args = init_args()
    if args is None:
        sys.exit(1)
    jobstatus = get_job_status(args.jobid, args.server)
    exit(jobstatus)
