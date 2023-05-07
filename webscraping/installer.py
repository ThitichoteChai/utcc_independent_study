import subprocess
import pkg_resources
import os

packages = ["beautifulsoup4", 
            "requests", 
            "pip-system-certs", 
            "pyopenssl", 
            "selenium", 
            "webdriver-manager", 
            "datefinder", 
            "pygame", 
            "termcolor"
            #,"logging"
           ]

def install_packages(packages):
    for package in packages:
        try:
            pkg_resources.get_distribution(package)
            print("Package '{}' is already installed.".format(package))
        except pkg_resources.DistributionNotFound:
            result = subprocess.run(["pip", "install", package], capture_output=True)
            if result.returncode != 0:
                print("Error installing package '{}':".format(package))
                print(result.stderr.decode('utf-8'))
                return False
            else:
                print("Package '{}' was installed successfully.".format(package))
    return True

bool_package = install_packages(packages)

import requests
import time
import csv
import requests
import re
import logging
import datetime
import pygame
import pandas as pd
import numpy as np
import lxml.html as lh
from time import sleep
from termcolor import colored
from datetime import datetime
from time import localtime, strftime
from selenium import webdriver
from colorama import Fore, Back, Style
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.common.keys import Keys