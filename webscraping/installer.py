import subprocess
import pkg_resources

packages = ["beautifulsoup4", 
            "requests", 
            "pip-system-certs", 
            "pyopenssl", 
            "selenium", 
            "webdriver-manager", 
            "datefinder", 
            "pygame", 
            "termcolor"]

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