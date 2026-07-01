import xnat
import getpass

# ria_pull.py
# Author: Sjors
# Description: Script to connect to the RIA server and pull MRI data for selected patients.

# # Ask for user credentials
# username = input("Enter your XNAT username: ")
# password = getpass.getpass("Enter your XNAT password: ")

# # Setup connection to RIA server via xnat api
# with xnat.connect('https://ria.ds.umcutrecht.nl', user=username, password=password) as session:
#     project = session.projects['18-109_psd']
#     subjects = project.subjects

#     print(f"Connected to RIA server and found {len(subjects)} subjects.")