import shutil
# https://stackoverflow.com/questions/1855095/how-to-create-a-zip-archive-of-a-directory
def backup_project_as_zip(project_dir, zip_file):
    shutil.make_archive(zip_file.replace('.zip',''), 'zip', project_dir)
    pass