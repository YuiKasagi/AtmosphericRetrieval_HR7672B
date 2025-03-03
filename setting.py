import socket
def set_path():
    """
    if(socket.gethostname()) == 'manbou':
        path_obs="/home/kawashima/exojax/calc/obs_data"
        path_data="/home/kawashima/database"
        path_repo="/home/kawashima/exojax/github"
    elif(socket.gethostname()) == 'higuma':
        path_obs="."
        path_data="/home/kawashima/database"
        path_repo="/home/kawashima/ExoJAX/github"
    else:
    """
    path_obs="/home/yuikasagi/Develop/exojax/data"
    path_data="/home/yuikasagi/Develop/exojax/database"
#    path_repo="/home/kawashimayi/exojax/github"
    path_gpdata = "/home/yuikasagi/Develop/exojax/figure"
    path_save="/home/yuikasagi/Develop/exojax/output"

    return path_obs, path_data, path_gpdata, path_save


"""
import git, datetime
import os
import subprocess
def git_install(path_repo, branch):
    repo = git.Repo(path_repo)
    repo.git.checkout(branch)

    print('******************************')
    print("VERSION INFORMATION")
    print("branch:", repo.active_branch.name)
    for item in repo.iter_commits(repo.active_branch.name, max_count=1):
        dt = datetime.datetime.fromtimestamp(item.authored_date).strftime("%Y-%m-%d %H:%M:%S")
        print("commit:", item.hexsha)
        print("Author:", item.author)
        print("Date:", dt)
        print(item.message)
    print('******************************')

    calc_dir = os.getcwd()
    os.chdir(path_repo)

    log = subprocess.check_output(["python", "setup.py", "install"]).decode()
    if "Finished" in log:
        print("Successfully installed")
    else:
        print("Fail to install exojax")

    os.chdir(calc_dir)
"""
