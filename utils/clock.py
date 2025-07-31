import subprocess
import time
import yagmail
import os
import signal
import socket
def run_squeue_command():
    # 定义要执行的命令
    node =str(socket.gethostname())
    command = ['squeue', '-o', '"%.18i %.9P %.8j %.8u %.2t %.10M %.6D %C %R %b"', '-w', node]

    try:
        # 使用 subprocess.run 执行命令，并捕获输出
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        # 返回命令的输出结果
        return result.stdout
    except subprocess.CalledProcessError as e:
        # 处理执行命令时的错误
        print("Error executing command:", e)
        return None

def check_users(output):
    # 将输出按行分割
    lines = output.strip().split("\n")
    # 提取用户栏的用户名
    users = [line.split()[4] for line in lines[1:]]  # 跳过标题行
    # 检查是否只有 wenlab 用户
    return all(user == "wenlab" for user in users)

def send_email(to_email):
    # 设置邮箱和授权码
    email = 'jack_ucla@163.com'
    password = 'ONLYCRSHFWTYZZTO'

    # 初始化yagmail.SMTP对象
    yag = yagmail.SMTP(user=email, password=password, host='smtp.163.com')

    # 邮件内容
    subject = 'Non-wenlab user detected'
    contents = 'A non-wenlab user has been detected in the squeue output.'

    try:
        # 发送邮件
        yag.send(to=to_email, subject=subject, contents=contents)
        print("Email sent successfully")
    except Exception as e:
        print("Failed to send email:", e)
def kill_processes(pids):
    """
    用于批量结束进程
    """
    for pid in pids:
        try:
            os.kill(int(pid), signal.SIGTERM)
            print(f"Process {pid} has been terminated.")
        except OSError as e:
            print(f"Error: {e.strerror}. Process {pid} could not be terminated.")
        except ValueError:
            print(f"Invalid PID: {pid}. It must be an integer.")


def get_wenlab_gpu_processes():
    # 获取当前所有使用GPU的进程信息
    result = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], stdout=subprocess.PIPE,
                            text=True)

    # 提取所有使用GPU的进程ID
    gpu_pids = result.stdout.strip().split('\n')

    wenlab_processes = []

    for pid in gpu_pids:
        # 使用ps命令获取进程的用户名
        ps_result = subprocess.run(['ps', '-o', 'user=', '-p', pid], stdout=subprocess.PIPE, text=True)
        user = ps_result.stdout.strip()

        # 如果用户名是wenlab，则记录该进程ID
        if user == 'wenlab':
            wenlab_processes.append(pid)

    return wenlab_processes
#获取GPU进程

if __name__ == "__main__":
    to_email = 'hefengcs@gmail.com'  # 收件人的电子邮件地址
    while True:
        output = run_squeue_command()
        if output:
            result = check_users(output)
            if not result:
                send_email(to_email)
                #获取wenlab用户的所有GPU进程号
                wenlab_gpu_processes = get_wenlab_gpu_processes()
                print("Wenlab user's GPU processes:", wenlab_gpu_processes)
                #结束所有wenlab用户的GPU进程
                kill_processes(wenlab_gpu_processes)

                break
        print(str(socket.gethostname()) +" Good! No non-wenlab users detected.")
        time.sleep(5)  # 每5秒查询一次
