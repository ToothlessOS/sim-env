import socket

HOST = '127.0.0.1'  # 服务器地址
PORT = 65432        # 服务器端口

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.connect((HOST, PORT))
    s.sendall('Hello, world!'.encode())
    data = s.recv(1024)

print('Received', data.decode())