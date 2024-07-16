import socket

HOST = '127.0.0.1'  # 服务器地址
PORT = 65432        # 服务器端口

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((HOST, PORT))
    s.listen()
    conn, addr = s.accept()
    with conn:
        print('Connected by', addr)
        while True:
            data = conn.recv(1024).decode()
            if not data:
                break
            print('Received', data)
            conn.sendall(data.upper().encode())