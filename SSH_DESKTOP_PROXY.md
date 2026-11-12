# 通过台式机连接实验室服务器及转发代理

在发起 SSH 连接的电脑上，将以下配置添加到 `~/.ssh/config`：

```sshconfig
# 跳板：台式机
Host labdesktop
    HostName 100.73.198.24
    User Celestine
    IdentityFile ~/.ssh/id_rsa
    IdentitiesOnly yes

# 终点：实验室服务器
Host liumingzhuo
    HostName 10.160.22.114
    User root
    Port 20032
    ProxyJump labdesktop
    IdentityFile ~/.ssh/id_rsa
    IdentitiesOnly yes
    RemoteForward 127.0.0.1:17891 127.0.0.1:7897
    ServerAliveInterval 30
    ServerAliveCountMax 3
```

然后在配置所在的电脑上连接：

```bash
ssh liumingzhuo
```

连接经过 `labdesktop` 跳板。`RemoteForward` 让服务器上的 `127.0.0.1:17891` 转发到**发起该 SSH 连接的电脑**上的 `127.0.0.1:7897`；流量虽经台式机跳板，默认并不会因此转发到台式机的 7897 端口。若希望使用台式机代理，需要从台式机发起上述连接，或另行建立明确指向台式机代理的转发。

服务器上可对需要使用 HTTP 代理的单个命令设置：

```bash
HTTP_PROXY=http://127.0.0.1:17891 \
HTTPS_PROXY=http://127.0.0.1:17891 \
command
```

将 `command` 替换为实际命令。代理服务和对应 SSH 转发必须运行且可用；SSH 连接断开后，该转发也会停止。`ServerAliveInterval` 和 `ServerAliveCountMax` 用于检测失联连接。

可在服务器上检查代理到 W&B API 的连接：

```bash
curl --proxy http://127.0.0.1:17891 --connect-timeout 5 --max-time 15 \
  -o /dev/null -sS -w 'HTTP %{http_code}\n' https://api.wandb.ai/graphql
```

该 GET 请求返回 HTTP 405 表示已收到服务响应，但不代表训练数据上传一定成功；还需查看 W&B 运行状态和实际同步结果。不要将私钥或代理凭据写入文档或提交到仓库。
