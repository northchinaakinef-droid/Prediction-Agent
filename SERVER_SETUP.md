# PredictionAgent 服务器准备清单

## 推荐购买

- 云厂商：腾讯云轻量应用服务器 Lighthouse。
- 地域：中国香港。
- 系统镜像：Ubuntu 24.04 LTS，x86_64；不要选 Windows 或预装网站应用镜像。
- 起步配置：2 vCPU、2 GB 内存、系统盘至少 60 GB、有公网 IPv4。
- 购买时长：先购买 1 个月，验证 Polymarket、Valve、Oracle 数据源和飞书连续运行后再续费。
- 不需要 GPU、数据库实例、域名、CDN、负载均衡或额外公网 IP。

选择香港节点是因为主要数据源在境外，且无需为这个只出站采集、健康接口仅绑定本机的服务办理 ICP 备案。腾讯云官方说明，中国香港及境外实例无需备案；套餐通常只能升级、不能降级，因此先选 2核2GB 更稳妥。

官方入口：

- https://cloud.tencent.com/product/lighthouse
- https://cloud.tencent.com/document/product/1207/44580
- https://cloud.tencent.com/document/product/243/18908

备选为阿里云轻量应用服务器中国香港，规格同样选 2核2GB/Ubuntu 24.04。购买页最终价格和库存可能变化：

- https://help.aliyun.com/zh/simple-application-server
- https://help.aliyun.com/zh/simple-application-server/product-overview/usage-notes

## 创建实例时的安全设置

1. 使用 SSH 密钥登录，不把 root 密码发给任何人。
2. 防火墙只开放 TCP 22；最好把来源限制为你当前公网 IP。
3. 不开放 8080 到公网。Docker 配置已经把健康接口绑定到 `127.0.0.1:8080`。
4. 开启云厂商登录告警、欠费/到期提醒和自动续费提醒。
5. 部署完成并稳定运行后创建一次系统盘快照。

## 购买后需要告诉我

只提供以下非敏感信息：

1. 云厂商：腾讯云或阿里云。
2. 实例公网 IP。
3. 系统版本，例如 Ubuntu 24.04。
4. 登录用户名，通常为 `ubuntu` 或 `root`。
5. 你希望采用哪种部署协作方式：
   - 你在云控制台终端执行我给出的命令；或
   - 你把 SSH 私钥保存在自己的电脑上，只告诉我本地密钥文件路径，不粘贴密钥内容。
6. 飞书使用“群自定义机器人”还是“自建应用私聊”。
7. 每天汇总推送时间；当前设定为北京时间 06:30。

## 不要在聊天中提供

- SSH 私钥或 root 密码。
- 飞书 App Secret、Webhook 完整地址或签名密钥。
- Polymarket 钱包私钥、助记词、API Secret。
- 云账号密码、短信验证码或支付信息。

飞书密钥会由你直接写进服务器的 `/opt/prediction-agent/.env`，文件权限设为 `600`，不会进入 Git、日报或聊天记录。本项目当前没有自动交易模块，部署也不需要任何钱包信息。

## 部署完成后的验收

1. Docker 容器状态为 `running`。
2. `curl http://127.0.0.1:8080/health` 返回最近扫描时间且无错误。
3. 每 30 分钟产生一次未来 30 小时的静默扫描。
4. `data/daily/paper.db` 持续增加有效前向记录，并自动尝试结算旧比赛。
5. 每天固定时间收到一次飞书汇总；没有合格赛事时明确显示 `NO_BET`。
6. NBA、LoL、CS2 的 `real_money_approved` 均保持 `false`，服务器不得自动下单。
