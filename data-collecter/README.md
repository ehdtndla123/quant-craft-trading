# 코인 관련 데이터를 수집하는 서버입니다.

# Requirements
- Python 3.9
- Ubuntu 20.04
- Kafka

# Installation
1. Install Python 3.9
2. Install Kafka
3. Install requirements
4. Install InfluxDB

### config.yaml 파일을 수정하시기 바랍니다.

```bash
sudo apt update && sudo apt upgrade -y


curl --silent --location -O \
https://repos.influxdata.com/influxdata-archive.key

echo "943666881a1b8d9b849b74caebf02d3465d6beb716510d86a39f6c8e8dac7515  influxdata-archive.key" \
| sha256sum --check - && cat influxdata-archive.key \
| gpg --dearmor \
| sudo tee /etc/apt/trusted.gpg.d/influxdata-archive.gpg > /dev/null \
&& echo 'deb [signed-by=/etc/apt/trusted.gpg.d/influxdata-archive.gpg] https://repos.influxdata.com/debian stable main' \
| sudo tee /etc/apt/sources.list.d/influxdata.list

sudo apt-get update && sudo apt-get install influxdb2

sudo apt install python3 python3-pip python3-venv -y

sudo systemctl start influxdb
sudo service influxdb status

influx setup

python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

# Start
```bash
python main.py
```


# Description
- 코인 관련 데이터를 수집하는 서버입니다.
- 코인 관련 데이터를 수집하여 Kafka에 전송합니다.