
#!/bin/bash

wget -qO - https://www.mongodb.org/static/pgp/server-6.0.asc | sudo apt-key add -
/etc/apt/sources.list.d/mongodb-org-6.0.list
echo "deb [ arch=amd64,arm64 ] https://repo.mongodb.org/apt/ubuntu focal/mongodb-org/6.0 multiverse" | sudo tee /etc/apt/sources.list.d/mongodb-org-6.0.list
sudo apt-get update
sudo apt-get install -y mongodb-org

#### configs & data dirs
# /var/lib/mongodb
# /var/log/mongodb
# /etc/mongod.conf

#### running mongo
# sudo systemctl start mongod
# sudo systemctl status mongod
# sudo systemctl enable mongod
# sudo systemctl stop mongod
# sudo systemctl restart mongod