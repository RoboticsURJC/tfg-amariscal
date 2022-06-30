# Instrucciones VNC Jetson Nano

sudo apt install tigervnc-standalone-server

vncpasswd

cd ~/.vnc 

sudo vim xstartup

<pre>
!/bin/sh
export XDG_RUNTIME_DIR=/run/user/1000
export XKL_XMODMAP_DISABLE=1
unset SESSION_MANAGER
unset DBUS_SESSION_BUS_ADDRESS
xrdb /home/alvaro/.Xresources
xsetroot -solid grey
gnome-session &
</pre>

sudo chmod 755 ~/.vnc/xstartup

ls -al /home/alvaro/.Xresources

touch /home/alvaro/.Xresources

cd /etc/systemd/system

sudo vim vncserver@.service

<pre>
[Unit]
Description=Start TigerVNC Server at startup
After=syslog.target network.target

[Service]
Type=forking
User=alvaro
Group=alvaro
WorkingDirectory=/home/alvaro
PIDFile=/home/alvaro/.vnc/%H:%i.pid
ExecStartPre=-/usr/bin/vncserver -kill :%i > /dev/null 2>&1
ExecStart=/usr/bin/vncserver :%i -depth 24 -geometry 1920x1080 -nolisten tcp
ExecStop=/usr/bin/vncserver -kill :%i

[Install]
WantedBy=multi-user.target
</pre>

sudo vim /etc/vnc.conf
<pre>
localhost = "no";
</pre>

sudo vim /etc/gdm3/custom.conf
<pre>
AutomaticLoginEnable=true
AutomaticLogin=alvaro
</pre>

sudo systemctl daemon-reload 

sudo systemctl enable vncserver@1

sudo systemctl start vncserver@1 

Utilizando Remmina nos conectamos 192.168.1.XXX:5901

Fuente: https://forums.developer.nvidia.com/t/how-to-setup-tigervnc-on-jetson-nano/174244