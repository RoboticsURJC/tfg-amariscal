CPU_temp=$(cat /sys/class/thermal/thermal_zone1/temp)
GPU_temp=$(cat /sys/class/thermal/thermal_zone2/temp)

cpu=$((CPU_temp/1000))
gpu=$((GPU_temp/1000))

alias temp="source ~/.bashrc && echo CPU: $cpu GPU: $gpu"
