### Monitorizar temperatura de la Jetson Nano

Metemos el siguiente código en el .bashrc:

<pre>
CPU_temp=$(cat /sys/class/thermal/thermal_zone1/temp)
GPU_temp=$(cat /sys/class/thermal/thermal_zone2/temp)

cpu=$((CPU_temp/1000))
gpu=$((GPU_temp/1000))

alias temp="source ~/.bashrc && echo CPU: $cpu GPU: $gpu"
</pre>

Esto es debido a que hemos sufrido algunos apagados repentinos cuando estamos usando TensorFlow. Esto me ocurre cuando ejecuto el programa varias veces.

Vemos que la temperatura no es muy alta entorno a 45 grados tanto CPU como GPU
