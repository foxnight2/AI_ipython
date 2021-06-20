# Linux commands


- find ./ -name "*.py" -or -name "*.cpp" | xargs grep "xxx" | wc 
- sort -r -t $'\t' -k 7

---
- nohup command &
- ps aux 

---
- command &
- jobs -l
- fg %num
- kill %num

---

- scp -r /path/to/file user@ip:/home/path/to/file
- scp -r user@ip:/home/path/to/file /path/to/file


---

 df -lh /


---

## tmux
- tmux new -s L
- tmux detach
- tmux attach -t L


## fuser
- apt install psmisc
- fuser -v /dev/nvidia*
- ps -aux | grep test_main.py  | awk '{print $2}' | uniq | xargs kill -9


## kill
- kill -9 pid
- pkill -9 python
