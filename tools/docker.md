# Docker


- sudo
- service docker start
- docker images
- docker container ls
- docker ps -a
- systemctl start docker

---

- docker run --name [container-name] --shm-size=4g --network host -v $PWD:/master -it [image:tag]/[image-id] /bin/bash

---
- docker start [container-name]
- docker restart [container-name]
- docker attach [container-name]
- docker exec -it [container-name] /bin/bash

- exit / ctrl+D

---

- docker stop [id]
- docker rm [id]

---

- docker commit -a "user-name" -m "commit" [container-id] [image:tag]
