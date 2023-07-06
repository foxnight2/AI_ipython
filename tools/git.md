
## ssh
- ls ~/.ssh/
- ssh-keygen -t rsa -b 2048 -C "xxx@gmail.com"
- eval "$(ssh-agent -s)"
- ssh-add ~/.ssh/id_rsa
---
- ssh-keygen -t rsa -C "user.email"
- cat ~/.ssh/id_rsa.pub
- ssh -T git@github.com


## config 
- --global
- git config user.name "xxx"
- git config user.email "xxx"
- git config --global core.editor "vim"

## folk
- sync with orginal repo
- git folk 
- git clone xxx
- git remote add upstream git@github.com:PaddlePaddle/Paddle.git
- git remote -v
- git remote remove upstream

---
- git fetch upstream
- git pull upstream develop
---
- git branch -r
- git checkout -b newbranch upstream/branchname
- git checkout -b newbranch
---
- git checkout originlocalbranch
- git diff newbranch
- git merge newbranch
- git push 

---
- git push origin newbranch:newbranch
- git push origin :newbranch

## branch
- git checkout xxx
- git branch -d xxx

## tag
- git tag xx
- git tag -a xx -m 'xx'
- git push origin xx
- git show xx
---
- git tag -d xx
- git push origin :xx
---
- git checkout -b [new-brach-name] [tag-name]

## commit
- git commit -m ""
- git commit --amend


## pull
- Pull is not possible because you have unmerged files.
- git reset --hard FETCH_HEAD
- git pull


## detach
- git branch -b tmp
- git checkout dev
- git merge tmp
- git push /dev


## cherry-pick
 - git cherry-pick A^...B  [A, B]
 - git cherry-pick --continue
 - git cherry-pick --abort


 


## log
- git log --raw
- git log --patch
- git log --oneline
- git log --pretty=format:'%Cred%h%Creset -%C(yellow)%d%Creset %s %Cgreen(%cd) %C(bold blue)<author:%an>%Creset' v1.0..


## stash
- Git stash saves the uncommitted changes locally (.git/refs/stash), allowing you to make changes, switch branches, and perform other Git operations.
- save on branch A
- git stash [save [info]]
- git checkout B
- edit -> add -> commit -> push 
- git checkout A
- git stash pop [stash@{0}]
- git stash list
- git stash clear
- [git-stash](https://opensource.com/article/21/4/git-stash)
	


# pre-commit
- pip install pre-commit
- cd /repo
- pre-commit install
- git add . 
- git commit -m 'update'


# develop
- git remote add upstream git@github.com:PaddlePaddle/Paddle.git
- git remote add upstream https://github.com/PaddlePaddle/PaddleDetection.git
- git remote rm upstream
- git remote -v
- git fetch upstream [|develop]
- git checkout -b newbranch-1 upstream/develop
- edit -> add -> commit [hash-id]
- git checkout -b newbranch-2 upstream/release
- git cherry-pick [hash-id]
- git push origin newbranch-1:newbranch-1
- git push origin newbranch-2:newbranch-2


# remote
- git remote add xxx https://github.com/xxx/Paddle.git 
- git fetch xxx 
- git checkout -b [new-name] [xxx/branch]



# submodules
- git submodule add url [path/name]
- git submodule init
- git submodule update
- git submodule update --init --recursive
- git submodule foreach git pull
- git submodule add https://github.com/IDEA-Research/GroundingDINO.git uaig/grounding_dino



# github
- https://shields.io/badges
- https://star-history.com/