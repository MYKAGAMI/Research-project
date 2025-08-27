#!/bin/bash
# 最简化的一键提交 + 推送脚本
# 用法：bash update.sh "你的提交说明"

git add .
git commit -m "${1:-Update code}"
git push
