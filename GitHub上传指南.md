# GitHub 上传指南

## 📋 前提条件

1. 已安装 Git
2. 已创建 GitHub 账号
3. 项目已初始化 Git 并完成首次提交（✅ 已完成）

## 🚀 上传步骤

### 方法一：使用 GitHub 网页创建仓库（推荐新手）

#### 步骤 1: 在 GitHub 上创建新仓库

1. 登录 [GitHub](https://github.com)
2. 点击右上角的 **"+"** 按钮，选择 **"New repository"**
3. 填写仓库信息：
   - **Repository name**: `machine-translation` 或 `机器翻译`（建议使用英文）
   - **Description**: 可选，例如 "基于Transformer+GCN的机器翻译项目"
   - **Visibility**: 选择 Public（公开）或 Private（私有）
   - **⚠️ 重要**: **不要**勾选 "Initialize this repository with a README"（因为本地已有代码）
4. 点击 **"Create repository"**

#### 步骤 2: 连接本地仓库到 GitHub

在项目目录下执行以下命令（将 `YOUR_USERNAME` 替换为你的 GitHub 用户名，`REPO_NAME` 替换为仓库名）：

```powershell
# 添加远程仓库地址
git remote add origin https://github.com/YOUR_USERNAME/REPO_NAME.git

# 推送代码到 GitHub
git branch -M main
git push -u origin main
```

**示例**：
```powershell
git remote add origin https://github.com/zhangsan/machine-translation.git
git branch -M main
git push -u origin main
```

#### 步骤 3: 输入 GitHub 凭证

- 如果使用 HTTPS，会提示输入用户名和密码
- **密码**需要使用 **Personal Access Token**（不是 GitHub 登录密码）
  - 生成 Token: GitHub → Settings → Developer settings → Personal access tokens → Tokens (classic) → Generate new token
  - 权限选择：至少勾选 `repo` 权限

---

### 方法二：使用 SSH（推荐，更安全）

#### 步骤 1: 生成 SSH 密钥（如果还没有）

```powershell
# 检查是否已有 SSH 密钥
ls ~/.ssh

# 如果没有，生成新的 SSH 密钥
ssh-keygen -t ed25519 -C "your_email@example.com"
# 按 Enter 使用默认路径，可以设置密码或直接回车
```

#### 步骤 2: 添加 SSH 密钥到 GitHub

```powershell
# 复制公钥内容
cat ~/.ssh/id_ed25519.pub
# 或 Windows PowerShell:
Get-Content ~/.ssh/id_ed25519.pub
```

1. 复制输出的公钥内容
2. 登录 GitHub → Settings → SSH and GPG keys → New SSH key
3. 粘贴公钥，添加标题，保存

#### 步骤 3: 使用 SSH 地址连接

```powershell
# 添加远程仓库（使用 SSH 地址）
git remote add origin git@github.com:YOUR_USERNAME/REPO_NAME.git

# 推送代码
git branch -M main
git push -u origin main
```

---

### 方法三：使用 GitHub CLI（最简单）

#### 步骤 1: 安装 GitHub CLI

访问 [GitHub CLI 官网](https://cli.github.com/) 下载安装

#### 步骤 2: 登录并创建仓库

```powershell
# 登录 GitHub
gh auth login

# 在项目目录下创建并推送仓库
gh repo create --public --source=. --remote=origin --push
```

---

## 🔄 后续更新代码

上传后，如果修改了代码，使用以下命令更新 GitHub：

```powershell
# 查看修改的文件
git status

# 添加修改的文件
git add .

# 提交修改
git commit -m "描述你的修改内容"

# 推送到 GitHub
git push
```

---

## ❓ 常见问题

### Q1: 提示 "remote origin already exists"

**解决方案**：
```powershell
# 查看现有远程仓库
git remote -v

# 删除现有远程仓库
git remote remove origin

# 重新添加
git remote add origin https://github.com/YOUR_USERNAME/REPO_NAME.git
```

### Q2: 推送时提示认证失败

**解决方案**：
- 使用 Personal Access Token 代替密码
- 或使用 SSH 方式连接

### Q3: 分支名称冲突（master vs main）

**解决方案**：
```powershell
# 重命名本地分支为 main
git branch -M main

# 推送时指定分支
git push -u origin main
```

### Q4: 想忽略某些文件但已经提交了

**解决方案**：
```powershell
# 从 Git 中删除但保留本地文件
git rm --cached 文件名

# 提交删除
git commit -m "Remove file from git"

# 推送到 GitHub
git push
```

---

## 📝 快速命令参考

```powershell
# 查看远程仓库
git remote -v

# 查看提交历史
git log --oneline

# 查看当前状态
git status

# 添加所有文件
git add .

# 提交
git commit -m "提交信息"

# 推送
git push

# 拉取更新
git pull
```

---

## ✅ 完成检查

上传成功后，你应该能够：
1. 在 GitHub 上看到你的仓库
2. 看到所有项目文件
3. 看到 README.md 正确显示

**恭喜！你的项目已成功上传到 GitHub！🎉**






