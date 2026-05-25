#!/usr/bin/env python3
"""
脚本用于将 Factory 资源从 ISAACLAB_NUCLEUS_DIR 下载到本地目录
"""
import os
import shutil
import urllib.request
from pathlib import Path
from urllib.error import URLError, HTTPError


# 首先启动 AppLauncher 以初始化 Isaac Sim 环境
from isaaclab.app import AppLauncher

# 创建 AppLauncher（使用 headless 模式，因为我们不需要图形界面）
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

# 现在可以安全地导入 isaaclab 模块
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

# ASSET_FOLDER_NAME = "AutoMate"
ASSET_FOLDER_NAME = "Factory"


def is_url(path_str):
    """检查路径是否是 URL"""
    return path_str.startswith(('http://', 'https://'))


def download_file(url, target_path):
    """从 URL 下载文件到目标路径"""
    try:
        # 确保目标目录存在
        target_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 修复 URL（处理可能的格式问题）
        if url.startswith('https:/') and not url.startswith('https://'):
            url = url.replace('https:/', 'https://', 1)
        elif url.startswith('http:/') and not url.startswith('http://'):
            url = url.replace('http:/', 'http://', 1)
        
        # 下载文件
        print(f"  下载中: {url}")
        urllib.request.urlretrieve(url, target_path)
        return True
    except HTTPError as e:
        print(f"  ✗ HTTP 错误 {e.code}: {e.reason}")
        return False
    except URLError as e:
        print(f"  ✗ URL 错误: {e.reason}")
        return False
    except Exception as e:
        print(f"  ✗ 发生错误: {e}")
        return False

# 定义需要下载的资源文件列表
FACTORY_ASSETS = [
    "factory_peg_8mm.usd",
    "factory_hole_8mm.usd",
    "factory_gear_base.usd",
    "factory_gear_medium.usd",
    "factory_gear_small.usd",
    "factory_gear_large.usd",
    "factory_nut_m16.usd",
    "factory_bolt_m16.usd",
    "franka_mimic.usd",
]

# FACTORY_ASSETS = [
#     '00004/socket.usd', '00004/plug.usd', '00007/socket.usd', '00007/plug.usd', '00014/socket.usd', '00014/plug.usd', 
#     '00015/socket.usd', '00015/plug.usd', '00016/socket.usd', '00016/plug.usd', '00021/socket.usd', '00021/plug.usd', 
#     '00028/socket.usd', '00028/plug.usd', '00030/socket.usd', '00030/plug.usd', '00032/socket.usd', '00032/plug.usd', 
#     '00042/socket.usd', '00042/plug.usd', '00062/socket.usd', '00062/plug.usd', '00074/socket.usd', '00074/plug.usd', 
#     '00077/socket.usd', '00077/plug.usd', '00078/socket.usd', '00078/plug.usd', '00081/socket.usd', '00081/plug.usd', 
#     '00083/socket.usd', '00083/plug.usd', '00103/socket.usd', '00103/plug.usd', '00110/socket.usd', '00110/plug.usd', 
#     '00117/socket.usd', '00117/plug.usd', '00133/socket.usd', '00133/plug.usd', '00138/socket.usd', '00138/plug.usd', 
#     '00141/socket.usd', '00141/plug.usd', '00143/socket.usd', '00143/plug.usd', '00163/socket.usd', '00163/plug.usd', 
#     '00175/socket.usd', '00175/plug.usd', '00186/socket.usd', '00186/plug.usd', '00187/socket.usd', '00187/plug.usd', 
#     '00190/socket.usd', '00190/plug.usd', '00192/socket.usd', '00192/plug.usd', '00210/socket.usd', '00210/plug.usd', 
#     '00211/socket.usd', '00211/plug.usd', '00213/socket.usd', '00213/plug.usd', '00255/socket.usd', '00255/plug.usd', 
#     '00256/socket.usd', '00256/plug.usd', '00271/socket.usd', '00271/plug.usd', '00293/socket.usd', '00293/plug.usd', 
#     '00296/socket.usd', '00296/plug.usd', '00301/socket.usd', '00301/plug.usd', '00308/socket.usd', '00308/plug.usd', 
#     '00318/socket.usd', '00318/plug.usd', '00319/socket.usd', '00319/plug.usd', '00320/socket.usd', '00320/plug.usd', 
#     '00329/socket.usd', '00329/plug.usd', '00340/socket.usd', '00340/plug.usd', '00345/socket.usd', '00345/plug.usd', 
#     '00346/socket.usd', '00346/plug.usd', '00360/socket.usd', '00360/plug.usd', '00388/socket.usd', '00388/plug.usd', 
#     '00410/socket.usd', '00410/plug.usd', '00417/socket.usd', '00417/plug.usd', '00422/socket.usd', '00422/plug.usd', 
#     '00426/socket.usd', '00426/plug.usd', '00437/socket.usd', '00437/plug.usd', '00444/socket.usd', '00444/plug.usd', 
#     '00446/socket.usd', '00446/plug.usd', '00470/socket.usd', '00470/plug.usd', '00471/socket.usd', '00471/plug.usd', 
#     '00480/socket.usd', '00480/plug.usd', '00486/socket.usd', '00486/plug.usd', '00499/socket.usd', '00499/plug.usd', 
#     '00506/socket.usd', '00506/plug.usd', '00514/socket.usd', '00514/plug.usd', '00537/socket.usd', '00537/plug.usd', 
#     '00553/socket.usd', '00553/plug.usd', '00559/socket.usd', '00559/plug.usd', '00581/socket.usd', '00581/plug.usd', 
#     '00597/socket.usd', '00597/plug.usd', '00614/socket.usd', '00614/plug.usd', '00615/socket.usd', '00615/plug.usd', 
#     '00638/socket.usd', '00638/plug.usd', '00648/socket.usd', '00648/plug.usd', '00649/socket.usd', '00649/plug.usd', 
#     '00652/socket.usd', '00652/plug.usd', '00659/socket.usd', '00659/plug.usd', '00681/socket.usd', '00681/plug.usd', 
#     '00686/socket.usd', '00686/plug.usd', '00700/socket.usd', '00700/plug.usd', '00703/socket.usd', '00703/plug.usd', 
#     '00726/socket.usd', '00726/plug.usd', '00731/socket.usd', '00731/plug.usd', '00741/socket.usd', '00741/plug.usd', 
#     '00755/socket.usd', '00755/plug.usd', '00768/socket.usd', '00768/plug.usd', '00783/socket.usd', '00783/plug.usd', 
#     '00831/socket.usd', '00831/plug.usd', '00855/socket.usd', '00855/plug.usd', '00860/socket.usd', '00860/plug.usd', 
#     '00863/socket.usd', '00863/plug.usd', '01026/socket.usd', '01026/plug.usd', '01029/socket.usd', '01029/plug.usd', 
#     '01036/socket.usd', '01036/plug.usd', '01041/socket.usd', '01041/plug.usd', '01053/socket.usd', '01053/plug.usd', 
#     '01079/socket.usd', '01079/plug.usd', '01092/socket.usd', '01092/plug.usd', '01102/socket.usd', '01102/plug.usd', 
#     '01125/socket.usd', '01125/plug.usd', '01129/socket.usd', '01129/plug.usd', '01132/socket.usd', '01132/plug.usd', 
#     '01136/socket.usd', '01136/plug.usd'
# ]


# 本地资源目录（相对于项目根目录）
LOCAL_ASSETS_DIR = Path(__file__).parent / "assets" / ASSET_FOLDER_NAME



def download_assets():
    """下载 Factory 资源到本地目录"""
    source_base = ISAACLAB_NUCLEUS_DIR
    target_dir = LOCAL_ASSETS_DIR
    
    print(f"源路径: {source_base}")
    print(f"目标目录: {target_dir}")
    print(f"ISAACLAB_NUCLEUS_DIR: {ISAACLAB_NUCLEUS_DIR}")
    
    # 检查是否是 URL
    is_cloud_source = is_url(source_base)
    
    if is_cloud_source:
        print("\n检测到云端资源，将从 URL 下载...")
        base_url = source_base.rstrip('/')
        factory_url = f"{base_url}/{ASSET_FOLDER_NAME}"
    else:
        source_dir = Path(source_base) / ASSET_FOLDER_NAME
        if not source_dir.exists():
            print(f"\n错误: 源目录不存在: {source_dir}")
            print("请确保 Isaac Lab 已正确安装，并且资源已下载到本地缓存")
            return False
        factory_url = None
    
    # 创建目标目录
    target_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n创建目标目录: {target_dir}")
    
    # 下载每个资源文件
    success_count = 0
    failed_files = []
    
    print("\n下载资源文件...")
    for asset_file in FACTORY_ASSETS:
        target_path = target_dir / asset_file
        
        # 如果文件已存在，跳过
        if target_path.exists():
            print(f"  ⊙ {asset_file} (已存在，跳过)")
            success_count += 1
            continue
        
        if is_cloud_source:
            # 从云端下载
            file_url = f"{factory_url}/{asset_file}"
            if download_file(file_url, target_path):
                print(f"  ✓ {asset_file}")
                success_count += 1
            else:
                failed_files.append(asset_file)
        else:
            # 从本地复制
            source_path = Path(source_base) / ASSET_FOLDER_NAME / asset_file
            if source_path.exists():
                try:
                    shutil.copy2(source_path, target_path)
                    print(f"  ✓ {asset_file}")
                    success_count += 1
                except Exception as e:
                    print(f"  ✗ {asset_file}: {e}")
                    failed_files.append(asset_file)
            else:
                print(f"  ✗ 源文件不存在: {asset_file}")
                failed_files.append(asset_file)
    
    # 检查是否有 USD 文件引用的其他资源（如纹理、材质等）
    # 注意：对于云端资源，我们无法列出目录，所以跳过这一步
    if not is_cloud_source:
        print("\n检查是否有其他依赖资源...")
        source_dir = Path(source_base) / ASSET_FOLDER_NAME
        if source_dir.exists():
            all_files = list(source_dir.rglob("*"))
            for file_path in all_files:
                if file_path.is_file():
                    relative_path = file_path.relative_to(source_dir)
                    # 跳过已经下载的文件
                    if relative_path.name in FACTORY_ASSETS:
                        continue
                    target_file = target_dir / relative_path
                    target_file.parent.mkdir(parents=True, exist_ok=True)
                    if not target_file.exists():
                        try:
                            shutil.copy2(file_path, target_file)
                            print(f"  ✓ 已复制依赖资源: {relative_path}")
                        except Exception as e:
                            print(f"  ✗ 复制依赖资源失败 {relative_path}: {e}")
    else:
        print("\n注意: 云端资源模式下，无法自动下载依赖资源")
    
    # 输出结果
    print(f"\n{'='*60}")
    print(f"下载完成!")
    print(f"成功: {success_count}/{len(FACTORY_ASSETS)}")
    if failed_files:
        print(f"失败的文件: {failed_files}")
    print(f"本地资源目录: {target_dir}")
    print(f"{'='*60}")
    
    return len(failed_files) == 0


if __name__ == "__main__":
    success = False
    try:
        success = download_assets()
        if success:
            print("\n下一步: 运行以下命令更新 ASSET_DIR:")
            print(f"  ./isaaclab.sh -p update_asset_dir.py")
        else:
            print("\n下载过程中出现错误，请检查上述输出")
    except Exception as e:
        print(f"\n发生错误: {e}")
        success = False
    finally:
        # 关闭 simulation app
        simulation_app.close()
    exit(0 if success else 1)
