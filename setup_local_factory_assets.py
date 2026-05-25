#!/usr/bin/env python3
"""
一体化脚本：下载 Factory 资源到本地并更新 ASSET_DIR
使用方法: ./isaaclab.sh -p setup_local_factory_assets.py
"""
import os
import shutil
import urllib.request
import urllib.parse
from pathlib import Path
from urllib.error import URLError, HTTPError

# 首先启动 AppLauncher 以初始化 Isaac Sim 环境
from isaaclab.app import AppLauncher

# 创建 AppLauncher（使用 headless 模式，因为我们不需要图形界面）
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

# 现在可以安全地导入 isaaclab 模块
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

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

# 配置文件路径
CONFIG_FILE = Path(__file__).parent / "source" / "tacex_tasks" / "tacex_tasks" / "factory" / "factory_tasks_cfg.py"

# 本地资源目录（相对于项目根目录）
LOCAL_ASSETS_DIR = Path(__file__).parent / "assets" / "Factory"


def is_url(path_str):
    """检查路径是否是 URL"""
    return path_str.startswith(('http://', 'https://'))


def download_file(url, target_path, show_progress=True):
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
        if show_progress:
            print(f"  下载中: {url}")
        
        # 使用 urllib 下载
        def show_progress_hook(count, block_size, total_size):
            if total_size > 0:
                percent = int(count * block_size * 100 / total_size)
                print(f"\r  进度: {percent}%", end='', flush=True)
        
        if show_progress:
            urllib.request.urlretrieve(url, target_path, reporthook=show_progress_hook)
            print()  # 换行
        else:
            urllib.request.urlretrieve(url, target_path)
        
        return True
    except HTTPError as e:
        print(f"  ✗ HTTP 错误 {e.code}: {e.reason}")
        if e.code == 404:
            print(f"    文件不存在: {url}")
        return False
    except URLError as e:
        print(f"  ✗ URL 错误: {e.reason}")
        return False
    except Exception as e:
        print(f"  ✗ 发生错误: {e}")
        return False


def download_assets():
    """下载 Factory 资源到本地目录"""
    source_base = ISAACLAB_NUCLEUS_DIR
    target_dir = LOCAL_ASSETS_DIR
    
    print(f"{'='*60}")
    print("步骤 1: 下载 Factory 资源")
    print(f"{'='*60}")
    print(f"源路径: {source_base}")
    print(f"目标目录: {target_dir}")
    print(f"ISAACLAB_NUCLEUS_DIR: {ISAACLAB_NUCLEUS_DIR}")
    
    # 检查是否是 URL
    is_cloud_source = is_url(source_base)
    
    if is_cloud_source:
        print("\n检测到云端资源，将从 URL 下载...")
        # 构建基础 URL
        base_url = source_base.rstrip('/')
        factory_url = f"{base_url}/Factory"
    else:
        # 本地路径
        source_dir = Path(source_base) / "Factory"
        if not source_dir.exists():
            print(f"\n错误: 源目录不存在: {source_dir}")
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
            source_path = Path(source_base) / "Factory" / asset_file
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
    # USD 文件中的相对路径引用应该仍然可以工作
    if not is_cloud_source:
        print("\n检查并复制依赖资源...")
        source_dir = Path(source_base) / "Factory"
        if source_dir.exists():
            all_files = list(source_dir.rglob("*"))
            dependency_count = 0
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
                            dependency_count += 1
                            if dependency_count <= 10:  # 只显示前10个
                                print(f"  ✓ {relative_path}")
                        except Exception as e:
                            print(f"  ✗ {relative_path}: {e}")
            if dependency_count > 10:
                print(f"  ... 共复制 {dependency_count} 个依赖资源")
    else:
        print("\n注意: 云端资源模式下，无法自动下载依赖资源")
        print("如果 USD 文件引用了其他资源，可能需要手动下载或修改路径")
    
    # 输出结果
    print(f"\n下载结果: 成功 {success_count}/{len(FACTORY_ASSETS)}")
    if failed_files:
        print(f"失败的文件: {failed_files}")
        return False
    
    return True


def update_asset_dir():
    """更新配置文件中的 ASSET_DIR"""
    print(f"\n{'='*60}")
    print("步骤 2: 更新 ASSET_DIR 配置")
    print(f"{'='*60}")
    
    if not CONFIG_FILE.exists():
        print(f"错误: 配置文件不存在: {CONFIG_FILE}")
        return False
    
    # 检查本地资源目录是否存在
    if not LOCAL_ASSETS_DIR.exists():
        print(f"错误: 本地资源目录不存在: {LOCAL_ASSETS_DIR}")
        return False
    
    # 获取绝对路径
    abs_local_dir = LOCAL_ASSETS_DIR.resolve()
    
    # 读取配置文件
    with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 替换 ASSET_DIR 的定义
    old_pattern = 'ASSET_DIR = f"{ISAACLAB_NUCLEUS_DIR}/Factory"'
    new_line = f'ASSET_DIR = r"{abs_local_dir}"'
    
    if old_pattern in content:
        content = content.replace(old_pattern, new_line)
        print(f"✓ 已更新 ASSET_DIR")
        print(f"  旧值: {old_pattern}")
        print(f"  新值: {new_line}")
    else:
        # 检查是否已经更新过
        lines = content.split('\n')
        updated = False
        for i, line in enumerate(lines):
            if line.strip().startswith('ASSET_DIR ='):
                old_line = line.strip()
                # 保持原有的缩进
                indent = len(line) - len(line.lstrip())
                new_line_with_indent = ' ' * indent + f'ASSET_DIR = r"{abs_local_dir}"'
                lines[i] = new_line_with_indent
                content = '\n'.join(lines)
                print(f"✓ 已更新 ASSET_DIR")
                print(f"  旧值: {old_line}")
                print(f"  新值: {new_line_with_indent}")
                updated = True
                break
        
        if not updated:
            print("警告: 未找到 ASSET_DIR 定义")
            return False
    
    # 写回文件
    with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"\n✓ 配置文件已更新: {CONFIG_FILE}")
    return True


def main():
    """主函数"""
    print("\n" + "="*60)
    print("Factory 资源本地化设置")
    print("="*60)
    
    # 步骤1: 下载资源
    if not download_assets():
        print("\n下载资源失败，请检查错误信息")
        return False
    
    # 步骤2: 更新配置
    if not update_asset_dir():
        print("\n更新配置失败，请检查错误信息")
        return False
    
    # 完成
    print(f"\n{'='*60}")
    print("✓ 设置完成!")
    print(f"{'='*60}")
    print(f"本地资源目录: {LOCAL_ASSETS_DIR.resolve()}")
    print(f"配置文件: {CONFIG_FILE}")
    print("\n现在可以在运行时直接使用本地资源，无需从云端下载")
    return True


if __name__ == "__main__":
    success = False
    try:
        success = main()
    except Exception as e:
        print(f"\n发生错误: {e}")
        success = False
    finally:
        # 关闭 simulation app
        simulation_app.close()
    exit(0 if success else 1)
