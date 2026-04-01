import os
from pathlib import Path

def generate_tree_with_contents(startpath, ignore_dirs=None, output_file=None):
    startpath = Path(startpath).resolve()
    if ignore_dirs is None:
        ignore_dirs = {"logs",".vscode",".idea",'.git', '__pycache__', 'node_modules', 'venv', 'dist', 'build',"reference_code","clip_dataset","nuscenes_mini","e2e_dataset_10Hz"}

    # 收集所有 .py 文件路径
    py_files = []

    # 用于打印树形结构的函数（递归）
    def _walk_tree(path, prefix=''):
        items = sorted(path.iterdir(), key=lambda x: (x.is_file(), x.name))
        # 过滤忽略文件夹
        items = [item for item in items if not (item.is_dir() and item.name in ignore_dirs)]
        for i, item in enumerate(items):
            is_last = (i == len(items) - 1)
            connector = '└── ' if is_last else '├── '
            print(f'{prefix}{connector}{item.name}')
            if item.is_dir():
                extension = '    ' if is_last else '│   '
                _walk_tree(item, prefix + extension)
            elif item.suffix == '.py':
                py_files.append(item)  # 记录 .py 文件

# 打印树形结构
    print(startpath.name + '/')
    _walk_tree(startpath)

    # 输出文件内容
    print('\n' + '='*60)
    print('文件内容列表：')
    print('='*60)
    for py_file in py_files:
        print(f'\n文件: {py_file.relative_to(startpath)}')
        print('-' * 40)
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()
            print(content)
        except Exception as e:
            print(f'无法读取文件: {e}')
        print('-' * 40)

# 使用示例
if __name__ == '__main__':
    generate_tree_with_contents('.', output_file=None)  # 可指定 output_file 将输出写入文件